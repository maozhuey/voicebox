import { getVersion } from '@tauri-apps/api/app';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { emit, listen } from '@tauri-apps/api/event';
import { open, save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';
import { relaunch } from '@tauri-apps/plugin-process';
import { open as openExternal } from '@tauri-apps/plugin-shell';
import { check, type Update } from '@tauri-apps/plugin-updater';
import type {
  Platform,
  PlatformAudio,
  PlatformFilesystem,
  PlatformLifecycle,
  PlatformUpdater,
  ServerLogEntry,
  UpdateStatus,
} from './types';

const INITIAL_UPDATE_STATUS: UpdateStatus = {
  checking: false,
  available: false,
  downloading: false,
  installing: false,
  readyToInstall: false,
};

function unsupported(operation: string): Promise<never> {
  return Promise.reject(new Error(`${operation} 仅在 Voicebox 桌面应用中可用。`));
}

function downloadFile(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function createBrowserPlatform(): Platform {
  const filesystem: PlatformFilesystem = {
    // 网页开发模式只负责呈现与联调，导出仍应产生可下载文件，避免因桌面文件对话框不可用而中断业务流程。
    async saveFile(filename, blob) {
      downloadFile(filename, blob);
    },
    openPath: () => unsupported('打开本地目录'),
    pickDirectory: () => unsupported('选择本地目录'),
  };

  const updater: PlatformUpdater = {
    checkForUpdates: () => unsupported('检查更新'),
    downloadAndInstall: () => unsupported('下载更新'),
    restartAndInstall: () => unsupported('安装更新'),
    getStatus: () => INITIAL_UPDATE_STATUS,
    subscribe: () => () => {},
  };

  const audio: PlatformAudio = {
    isSystemAudioSupported: async () => false,
    startSystemAudioCapture: () => unsupported('系统音频采集'),
    stopSystemAudioCapture: () => unsupported('系统音频采集'),
    listOutputDevices: () => unsupported('列出音频输出设备'),
    playToDevices: () => unsupported('多设备播放'),
    stopPlayback: () => {},
  };

  const lifecycle: PlatformLifecycle = {
    // 浏览器不会管理本地 sidecar；MainApp 会连接开发者已启动的本地服务。
    startServer: () => unsupported('启动本地服务'),
    stopServer: () => Promise.resolve(),
    restartServer: () => unsupported('重启本地服务'),
    setKeepServerRunning: () => Promise.resolve(),
    setBackendOverride: () => Promise.resolve(),
    setupWindowCloseHandler: () => Promise.resolve(),
    subscribeToServerLogs: () => () => {},
  };

  return {
    filesystem,
    updater,
    audio,
    lifecycle,
    metadata: {
      getVersion: async () => '开发版',
      isTauri: false,
    },
  };
}

function createTauriUpdater(): PlatformUpdater {
  let status = { ...INITIAL_UPDATE_STATUS };
  let pendingUpdate: Update | null = null;
  const listeners = new Set<(next: UpdateStatus) => void>();

  const publish = (next: Partial<UpdateStatus>) => {
    status = { ...status, ...next };
    listeners.forEach((listener) => listener(status));
  };

  return {
    async checkForUpdates() {
      publish({ checking: true, error: undefined });
      try {
        pendingUpdate = await check();
        publish({
          checking: false,
          available: pendingUpdate !== null,
          version: pendingUpdate?.version,
        });
      } catch (error) {
        publish({ checking: false, error: error instanceof Error ? error.message : String(error) });
      }
    },
    async downloadAndInstall() {
      if (!pendingUpdate) {
        throw new Error('没有可安装的更新。');
      }
      let downloadedBytes = 0;
      publish({ downloading: true, error: undefined });
      try {
        await pendingUpdate.download((event) => {
          if (event.event === 'Started') {
            publish({ totalBytes: event.data.contentLength });
          } else if (event.event === 'Progress') {
            downloadedBytes += event.data.chunkLength;
            publish({
              downloadedBytes,
              downloadProgress:
                status.totalBytes && status.totalBytes > 0
                  ? Math.round((downloadedBytes / status.totalBytes) * 100)
                  : undefined,
            });
          }
        });
        publish({ downloading: false, readyToInstall: true });
      } catch (error) {
        publish({ downloading: false, error: error instanceof Error ? error.message : String(error) });
      }
    },
    async restartAndInstall() {
      if (!pendingUpdate) {
        throw new Error('没有已下载的更新。');
      }
      publish({ installing: true, error: undefined });
      try {
        await pendingUpdate.install();
        await relaunch();
      } catch (error) {
        publish({ installing: false, error: error instanceof Error ? error.message : String(error) });
      }
    },
    getStatus: () => status,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

function createTauriPlatform(): Platform {
  const filesystem: PlatformFilesystem = {
    async saveFile(filename, blob, filters) {
      const target = await save({ defaultPath: filename, filters });
      if (target) {
        await writeFile(target, new Uint8Array(await blob.arrayBuffer()));
      }
    },
    openPath: (path) => openExternal(path),
    async pickDirectory(title) {
      const selected = await open({ title, directory: true, multiple: false });
      return typeof selected === 'string' ? selected : null;
    },
  };

  const audio: PlatformAudio = {
    isSystemAudioSupported: () => invoke<boolean>('is_system_audio_supported'),
    startSystemAudioCapture: (maxDurationSecs) =>
      invoke('start_system_audio_capture', { maxDurationSecs }),
    async stopSystemAudioCapture() {
      const base64Audio = await invoke<string>('stop_system_audio_capture');
      const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
      return new Blob([bytes], { type: 'audio/wav' });
    },
    listOutputDevices: () => invoke('list_audio_output_devices'),
    playToDevices: (audioData, deviceIds) =>
      invoke('play_audio_to_devices', { audioData: Array.from(audioData), deviceIds }),
    stopPlayback: () => {
      void invoke('stop_audio_playback');
    },
  };

  const lifecycle: PlatformLifecycle = {
    async startServer(remote, modelsDir) {
      const serverUrl = await invoke<string>('start_server', { remote, modelsDir });
      lifecycle.onServerReady?.();
      return serverUrl;
    },
    stopServer: () => invoke('stop_server'),
    restartServer: (modelsDir) => invoke('restart_server', { modelsDir }),
    setKeepServerRunning: (keepRunning) =>
      invoke('set_keep_server_running', { keepRunning }),
    setBackendOverride: (backend) => invoke('set_backend_override', { backend }),
    async setupWindowCloseHandler() {
      await listen('window-close-requested', async () => {
        await emit('window-close-allowed');
      });
    },
    subscribeToServerLogs(callback) {
      let active = true;
      let unlisten: (() => void) | undefined;
      void listen<ServerLogEntry>('server-log', (event) => {
        if (active) callback(event.payload);
      }).then((remove) => {
        unlisten = remove;
      });
      return () => {
        active = false;
        unlisten?.();
      };
    },
  };

  return {
    filesystem,
    updater: createTauriUpdater(),
    audio,
    lifecycle,
    metadata: {
      getVersion,
      isTauri: true,
    },
  };
}

export function createPlatform(isDesktop = isTauri()): Platform {
  return isDesktop ? createTauriPlatform() : createBrowserPlatform();
}

export const platform = createPlatform();
