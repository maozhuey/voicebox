import { RouterProvider } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { DictateWindow } from '@/components/DictateWindow/DictateWindow';
import ShinyText from '@/components/ShinyText';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { useAutoUpdater } from '@/hooks/useAutoUpdater';
import { useThemeSync } from '@/hooks/useThemeSync';
import { apiClient } from '@/lib/api/client';
import type { HealthResponse } from '@/lib/api/types';
import { useChordSync } from '@/lib/hooks/useChordSync';
import { TOP_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { cn } from '@/lib/utils/cn';
import { toChineseErrorMessage } from '@/lib/utils/errorMessage';
import { usePlatform } from '@/platform/PlatformContext';
import { router } from '@/router';
import { useLogStore } from '@/stores/logStore';
import {
  getDefaultServerUrl,
  isLoopbackVoiceboxServerUrl,
  useServerStore,
} from '@/stores/serverStore';

function isDictateView(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('view') === 'dictate';
}

/**
 * Validate that a health response has the expected Voicebox-specific shape.
 * Prevents misidentifying an unrelated service on the same port.
 */
function isVoiceboxHealthResponse(health: HealthResponse): boolean {
  return (
    health?.status === 'healthy' &&
    typeof health.model_loaded === 'boolean' &&
    typeof health.gpu_available === 'boolean'
  );
}

/**
 * Check whether a startup error indicates the port is occupied by an external
 * server (which we should try to reuse via health-check polling) vs. a real
 * failure (missing sidecar, signing issue, etc.) that should surface immediately.
 */
function isPortInUseError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('already in use') ||
    msg.includes('port') ||
    msg.includes('EADDRINUSE') ||
    msg.includes('address already in use')
  );
}

const LOADING_MESSAGES = [
  '正在预热张量…',
  '正在校准语音合成引擎…',
  '正在初始化声音模型…',
  '正在加载神经网络…',
  '正在准备音频处理流程…',
  '正在优化波形生成器…',
  '正在调节频率分析器…',
  '正在构建声音特征…',
  '正在配置文本转语音核心…',
  '正在同步音频缓冲区…',
  '正在连接模型…',
  '正在预处理训练数据…',
  '正在验证声音样本…',
  '正在编译推理引擎…',
  '正在映射音素序列…',
  '正在对齐韵律参数…',
  '正在启动语音合成…',
  '正在微调声学模型…',
  '正在准备声音克隆参数…',
  '正在初始化 Qwen TTS 框架…',
];

function App() {
  useThemeSync();

  // The dictate window runs in a separate Tauri webview that must skip
  // server bootstrap (the main window owns that lifecycle) and render only
  // the floating recording surface. Split into a sibling component so the
  // main app's hooks are not called on the dictate path.
  if (isDictateView()) {
    return <DictateWindow />;
  }
  return <MainApp />;
}

function MainApp() {
  const platform = usePlatform();
  const [serverReady, setServerReady] = useState(false);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);
  // When non-null, replaces the cycling loading message with a retry hint.
  const [startupRetryHint, setStartupRetryHint] = useState<string | null>(null);
  const serverStartingRef = useRef(false);
  // Holds a cancel function for any in-flight startServer retry chain so the
  // effect cleanup can abort pending timers if the component unmounts.
  const cancelStartRetryRef = useRef<(() => void) | null>(null);

  // Automatically check for app updates on startup and show toast notifications
  useAutoUpdater({ checkOnMount: true, showToast: true });

  // Replay the saved chord into the Rust hotkey listener every time
  // capture_settings resolves or the user edits the chord.
  useChordSync();

  // Sync stored setting to Rust on startup
  useEffect(() => {
    if (platform.metadata.isTauri) {
      const keepRunning = useServerStore.getState().keepServerRunningOnClose;
      platform.lifecycle.setKeepServerRunning(keepRunning).catch((error) => {
        console.error('Failed to sync initial setting to Rust:', error);
      });
    }
    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.metadata.isTauri, platform.lifecycle]);

  // Setup lifecycle callbacks
  useEffect(() => {
    platform.lifecycle.onServerReady = () => {
      setServerReady(true);
    };
    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.lifecycle]);

  // Subscribe to server logs
  useEffect(() => {
    const unsubscribe = platform.lifecycle.subscribeToServerLogs((entry) => {
      useLogStore.getState().addEntry(entry);
    });
    return unsubscribe;
  }, [platform.lifecycle]);

  // Setup window close handler and auto-start server when running in Tauri (production only)
  useEffect(() => {
    if (!platform.metadata.isTauri) {
      const serverUrl = getDefaultServerUrl();
      const currentServerUrl = useServerStore.getState().serverUrl;
      if (currentServerUrl !== serverUrl && isLoopbackVoiceboxServerUrl(currentServerUrl)) {
        useServerStore.getState().setServerUrl(serverUrl);
      }
      setServerReady(true); // Web assumes server is running
      return;
    }

    // Setup window close handler to check setting and stop server if needed
    // This works in both dev and prod, but will only stop server if it was started by the app
    platform.lifecycle.setupWindowCloseHandler().catch((error) => {
      console.error('Failed to setup window close handler:', error);
    });

    // Only auto-start server in production mode
    // In dev mode, user runs server separately
    if (!import.meta.env?.PROD) {
      console.log('Dev mode: Skipping auto-start of server (run it separately)');
      setServerReady(true); // Mark as ready so UI doesn't show loading screen
      // Mark that server was not started by app (so we don't try to stop it on close)
      window.__voiceboxServerStartedByApp = false;
      return;
    }

    // Auto-start server in production
    if (serverStartingRef.current) {
      return;
    }

    serverStartingRef.current = true;
    const isRemote = useServerStore.getState().mode === 'remote';
    const customModelsDir = useServerStore.getState().customModelsDir;
    console.log(`Production mode: Starting bundled server... (remote: ${isRemote})`);

    platform.lifecycle
      .startServer(isRemote, customModelsDir)
      .then((serverUrl) => {
        console.log('Server is ready at:', serverUrl);
        // Update the server URL in the store with the dynamically assigned port
        useServerStore.getState().setServerUrl(serverUrl);
        setServerReady(true);
        // Mark that we started the server (so we know to stop it on close)
        window.__voiceboxServerStartedByApp = true;
      })
      .catch((error) => {
        console.error('Failed to auto-start server:', error);
        window.__voiceboxServerStartedByApp = false;

        // Port-in-use errors mean something is already listening on 17493.
        // Fall back to health-check polling — the running process may be a
        // legitimate external server (e.g. started via python/uvicorn/Docker)
        // that just hasn't reported ready yet.
        if (isPortInUseError(error)) {
          // Fall back to polling: the server may already be running externally
          // (e.g. started via python/uvicorn/Docker). Poll the health endpoint
          // until it responds with a valid Voicebox payload, then transition to
          // the main UI.
          console.log('Falling back to health-check polling...');
          const pollInterval = setInterval(async () => {
            try {
              const health = await apiClient.getHealth();
              if (!isVoiceboxHealthResponse(health)) {
                console.log('Health response is not from a Voicebox server, keep polling...');
                return;
              }
              console.log('External Voicebox server detected via health check');
              clearInterval(pollInterval);
              setServerReady(true);
            } catch {
              // Server not ready yet, keep polling
            }
          }, 2000);

          // Stop polling after 2 minutes and surface the failure
          setTimeout(() => {
            clearInterval(pollInterval);
            serverStartingRef.current = false;
            setStartupError('两分钟内无法连接 Voicebox 服务器。请确认服务器正在运行，然后重试。');
          }, 120_000);
          return;
        }

        // Real startup failure (spawn rejection, sidecar timeout, signing
        // issue, etc.). The bundled sidecar is usually slow on first launch
        // (PyInstaller extracts + torch/MLX import can take 60-90s) and the
        // Tauri command's own 120s wait may race the React startup, so retry
        // a few times before giving up. This avoids leaving the user stuck
        // on the error screen after a transient failure (e.g. MLX runtime
        // poisoned in a previous run, or a brief HF Hub hiccup during the
        // first import).
        const maxAttempts = 3;
        const retryDelayMs = 3000;
        let attempt = 1;
        let retryTimer: ReturnType<typeof setTimeout> | null = null;
        let cancelled = false;

        const tryAgain = () => {
          if (cancelled) return;
          if (attempt >= maxAttempts) {
            console.error('Real startup failure — giving up after retries');
            serverStartingRef.current = false;
            setStartupRetryHint(null);
            setStartupError(
              toChineseErrorMessage(error, '服务器启动失败，请检查本地服务配置。'),
            );
            return;
          }
          attempt += 1;
          console.log(
            `Retrying startServer (attempt ${attempt}/${maxAttempts}) in ${retryDelayMs / 1000}s...`,
          );
          setStartupRetryHint(
            `服务器启动遇到问题,正在重试 (${attempt}/${maxAttempts})…`,
          );
          retryTimer = setTimeout(async () => {
            if (cancelled) return;
            try {
              const serverUrl = await platform.lifecycle.startServer(isRemote, customModelsDir);
              console.log('Server is ready at:', serverUrl);
              useServerStore.getState().setServerUrl(serverUrl);
              setServerReady(true);
              window.__voiceboxServerStartedByApp = true;
              setStartupRetryHint(null);
            } catch (retryError) {
              console.error(`startServer attempt ${attempt} failed:`, retryError);
              if (isPortInUseError(retryError)) {
                // Port came up while we were retrying — switch to polling path.
                cancelled = true;
                if (retryTimer) clearTimeout(retryTimer);
                setStartupRetryHint(null);
                const pollInterval = setInterval(async () => {
                  try {
                    const health = await apiClient.getHealth();
                    if (!isVoiceboxHealthResponse(health)) return;
                    clearInterval(pollInterval);
                    setServerReady(true);
                  } catch {
                    /* keep polling */
                  }
                }, 2000);
                setTimeout(() => {
                  clearInterval(pollInterval);
                  serverStartingRef.current = false;
                  setStartupError('两分钟内无法连接 Voicebox 服务器。请确认服务器正在运行，然后重试。');
                }, 120_000);
                return;
              }
              tryAgain();
            }
          }, retryDelayMs);
        };

        // Stash the cancel handle on the ref so the effect cleanup can stop
        // pending retries when the component unmounts.
        cancelStartRetryRef.current = () => {
          cancelled = true;
          if (retryTimer) clearTimeout(retryTimer);
        };

        tryAgain();
      });

    // Cleanup: stop server on actual unmount (not StrictMode remount)
    // Note: Window close is handled separately in Tauri Rust code
    return () => {
      // Window close event handles server shutdown based on setting
      serverStartingRef.current = false;
      // Abort any pending startServer retry chain from a previous run.
      if (cancelStartRetryRef.current) {
        cancelStartRetryRef.current();
        cancelStartRetryRef.current = null;
      }
    };
    // Empty dependency array - platform is stable from context, only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.metadata.isTauri, platform.lifecycle]);

  // Cycle through loading messages every 3 seconds
  useEffect(() => {
    if (!platform.metadata.isTauri || serverReady) {
      return;
    }

    const interval = setInterval(() => {
      setLoadingMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 3000);

    return () => clearInterval(interval);
  }, [serverReady, platform.metadata.isTauri]);

  // Show loading screen while server is starting in Tauri
  if (platform.metadata.isTauri && !serverReady) {
    return (
      <div
        className={cn(
          'min-h-screen bg-background flex items-center justify-center',
          TOP_SAFE_AREA_PADDING,
        )}
      >
        <TitleBarDragRegion />
        <div className="text-center space-y-6">
          <div className="flex justify-center relative">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-48 rounded-full bg-accent/20 blur-3xl" />
            </div>
            <img
              src={voiceboxLogo}
              alt="Voicebox"
              className="w-48 h-48 object-contain animate-fade-in-scale relative z-10"
            />
          </div>
          {startupError ? (
            <div className="animate-fade-in-delayed max-w-md mx-auto space-y-3">
              <p className="text-lg font-medium text-destructive">服务器启动失败</p>
              <p className="text-sm text-muted-foreground">{startupError}</p>
              <button
                type="button"
                className="mt-2 px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                onClick={() => {
                  setStartupError(null);
                  serverStartingRef.current = false;
                  // Trigger a re-mount of the effect by toggling state
                  window.location.reload();
                }}
              >
                重试
              </button>
            </div>
          ) : (
            <div className="animate-fade-in-delayed space-y-2">
              <ShinyText
                text={startupRetryHint ?? LOADING_MESSAGES[loadingMessageIndex]}
                className="text-lg font-medium text-muted-foreground"
                speed={2}
                color="hsl(var(--muted-foreground))"
                shineColor="hsl(var(--foreground))"
              />
              {startupRetryHint ? (
                <p className="text-xs text-muted-foreground/70">
                  这通常是因为上次会话的 MLX runtime 状态需要重置
                </p>
              ) : null}
            </div>
          )}
        </div>
      </div>
    );
  }

  return <RouterProvider router={router} />;
}

export default App;
