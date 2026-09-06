import { useCallback, useEffect, useRef, useState } from 'react';
import { convertToWav } from '@/lib/utils/audio';
import { toChineseErrorMessage } from '@/lib/utils/errorMessage';
import { usePlatform } from '@/platform/PlatformContext';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

type RecordingPhase = 'idle' | 'starting' | 'recording' | 'stopping';

interface RecordingSession {
  id: number;
  recorder: MediaRecorder | null;
  stream: MediaStream | null;
  chunks: Blob[];
  startedAt: number | null;
  cancelled: boolean;
  stopRequested: boolean;
}

export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<RecordingSession | null>(null);
  const phaseRef = useRef<RecordingPhase>('idle');
  const nextSessionIdRef = useRef(0);
  const timerRef = useRef<number | null>(null);

  const startRecording = useCallback(async () => {
    // Business rule: a hotkey may emit duplicate starts while getUserMedia is
    // still resolving. Treat starting/recording/stopping as one exclusive
    // session so two MediaRecorders can never share chunks or overwrite each
    // other's stream reference.
    if (phaseRef.current !== 'idle') return;

    const session: RecordingSession = {
      id: ++nextSessionIdRef.current,
      recorder: null,
      stream: null,
      chunks: [],
      startedAt: null,
      cancelled: false,
      stopRequested: false,
    };
    sessionRef.current = session;
    phaseRef.current = 'starting';
    // Starting counts as recording for callers so a quick key release is not
    // dropped while the browser is still opening the microphone.
    setIsRecording(true);

    try {
      setError(null);
      setDuration(0);

      // Check if getUserMedia is available
      // In Tauri, navigator.mediaDevices might not be available immediately
      if (typeof navigator === 'undefined') {
        const errorMsg = '浏览器录音接口不可用，可能是桌面客户端配置异常。';
        setError(errorMsg);
        throw new Error(errorMsg);
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        // Try waiting a bit for Tauri webview to initialize
        await new Promise((resolve) => setTimeout(resolve, 100));

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          console.error('MediaDevices check:', {
            hasNavigator: typeof navigator !== 'undefined',
            hasMediaDevices: !!navigator?.mediaDevices,
            hasGetUserMedia: !!navigator?.mediaDevices?.getUserMedia,
            isTauri: platform.metadata.isTauri,
          });

          const errorMsg = platform.metadata.isTauri
            ? '无法访问麦克风。请确认：\n1. 已在系统设置中授予应用麦克风权限（macOS：系统设置 > 隐私与安全性 > 麦克风）\n2. 授权后已重启应用\n3. 当前桌面客户端支持麦克风录音'
            : '无法访问麦克风。请确认当前页面使用 HTTPS 或 localhost，并已在浏览器中授予麦克风权限。';
          setError(errorMsg);
          throw new Error(errorMsg);
        }
      }

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      session.stream = stream;
      // stop/cancel can arrive while getUserMedia is pending. Close only this
      // session's stream; an immediately-started replacement owns a different
      // object and must not be stopped by this stale continuation.
      if (session.cancelled || session.stopRequested || sessionRef.current?.id !== session.id) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
        if (sessionRef.current?.id === session.id) {
          sessionRef.current = null;
          phaseRef.current = 'idle';
          setIsRecording(false);
        }
        return;
      }

      // Create MediaRecorder with preferred MIME type
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
      };

      // Fallback to default if webm not supported
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        delete options.mimeType;
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      session.recorder = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          session.chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const recordedDuration = session.startedAt
          ? (Date.now() - session.startedAt) / 1000
          : undefined;

        const webmBlob = new Blob(session.chunks, { type: 'audio/webm' });

        // Stop all tracks now that we have the data
        session.stream?.getTracks().forEach((track) => {
          track.stop();
        });
        session.stream = null;

        if (sessionRef.current?.id === session.id) {
          sessionRef.current = null;
          phaseRef.current = 'idle';
        }

        // Don't fire completion callback if the recording was cancelled
        if (session.cancelled) return;

        // Convert to WAV format to avoid needing ffmpeg on backend
        try {
          const wavBlob = await convertToWav(webmBlob);
          onRecordingComplete?.(wavBlob, recordedDuration);
        } catch (err) {
          console.error('Error converting audio to WAV:', err);
          // Fallback to original blob if conversion fails
          onRecordingComplete?.(webmBlob, recordedDuration);
        }
      };

      mediaRecorder.onerror = (event) => {
        setError('录音过程中发生错误');
        console.error('MediaRecorder error:', event);
      };

      // WebKit's MediaRecorder drops the WebM EBML header from chunks when
      // started with a timeslice, so concatenated blobs fail to parse in
      // both AudioContext and ffmpeg. Starting with no timeslice produces
      // exactly one dataavailable on stop() with a valid container.
      mediaRecorder.start();
      phaseRef.current = 'recording';
      session.startedAt = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (session.startedAt && sessionRef.current?.id === session.id) {
          const elapsed = (Date.now() - session.startedAt) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration when the caller opts in — dictation
          // sessions pass undefined and run until the user releases the
          // chord or hits stop; voice-clone sample recorders pass 29s to
          // keep reference clips short.
          if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
            if (mediaRecorder.state !== 'inactive') {
              phaseRef.current = 'stopping';
              mediaRecorder.stop();
              setIsRecording(false);
              if (timerRef.current !== null) {
                clearInterval(timerRef.current);
                timerRef.current = null;
              }
            }
          }
        }
      }, 100);
    } catch (err) {
      session.stream?.getTracks().forEach((track) => {
        track.stop();
      });
      if (sessionRef.current?.id !== session.id) return;
      sessionRef.current = null;
      phaseRef.current = 'idle';
      const errorMessage = toChineseErrorMessage(err, '无法访问麦克风，请检查权限设置。');
      setError(errorMessage);
      setIsRecording(false);
    }
  }, [maxDurationSeconds, onRecordingComplete, platform.metadata.isTauri]);

  const stopRecording = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;

    if (phaseRef.current === 'starting') {
      // A release before microphone readiness means no reliable audio was
      // captured. Detach immediately so the next hotkey press can start a new
      // session; the stale continuation's id check closes its eventual stream
      // and prevents a misleading empty/wrong upload.
      session.stopRequested = true;
      session.cancelled = true;
      sessionRef.current = null;
      phaseRef.current = 'idle';
      setIsRecording(false);
      return;
    }

    if (phaseRef.current === 'recording' && session.recorder) {
      phaseRef.current = 'stopping';
      session.recorder.stop();
      setIsRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, []);

  const cancelRecording = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    session.cancelled = true;
    session.stopRequested = true;
    session.chunks = [];

    if (session.recorder && session.recorder.state !== 'inactive') {
      // Detach before stop() so a dictate:restart can synchronously open its
      // replacement session. The old onstop closure owns its own recorder,
      // chunks, and stream and its id check prevents it from clearing the new
      // session when conversion finishes later.
      if (sessionRef.current?.id === session.id) {
        sessionRef.current = null;
        phaseRef.current = 'idle';
      }
      session.recorder.stop();
    } else {
      session.stream?.getTracks().forEach((track) => {
        track.stop();
      });
      if (sessionRef.current?.id === session.id) {
        sessionRef.current = null;
        phaseRef.current = 'idle';
      }
    }
    setIsRecording(false);
    setDuration(0);

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      const session = sessionRef.current;
      session?.stream?.getTracks().forEach((track) => {
        track.stop();
      });
      sessionRef.current = null;
      phaseRef.current = 'idle';
    };
  }, []);

  return {
    isRecording,
    duration,
    error,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
