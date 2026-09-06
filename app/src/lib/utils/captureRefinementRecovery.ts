import type { CaptureResponse } from '@/lib/api/types';

export type CaptureRefinementMode = 'automatic' | 'manual';

interface CaptureRefinementFailureInput {
  mode: CaptureRefinementMode;
  capture?: CaptureResponse;
  error: unknown;
}

export interface CaptureRefinementFailureRecovery {
  message: string;
  deliverRawTranscript: boolean;
}

function extractDiagnosticId(error: unknown): string | null {
  const message = error instanceof Error ? error.message : String(error ?? '');
  return (
    message.match(/CAPTURE_REFINEMENT_DIAGNOSTIC:([A-Za-z0-9-]+)/)?.[1] ??
    message.match(/诊断编号：([A-Za-z0-9-]+)/)?.[1] ??
    null
  );
}

/**
 * Decide whether an already-successful dictation may still deliver raw text.
 *
 * Business rule: automatic refinement is an optional second request after
 * Whisper has saved the capture. Its failure must not lose the recognized
 * text. A manual refine is user-initiated editing, so it must never trigger a
 * second clipboard/auto-paste delivery even if a raw transcript exists.
 */
export function getCaptureRefinementFailureRecovery({
  mode,
  capture,
  error,
}: CaptureRefinementFailureInput): CaptureRefinementFailureRecovery {
  const diagnosticId = extractDiagnosticId(error);
  const suffix = diagnosticId ? `诊断编号：${diagnosticId}` : '请稍后重试。';

  if (mode === 'automatic') {
    return {
      message: `转录成功，精修失败，已保留原始转录。${suffix}`,
      deliverRawTranscript: Boolean(capture?.transcript_raw),
    };
  }

  return {
    message: `精修失败，原有内容未改变。${suffix}`,
    deliverRawTranscript: false,
  };
}
