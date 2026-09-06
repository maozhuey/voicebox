import { describe, expect, test } from 'bun:test';
import type { CaptureResponse } from '../src/lib/api/types';
import { toChineseErrorMessage } from '../src/lib/utils/errorMessage';
import { getCaptureRefinementFailureRecovery } from '../src/lib/utils/captureRefinementRecovery';

const capture = {
  id: 'capture-1',
  audio_path: 'captures/capture-1.wav',
  source: 'dictation',
  language: 'zh',
  duration_ms: 1000,
  transcript_raw: '你好，哈喽。',
  transcript_segments: [],
  stt_model: 'turbo',
  created_at: '2026-09-07T00:00:00Z',
} satisfies CaptureResponse;

describe('capture refinement recovery', () => {
  test('automatic refinement failure delivers the saved raw transcript once', () => {
    const recovery = getCaptureRefinementFailureRecovery({
      mode: 'automatic',
      capture,
      error: new Error('CAPTURE_REFINEMENT_DIAGNOSTIC:cap-123abc'),
    });

    expect(recovery.deliverRawTranscript).toBe(true);
    expect(recovery.message).toBe('转录成功，精修失败，已保留原始转录。诊断编号：cap-123abc');
  });

  test('manual refinement failure leaves the current capture untouched', () => {
    const recovery = getCaptureRefinementFailureRecovery({
      mode: 'manual',
      capture,
      error: new Error('CAPTURE_REFINEMENT_DIAGNOSTIC:cap-123abc'),
    });

    expect(recovery.deliverRawTranscript).toBe(false);
    expect(recovery.message).toBe('精修失败，原有内容未改变。诊断编号：cap-123abc');
  });

  test('automatic recovery never delivers an empty raw transcript', () => {
    const recovery = getCaptureRefinementFailureRecovery({
      mode: 'automatic',
      capture: { ...capture, transcript_raw: '' },
      error: new Error('unexpected failure'),
    });

    expect(recovery.deliverRawTranscript).toBe(false);
    expect(recovery.message).toBe('转录成功，精修失败，已保留原始转录。请稍后重试。');
  });

  test('error mapping exposes a refinement diagnostic instead of a 500 code', () => {
    expect(toChineseErrorMessage('CAPTURE_REFINEMENT_DIAGNOSTIC:cap-123abc')).toBe(
      '精修失败，请重试。诊断编号：cap-123abc',
    );
  });
});
