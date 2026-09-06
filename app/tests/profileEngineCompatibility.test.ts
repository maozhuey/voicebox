import { describe, expect, test } from 'bun:test';
import type { VoiceProfileResponse } from '../src/lib/api/types';
import {
  getProfileEngineCompatibilityError,
  getProfilePreferredEngine,
  isProfileCompatibleWithEngine,
} from '../src/lib/utils/profileEngineCompatibility';
import { toChineseErrorMessage } from '../src/lib/utils/errorMessage';

const baseProfile = {
  language: 'zh',
  generation_count: 0,
  sample_count: 0,
  created_at: '2026-09-07T00:00:00Z',
  updated_at: '2026-09-07T00:00:00Z',
} satisfies Omit<VoiceProfileResponse, 'id' | 'name' | 'voice_type'>;

const vivian: VoiceProfileResponse = {
  ...baseProfile,
  id: 'vivian',
  name: '预设-Vivian',
  voice_type: 'preset',
  preset_engine: 'qwen_custom_voice',
  default_engine: 'qwen_custom_voice',
};

const cloned: VoiceProfileResponse = {
  ...baseProfile,
  id: 'clone',
  name: '我的声音',
  voice_type: 'cloned',
  default_engine: 'qwen',
};

describe('profile engine compatibility', () => {
  test('switching to Vivian selects its CustomVoice engine', () => {
    expect(getProfilePreferredEngine(vivian)).toBe('qwen_custom_voice');
    expect(isProfileCompatibleWithEngine(vivian, 'qwen_custom_voice')).toBe(true);
    expect(getProfileEngineCompatibilityError(vivian, 'qwen')).toContain('仅支持');
  });

  test('a cloned voice rejects CustomVoice without changing the selected profile', () => {
    expect(getProfilePreferredEngine(cloned)).toBe('qwen');
    expect(isProfileCompatibleWithEngine(cloned, 'qwen_custom_voice')).toBe(false);
    expect(getProfileEngineCompatibilityError(cloned, 'qwen_custom_voice')).toContain('仅支持预设声音');
  });

  test('keeps capture diagnostic identifiers while hiding raw server errors', () => {
    expect(toChineseErrorMessage('CAPTURE_DIAGNOSTIC:cap-123abc')).toBe(
      '听写失败，请重试。诊断编号：cap-123abc',
    );
  });
});
