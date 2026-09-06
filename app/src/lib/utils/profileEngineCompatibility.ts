import type { VoiceProfileResponse } from '@/lib/api/types';

const CLONING_ENGINES = new Set([
  'qwen',
  'luxtts',
  'chatterbox',
  'chatterbox_turbo',
  'tada',
  'cosyvoice',
]);

/**
 * 声音档案决定可用的合成引擎：预设声音只能使用它随档案保存的引擎，
 * 克隆声音不能被 CustomVoice/Kokoro 等预设引擎静默替换。
 */
export function getProfileEngineCompatibilityError(
  profile: VoiceProfileResponse | null | undefined,
  engine: string | undefined,
): string | null {
  if (!profile || !engine) return null;
  const voiceType = profile.voice_type || 'cloned';
  if (voiceType === 'preset' && profile.preset_engine !== engine) {
    return `预设声音“${profile.name}”仅支持 ${profile.preset_engine === 'qwen_custom_voice' ? 'Qwen CustomVoice' : profile.preset_engine}，请先切换声音或使用兼容模型。`;
  }
  if (voiceType === 'cloned' && !CLONING_ENGINES.has(engine)) {
    return `当前克隆声音不支持该模型；${engine === 'qwen_custom_voice' ? 'Qwen CustomVoice 仅支持预设声音。' : '请选择兼容模型。'}`;
  }
  return null;
}

export function isProfileCompatibleWithEngine(
  profile: VoiceProfileResponse | null | undefined,
  engine: string,
): boolean {
  return getProfileEngineCompatibilityError(profile, engine) === null;
}

export function getProfilePreferredEngine(
  profile: VoiceProfileResponse | null | undefined,
): string | undefined {
  return profile?.default_engine ?? profile?.preset_engine;
}
