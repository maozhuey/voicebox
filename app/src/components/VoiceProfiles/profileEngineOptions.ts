export const PRESET_ONLY_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);

// 新建弹窗和声音页右侧编辑器必须共用同一份默认引擎选项，
// 避免新增引擎后只有一个入口可以选择。
export const DEFAULT_ENGINE_OPTIONS = [
  { value: 'qwen', label: 'Qwen3-TTS', engine: 'qwen' },
  {
    value: 'qwen_custom_voice',
    label: 'Qwen CustomVoice',
    engine: 'qwen_custom_voice',
  },
  { value: 'luxtts', label: 'LuxTTS', engine: 'luxtts' },
  { value: 'chatterbox', label: 'Chatterbox', engine: 'chatterbox' },
  {
    value: 'chatterbox_turbo',
    label: 'Chatterbox Turbo',
    engine: 'chatterbox_turbo',
  },
  { value: 'tada', label: 'TADA', engine: 'tada' },
  { value: 'kokoro', label: 'Kokoro 82M', engine: 'kokoro' },
  {
    value: 'cosyvoice:rl',
    label: 'CosyVoice 3 0.5B RL',
    engine: 'cosyvoice',
    modelSize: 'rl',
  },
  {
    value: 'cosyvoice:base',
    label: 'CosyVoice 3 0.5B (Base)',
    engine: 'cosyvoice',
    modelSize: 'base',
  },
] as const;

export function getDefaultEngineSelection(
  engine?: string | null,
  modelSize?: string | null,
): string {
  if (engine === 'cosyvoice') return `cosyvoice:${modelSize === 'base' ? 'base' : 'rl'}`;
  return engine ?? '';
}

export function parseDefaultEngineSelection(value: string): {
  engine: string;
  modelSize?: string;
} {
  if (value.startsWith('cosyvoice:')) {
    const [, modelSize] = value.split(':');
    return { engine: 'cosyvoice', modelSize };
  }
  return { engine: value };
}
