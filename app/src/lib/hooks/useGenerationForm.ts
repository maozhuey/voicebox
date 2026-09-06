import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig } from '@/lib/api/types';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { useGenerationStore } from '@/stores/generationStore';
import { useUIStore } from '@/stores/uiStore';

const generationSchema = z.object({
  text: z.string().min(1, '').max(50000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.enum(['1.7B', '0.6B', '1B', '3B', 'rl', 'base']).optional(),
  instruct: z.string().max(500).optional(),
  cosyvoiceMode: z.enum(['reference', 'instruct']),
  dialect: z.enum(['mandarin', 'henan', 'sichuan']),
  naturalReading: z.boolean(),
  engine: z
    .enum([
      'qwen',
      'qwen_custom_voice',
      'luxtts',
      'chatterbox',
      'chatterbox_turbo',
      'tada',
      'kokoro',
      'cosyvoice',
    ])
    .optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
  getEffectsChain?: () => EffectConfig[] | undefined;
  getTargetStoryId?: () => string | null;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { t } = useTranslation();
  const { toast } = useToast();
  const generation = useGeneration();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);
  const { settings: genSettings } = useGenerationSettings();
  const maxChunkChars = genSettings?.max_chunk_chars ?? 800;
  const crossfadeMs = genSettings?.crossfade_ms ?? 50;
  const normalizeAudio = genSettings?.normalize_audio ?? true;
  const globalNaturalReading = genSettings?.natural_reading ?? false;
  const selectedEngine = useUIStore((state) => state.selectedEngine);
  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  useModelDownloadToast({
    modelName: downloadingModelName || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModelName,
  });

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: '1.7B',
      instruct: '',
      cosyvoiceMode: 'reference',
      dialect: 'mandarin',
      naturalReading: globalNaturalReading,
      engine: (selectedEngine as GenerationFormValues['engine']) || 'qwen',
      ...options.defaultValues,
    },
  });

  useEffect(() => {
    // 业务规则：每个新任务从全局自然朗读设置继承初始值；用户一旦手动切换，
    // 当前任务保持自己的选择，不被后续全局设置刷新覆盖。
    if (!form.getFieldState('naturalReading').isDirty) {
      form.setValue('naturalReading', globalNaturalReading, { shouldDirty: false });
    }
  }, [form, globalNaturalReading]);

  async function handleSubmit(
    data: GenerationFormValues,
    selectedProfileId: string | null,
  ): Promise<void> {
    if (!selectedProfileId) {
      toast({
        title: t('generation.errors.noProfileTitle'),
        description: t('generation.errors.noProfileDescription'),
        variant: 'destructive',
      });
      return;
    }

    try {
      const engine = data.engine || 'qwen';
      const modelName =
        engine === 'luxtts'
          ? 'luxtts'
          : engine === 'chatterbox'
            ? 'chatterbox-tts'
            : engine === 'chatterbox_turbo'
              ? 'chatterbox-turbo'
              : engine === 'tada'
                ? data.modelSize === '3B'
                  ? 'tada-3b-ml'
                  : 'tada-1b'
                : engine === 'kokoro'
                  ? 'kokoro'
                  : engine === 'cosyvoice'
                    ? data.modelSize === 'base'
                      ? 'cosyvoice3-0.5b'
                      : 'cosyvoice3-0.5b-rl'
                    : engine === 'qwen_custom_voice'
                      ? `qwen-custom-voice-${data.modelSize}`
                      : `qwen-tts-${data.modelSize}`;
      const displayName =
        engine === 'luxtts'
          ? 'LuxTTS'
          : engine === 'chatterbox'
            ? 'Chatterbox TTS'
            : engine === 'chatterbox_turbo'
              ? 'Chatterbox Turbo'
              : engine === 'tada'
                ? data.modelSize === '3B'
                  ? 'TADA 3B Multilingual'
                  : 'TADA 1B'
                : engine === 'kokoro'
                  ? 'Kokoro 82M'
                  : engine === 'cosyvoice'
                    ? data.modelSize === 'base'
                      ? 'CosyVoice 3 0.5B'
                      : 'CosyVoice 3 0.5B RL'
                    : engine === 'qwen_custom_voice'
                      ? data.modelSize === '1.7B'
                        ? 'Qwen CustomVoice 1.7B'
                        : 'Qwen CustomVoice 0.6B'
                      : data.modelSize === '1.7B'
                        ? 'Qwen TTS 1.7B'
                        : 'Qwen TTS 0.6B';

      // Check if model needs downloading
      try {
        const modelStatus = await apiClient.getModelStatus();
        const model = modelStatus.models.find((m) => m.model_name === modelName);

        if (model && !model.downloaded) {
          setDownloadingModelName(modelName);
          setDownloadingDisplayName(displayName);
        }
      } catch (error) {
        console.error('Failed to check model status:', error);
      }

      const hasModelSizes =
        engine === 'qwen' ||
        engine === 'qwen_custom_voice' ||
        engine === 'tada' ||
        engine === 'cosyvoice';
      // Base Qwen3-TTS accepts the kwarg but ignores it. CosyVoice 3 and
      // Qwen CustomVoice turn it into model-level delivery-style control.
      const supportsInstruct = engine === 'qwen_custom_voice' || engine === 'cosyvoice';
      const effectsChain = options.getEffectsChain?.();
      // 业务规则：正文和朗读指令是两个完全独立的输入通道。人物设定只允许
      // 预填 instruct，提交时绝不能用 instruct 替换用户输入的正文。
      const scriptText = data.text;
      // This now returns immediately with status="generating"
      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        // 业务规则：故事关联必须随生成请求一起持久化，不能等待
        // 完成后再依赖浏览器内存补关联，否则刷新会导致音频丢失归属。
        target_story_id: options.getTargetStoryId?.() || undefined,
        text: scriptText,
        language: data.language,
        seed: data.seed,
        model_size: hasModelSizes ? data.modelSize : undefined,
        engine,
        // 业务规则：参考音频跟随模式不能携带朗读指令，否则 CosyVoice 会从
        // zero-shot 切换到 instruct2，削弱参考录音里的方言和自然语气。
        instruct:
          supportsInstruct && !(engine === 'cosyvoice' && data.cosyvoiceMode === 'reference')
            ? data.instruct || undefined
            : undefined,
        cosyvoice_mode: engine === 'cosyvoice' ? data.cosyvoiceMode : undefined,
        // 方言下拉项只属于 instruct2 模式；参考跟随模式的口音由录音决定。
        dialect:
          engine === 'cosyvoice' && data.language === 'zh' && data.cosyvoiceMode === 'instruct'
            ? data.dialect
            : undefined,
        max_chunk_chars: maxChunkChars,
        crossfade_ms: crossfadeMs,
        // 当前任务可以覆盖全局默认值；后端只接收本次最终选择。
        natural_reading: data.naturalReading,
        normalize: normalizeAudio,
        effects_chain: effectsChain?.length ? effectsChain : undefined,
      });

      // Track this generation for SSE status updates
      addPendingGeneration(result.id);

      // Reset form immediately — user can start typing again
      form.reset({
        text: '',
        language: data.language,
        seed: undefined,
        modelSize: data.modelSize,
        instruct: '',
        cosyvoiceMode: data.cosyvoiceMode,
        dialect: data.dialect,
        // 提交完成即开始一个新任务，重新继承此时的全局默认值。
        naturalReading: globalNaturalReading,
        engine: data.engine,
      });
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: t('generation.errors.failedTitle'),
        description:
          error instanceof Error ? error.message : t('generation.errors.failedDescription'),
        variant: 'destructive',
      });
    } finally {
      setDownloadingModelName(null);
      setDownloadingDisplayName(null);
    }
  }

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
  };
}
