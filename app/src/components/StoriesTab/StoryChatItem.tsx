import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, Mic, MoreHorizontal, Music, Play, RotateCcw, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Textarea } from '@/components/ui/textarea';
import type { StoryItemDetail } from '@/lib/api/types';
import { ALL_LANGUAGES, type LanguageCode } from '@/lib/constants/languages';
import { cn } from '@/lib/utils/cn';
import { useServerStore } from '@/stores/serverStore';
import { useStoryStore } from '@/stores/storyStore';

interface StoryChatItemProps {
  item: StoryItemDetail;
  storyId: string;
  index: number;
  onRemove: () => void;
  onRegenerate?: () => void;
  currentTimeMs: number;
  isPlaying: boolean;
  dragHandleProps?: React.HTMLAttributes<HTMLButtonElement>;
  isDragging?: boolean;
}

const ENGINE_MODEL_LABELS: Record<string, string> = {
  qwen: 'Qwen3-TTS',
  qwen_custom_voice: 'Qwen CustomVoice',
  luxtts: 'LuxTTS',
  chatterbox: 'Chatterbox',
  chatterbox_turbo: 'Chatterbox Turbo',
  tada: 'TADA',
  kokoro: 'Kokoro 82M',
};

function getModelLabel(item: StoryItemDetail): string {
  if (item.engine === 'cosyvoice') {
    return item.model_size === 'base' ? 'CosyVoice 3 0.5B' : 'CosyVoice 3 0.5B RL';
  }

  const engineLabel = ENGINE_MODEL_LABELS[item.engine ?? ''] ?? item.engine ?? '—';
  return item.model_size ? `${engineLabel} ${item.model_size}` : engineLabel;
}

function getCosyVoiceDialect(item: StoryItemDetail): StoryItemDetail['dialect'] {
  if (item.dialect) return item.dialect;
  // Legacy takes predate the persisted dialect fields. Their effective
  // instruction is still stored, so recover a display-only value when safe.
  if (item.instruct?.includes('请用河南话表达。')) return 'henan';
  if (item.instruct?.includes('请用四川话表达。')) return 'sichuan';
  if (item.instruct?.includes('请用普通话表达。')) return 'mandarin';
  return undefined;
}

export function StoryChatItem({
  item,
  onRemove,
  onRegenerate,
  currentTimeMs,
  isPlaying,
  dragHandleProps,
  isDragging,
}: StoryChatItemProps) {
  const { t } = useTranslation();
  const seek = useStoryStore((state) => state.seek);
  const serverUrl = useServerStore((state) => state.serverUrl);
  const [avatarError, setAvatarError] = useState(false);

  const avatarUrl = `${serverUrl}/profiles/${item.profile_id}/avatar`;

  // Check if this item is currently playing based on timecode
  const itemStartMs = item.start_time_ms;
  const itemEndMs = item.start_time_ms + item.duration * 1000;
  const isCurrentlyPlaying = isPlaying && currentTimeMs >= itemStartMs && currentTimeMs < itemEndMs;
  const isGeneratedAudio = item.engine !== 'import';
  const languageLabel = ALL_LANGUAGES[item.language as LanguageCode] ?? item.language;
  const cosyvoiceMode =
    item.engine === 'cosyvoice'
      ? (item.cosyvoice_mode ?? (item.instruct ? 'instruct' : 'reference'))
      : undefined;
  const dialect = item.engine === 'cosyvoice' ? getCosyVoiceDialect(item) : undefined;
  const selectedVersionId = item.version_id ?? item.active_version_id;
  const activeEffects =
    item.versions
      ?.find((version) => version.id === selectedVersionId)
      ?.effects_chain?.filter((effect) => effect.enabled) ?? [];
  const effectsLabel = activeEffects.length
    ? activeEffects
        .map((effect) => t(`effects.types.${effect.type}.label`, { defaultValue: effect.type }))
        .join('、')
    : t('generation.effects.none');

  const handlePlay = () => {
    // Seek to the start of this item
    seek(itemStartMs);
  };

  const formatTime = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const milliseconds = Math.floor((ms % 1000) / 100);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
  };

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border transition-colors',
        isCurrentlyPlaying && 'bg-muted/70 border-primary',
        !isCurrentlyPlaying && 'hover:bg-muted/50',
        isDragging && 'opacity-50 shadow-lg',
      )}
    >
      {/* Drag Handle */}
      {dragHandleProps && (
        <button
          type="button"
          className="shrink-0 cursor-grab active:cursor-grabbing touch-none text-muted-foreground hover:text-foreground transition-colors"
          {...dragHandleProps}
        >
          <GripVertical className="h-5 w-5" />
        </button>
      )}

      {/* Voice Avatar */}
      <div className="shrink-0">
        <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center overflow-hidden">
          {item.engine === 'import' ? (
            <Music className="h-5 w-5 text-muted-foreground" />
          ) : !avatarError ? (
            <img
              src={avatarUrl}
              alt={`${item.profile_name} avatar`}
              className={cn(
                'h-full w-full object-cover transition-all duration-200',
                !isCurrentlyPlaying && 'grayscale',
              )}
              onError={() => setAvatarError(true)}
            />
          ) : (
            <Mic className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-medium text-sm truncate">
            {item.engine === 'import' ? item.text : item.profile_name}
          </span>
          {isGeneratedAudio && (
            <span className="text-xs text-muted-foreground">{item.language}</span>
          )}
          <span className="text-xs text-muted-foreground tabular-nums ml-auto">
            {formatTime(itemStartMs)}
          </span>
        </div>
        {item.engine === 'import' ? null : (
          <>
            <fieldset className="mb-2 flex flex-wrap gap-1.5 border-0 p-0">
              <legend className="sr-only">{t('storyContent.generationConfig.title')}</legend>
              {[
                t('storyContent.generationConfig.voice', { value: item.profile_name }),
                t('storyContent.generationConfig.language', { value: languageLabel }),
                t('storyContent.generationConfig.model', { value: getModelLabel(item) }),
                ...(cosyvoiceMode
                  ? [
                      t('storyContent.generationConfig.mode', {
                        value: t(`generation.cosyvoiceMode.${cosyvoiceMode}`),
                      }),
                    ]
                  : []),
                ...(dialect
                  ? [
                      t('storyContent.generationConfig.dialect', {
                        value: t(`generation.dialect.${dialect}`),
                      }),
                    ]
                  : []),
                t('storyContent.generationConfig.effects', { value: effectsLabel }),
                t('storyContent.generationConfig.naturalReading', {
                  value: t(
                    item.natural_reading
                      ? 'storyContent.generationConfig.enabled'
                      : 'storyContent.generationConfig.disabled',
                  ),
                }),
              ].map((label) => (
                <span
                  key={label}
                  className="rounded-full border border-border bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground"
                >
                  {label}
                </span>
              ))}
            </fieldset>
            {item.instruct && (
              <p className="mb-2 break-words text-xs leading-5 text-muted-foreground">
                {t('storyContent.generationConfig.instruct', { value: item.instruct })}
              </p>
            )}
            <Textarea
              value={item.text}
              className="flex-1 resize-none text-sm text-muted-foreground select-text bg-card cursor-text"
              readOnly
              onDoubleClick={handlePlay}
            />
          </>
        )}
      </div>

      {/* Actions */}
      <div className="shrink-0">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              aria-label={t('history.actions.menu')}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={handlePlay}>
              <Play className="mr-2 h-4 w-4" />
              {t('storyContent.itemActions.playFromHere')}
            </DropdownMenuItem>
            {onRegenerate && (
              <DropdownMenuItem onClick={onRegenerate}>
                <RotateCcw className="mr-2 h-4 w-4" />
                {t('storyContent.itemActions.regenerate')}
              </DropdownMenuItem>
            )}
            <DropdownMenuItem
              onClick={onRemove}
              className="text-destructive focus:text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {t('storyContent.itemActions.removeFromStory')}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

// Sortable wrapper component
export function SortableStoryChatItem(
  props: Omit<StoryChatItemProps, 'dragHandleProps' | 'isDragging'>,
) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: props.item.generation_id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      <StoryChatItem {...props} dragHandleProps={listeners} isDragging={isDragging} />
    </div>
  );
}
