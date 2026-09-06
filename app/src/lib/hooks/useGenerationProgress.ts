import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { HistoryListResponse } from '@/lib/api/types';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { toChineseErrorMessage } from '@/lib/utils/errorMessage';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

interface GenerationStatusEvent {
  id: string;
  status: 'loading_model' | 'generating' | 'completed' | 'failed' | 'not_found';
  duration?: number;
  error?: string;
  source?: string;
  progress_current?: number;
  progress_total?: number;
}

// Agent-initiated generations are played by the floating pill, not the
// main-window AudioPlayer. Skip autoplay here to avoid double-playback.
const AGENT_SOURCES = new Set(['mcp', 'rest']);

/**
 * Subscribes to SSE for all pending generations. When a generation completes,
 * invalidates the history query, removes it from pending, and auto-plays
 * if the player is idle.
 */
export function useGenerationProgress() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const pendingIds = useGenerationStore((s) => s.pendingGenerationIds);
  const removePendingGeneration = useGenerationStore((s) => s.removePendingGeneration);
  const removePendingStoryAdd = useGenerationStore((s) => s.removePendingStoryAdd);
  const consumeAutoPlaySuppression = useGenerationStore((s) => s.consumeAutoPlaySuppression);
  const isPlaying = usePlayerStore((s) => s.isPlaying);
  const setAudioWithAutoPlay = usePlayerStore((s) => s.setAudioWithAutoPlay);
  const { settings: genSettings } = useGenerationSettings();
  const autoplayOnGenerate = genSettings?.autoplay_on_generate ?? false;

  // Keep refs to avoid stale closures in EventSource handlers
  const isPlayingRef = useRef(isPlaying);
  const autoplayRef = useRef(autoplayOnGenerate);
  isPlayingRef.current = isPlaying;
  autoplayRef.current = autoplayOnGenerate;

  // Track active EventSource instances
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());

  // Unmount-only cleanup — close all SSE connections when the hook is torn down
  useEffect(() => {
    const sources = eventSourcesRef.current;
    return () => {
      for (const source of sources.values()) {
        source.close();
      }
      sources.clear();
    };
  }, []);

  useEffect(() => {
    const currentSources = eventSourcesRef.current;

    // Close SSE connections for IDs no longer pending
    for (const [id, source] of currentSources.entries()) {
      if (!pendingIds.has(id)) {
        source.close();
        currentSources.delete(id);
      }
    }

    // Open SSE connections for new pending IDs
    for (const id of pendingIds) {
      if (currentSources.has(id)) continue;

      const url = apiClient.getGenerationStatusUrl(id);
      const source = new EventSource(url);

      source.onmessage = async (event) => {
        try {
          const data: GenerationStatusEvent = JSON.parse(event.data);

          if (data.status === 'loading_model' || data.status === 'generating') {
            const activeStatus = data.status;
            // The history row is initially cached from POST /generate. Keep
            // that cache synchronized with live SSE events so a model that has
            // started synthesis does not remain visually stuck on
            // "Loading model..." until the entire audio finishes.
            queryClient.setQueriesData<HistoryListResponse>({ queryKey: ['history'] }, (cached) => {
              if (!cached) return cached;
              return {
                ...cached,
                items: cached.items.map((item) =>
                  item.id === data.id
                    ? {
                        ...item,
                        status: activeStatus,
                        progress_current: data.progress_current,
                        progress_total: data.progress_total,
                      }
                    : item,
                ),
              };
            });
          }

          if (data.status === 'completed') {
            source.close();
            currentSources.delete(id);
            removePendingGeneration(id);
            const autoPlaySuppressed = consumeAutoPlaySuppression(id);

            // Refetch history to pick up the completed generation
            queryClient.refetchQueries({ queryKey: ['history'] });

            // Resolve completion metadata once for both story refresh and optional
            // autoplay. target_story_id is persisted by the backend, while the
            // in-memory map remains only as a same-session fast path.
            const cachedGeneration = queryClient
              .getQueriesData<HistoryListResponse>({ queryKey: ['history'] })
              .flatMap(([, cached]) => cached?.items ?? [])
              .find((item) => item.id === id);
            let generation = cachedGeneration;
            if (!generation) {
              try {
                generation = await apiClient.getGeneration(id);
              } catch {
                // Completion itself remains valid if metadata refresh fails.
              }
            }

            const pendingStoryId = removePendingStoryAdd(id);
            const storyId = generation?.target_story_id ?? pendingStoryId;
            if (storyId) {
              // 业务规则：故事项由后端在发布 completed 前创建，前端只刷新
              // 数据，避免 SSE 断线时丢失关联，也避免前后端竞争重复添加。
              await Promise.all([
                queryClient.invalidateQueries({ queryKey: ['stories'] }),
                queryClient.invalidateQueries({ queryKey: ['stories', storyId] }),
              ]);
              toast({
                title: '已添加到故事',
                description: data.duration
                  ? `音频已生成（${data.duration.toFixed(2)} 秒）并添加到故事`
                  : '音频已生成并添加到故事',
              });
            } else {
              // toast({
              //   title: 'Generation complete!',
              //   description: data.duration
              //     ? `Audio generated (${data.duration.toFixed(2)}s)`
              //     : 'Audio generated',
              // });
            }

            // Auto-play if enabled and nothing is currently playing.
            // Skip agent-initiated sources — the floating pill window
            // plays those itself.
            const isAgentSpeak = data.source ? AGENT_SOURCES.has(data.source) : false;
            if (
              autoplayRef.current &&
              !autoPlaySuppressed &&
              !isPlayingRef.current &&
              !isAgentSpeak
            ) {
              const genAudioUrl = apiClient.getAudioUrl(id);
              // 业务规则：自动播放也必须告知用户当前是哪条音频。
              // 优先使用已有历史缓存，缓存缺失时再单独读取任务详情。
              const textSummary = generation?.text.replace(/\s+/g, ' ').trim().slice(0, 50);
              const playerTitle = generation
                ? `${generation.profile_name}${textSummary ? ` · ${textSummary}` : ''}`
                : '新生成的音频';
              setAudioWithAutoPlay(genAudioUrl, id, generation?.profile_id ?? null, playerTitle);
            }
          } else if (data.status === 'failed' || data.status === 'not_found') {
            source.close();
            currentSources.delete(id);
            removePendingGeneration(id);
            consumeAutoPlaySuppression(id);
            removePendingStoryAdd(id);

            queryClient.refetchQueries({ queryKey: ['history'] });

            toast({
              title: data.status === 'not_found' ? '未找到生成任务' : '生成失败',
              description: toChineseErrorMessage(data.error, '生成过程中发生错误'),
              variant: 'destructive',
            });
          }
        } catch {
          // Ignore parse errors from heartbeats etc
        }
      };

      source.onerror = () => {
        // SSE connection dropped — clean up and refresh history so any
        // completed/failed generation still appears in the list
        source.close();
        currentSources.delete(id);
        removePendingGeneration(id);
        consumeAutoPlaySuppression(id);
        queryClient.refetchQueries({ queryKey: ['history'] });
      };

      currentSources.set(id, source);
    }
  }, [
    pendingIds,
    removePendingGeneration,
    removePendingStoryAdd,
    consumeAutoPlaySuppression,
    queryClient,
    toast,
    setAudioWithAutoPlay,
  ]);
}
