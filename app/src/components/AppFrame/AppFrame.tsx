import { useRouterState } from '@tanstack/react-router';
import { AudioKeepAlive } from '@/components/AudioPlayer/AudioKeepAlive';
import { AudioPlayer } from '@/components/AudioPlayer/AudioPlayer';
import { StoryTrackEditor } from '@/components/StoriesTab/StoryTrackEditor';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { TOP_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useStory } from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { useStoryStore } from '@/stores/storyStore';

interface AppFrameProps {
  children: React.ReactNode;
}

export function AppFrame({ children }: AppFrameProps) {
  const platform = usePlatform();
  const routerState = useRouterState();
  const isStoriesRoute = routerState.location.pathname === '/stories';

  const selectedStoryId = useStoryStore((state) => state.selectedStoryId);
  const { data: story } = useStory(selectedStoryId);

  // Show track editor when on stories route with a selected story that has items
  const showTrackEditor = isStoriesRoute && selectedStoryId && story && story.items.length > 0;

  return (
    <div
      className={cn('h-screen bg-background flex flex-col overflow-hidden', TOP_SAFE_AREA_PADDING)}
    >
      <TitleBarDragRegion />
      {/* 业务规则：静音音频保活只用于 Tauri/WKWebView 的 CoreAudio 通道。
          网页端不启动，避免浏览器一直显示“正在播放”且无可见音频可关闭。 */}
      {platform.metadata.isTauri && <AudioKeepAlive />}
      {children}
      {showTrackEditor ? (
        <StoryTrackEditor storyId={story.id} items={story.items} />
      ) : (
        <AudioPlayer />
      )}
    </div>
  );
}
