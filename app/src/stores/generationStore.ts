import { create } from 'zustand';

interface GenerationState {
  /** IDs of generations currently in progress */
  pendingGenerationIds: Set<string>;
  /** Whether any generation is in progress (derived from pendingGenerationIds) */
  isGenerating: boolean;
  /** Map of generationId → storyId for deferred story additions */
  pendingStoryAdds: Map<string, string>;
  /** Pending tasks that must not reopen the player after the user closes it. */
  suppressedAutoPlayIds: Set<string>;
  addPendingGeneration: (id: string) => void;
  removePendingGeneration: (id: string) => void;
  suppressAutoPlayForPending: () => void;
  consumeAutoPlaySuppression: (id: string) => boolean;
  addPendingStoryAdd: (generationId: string, storyId: string) => void;
  removePendingStoryAdd: (generationId: string) => string | undefined;
  setActiveGenerationId: (id: string | null) => void;
  activeGenerationId: string | null;
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  pendingGenerationIds: new Set(),
  isGenerating: false,
  activeGenerationId: null,
  pendingStoryAdds: new Map(),
  suppressedAutoPlayIds: new Set(),

  addPendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.add(id);
      return { pendingGenerationIds: next, isGenerating: true };
    }),

  removePendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.delete(id);
      return { pendingGenerationIds: next, isGenerating: next.size > 0 };
    }),

  // 业务规则：用户主动关闭播放器时，只屏蔽当时已在队列中的任务。
  // 之后新建的任务仍遵循“生成后自动播放”设置。
  suppressAutoPlayForPending: () =>
    set((state) => ({
      suppressedAutoPlayIds: new Set([
        ...state.suppressedAutoPlayIds,
        ...state.pendingGenerationIds,
      ]),
    })),

  consumeAutoPlaySuppression: (id) => {
    const suppressed = get().suppressedAutoPlayIds.has(id);
    if (suppressed) {
      set((state) => {
        const next = new Set(state.suppressedAutoPlayIds);
        next.delete(id);
        return { suppressedAutoPlayIds: next };
      });
    }
    return suppressed;
  },

  addPendingStoryAdd: (generationId, storyId) =>
    set((state) => {
      const next = new Map(state.pendingStoryAdds);
      next.set(generationId, storyId);
      return { pendingStoryAdds: next };
    }),

  removePendingStoryAdd: (generationId) => {
    const storyId = get().pendingStoryAdds.get(generationId);
    if (storyId) {
      set((state) => {
        const next = new Map(state.pendingStoryAdds);
        next.delete(generationId);
        return { pendingStoryAdds: next };
      });
    }
    return storyId;
  },

  setActiveGenerationId: (id) => set({ activeGenerationId: id }),
}));
