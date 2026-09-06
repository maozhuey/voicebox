import { describe, expect, mock, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import type { VoiceProfileResponse } from '../src/lib/api/types';

let profiles: VoiceProfileResponse[] = [];
let selectedEngine = 'qwen';
let selectedProfileId: string | null = null;

mock.module('@/lib/hooks/useProfiles', () => ({
  useProfiles: () => ({ data: profiles, isLoading: false, error: null }),
}));

mock.module('@/stores/uiStore', () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      selectedEngine,
      selectedProfileId,
      setProfileDialogOpen: () => {},
    }),
}));

mock.module('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

mock.module('@/components/VoiceProfiles/ProfileCard', () => ({
  ProfileCard: ({ profile, disabled }: { profile: VoiceProfileResponse; disabled?: boolean }) => (
    <article
      data-disabled={disabled ? 'true' : 'false'}
      data-profile-id={profile.id}
      data-selected={selectedProfileId === profile.id ? 'true' : 'false'}
    >
      {profile.name}
    </article>
  ),
}));

mock.module('@/components/VoiceProfiles/ProfileForm', () => ({
  ProfileForm: () => null,
}));

const { ProfileList } = await import('../src/components/VoiceProfiles/ProfileList');

const firstProfiles: VoiceProfileResponse[] = [
  {
    id: 'clone-1',
    name: '我',
    language: 'zh',
    voice_type: 'cloned',
    generation_count: 0,
    sample_count: 1,
    created_at: '2026-09-06T00:00:00Z',
    updated_at: '2026-09-06T00:00:00Z',
  },
  {
    id: 'qwen-1',
    name: '预设-Vivian',
    language: 'zh',
    voice_type: 'preset',
    preset_engine: 'qwen_custom_voice',
    generation_count: 0,
    sample_count: 0,
    created_at: '2026-09-06T00:00:00Z',
    updated_at: '2026-09-06T00:00:00Z',
  },
  {
    id: 'kokoro-1',
    name: '预设-AF',
    language: 'en',
    voice_type: 'preset',
    preset_engine: 'kokoro',
    generation_count: 0,
    sample_count: 0,
    created_at: '2026-09-06T00:00:00Z',
    updated_at: '2026-09-06T00:00:00Z',
  },
];

function renderCards() {
  const html = renderToStaticMarkup(<ProfileList />);
  return [...html.matchAll(/<article([^>]*)>/g)].map(([, attributes]) => {
    const valueFor = (name: string) => attributes.match(new RegExp(`${name}="([^"]+)"`))?.[1];
    return {
      id: valueFor('data-profile-id'),
      disabled: valueFor('data-disabled'),
      selected: valueFor('data-selected'),
    };
  });
}

describe('ProfileList card order', () => {
  test('keeps API order when selecting and deselecting a non-first profile', () => {
    profiles = firstProfiles;
    selectedEngine = 'qwen_custom_voice';
    selectedProfileId = null;
    const initialCards = renderCards();

    // ProfileCard 点击后只更新选中档案；这里模拟该已存在的状态更新并验证列表不会重排。
    selectedProfileId = 'qwen-1';
    const selectedCards = renderCards();

    selectedProfileId = null;
    const deselectedCards = renderCards();

    expect(selectedCards.map((card) => card.id)).toEqual(initialCards.map((card) => card.id));
    expect(selectedCards[1]).toEqual({ id: 'qwen-1', disabled: 'false', selected: 'true' });
    expect(deselectedCards).toEqual(initialCards);
  });

  test('keeps order while an engine change updates unsupported cards', () => {
    profiles = firstProfiles;
    selectedProfileId = 'qwen-1';
    selectedEngine = 'qwen_custom_voice';
    const initialCards = renderCards();

    selectedEngine = 'kokoro';
    const updatedCards = renderCards();

    expect(updatedCards.map((card) => card.id)).toEqual(initialCards.map((card) => card.id));
    expect(updatedCards).toEqual([
      { id: 'clone-1', disabled: 'true', selected: 'false' },
      { id: 'qwen-1', disabled: 'true', selected: 'true' },
      { id: 'kokoro-1', disabled: 'false', selected: 'false' },
    ]);
  });

  test('uses the refreshed API order when the profile collection changes', () => {
    profiles = firstProfiles;
    selectedEngine = 'kokoro';
    selectedProfileId = null;
    renderCards();

    profiles = [firstProfiles[2], firstProfiles[0]];

    expect(renderCards().map((card) => card.id)).toEqual(['kokoro-1', 'clone-1']);
  });
});
