import { describe, expect, test } from 'bun:test';

import { createPlatform } from '../src/platform/platform';

describe('platform bootstrap', () => {
  test('provides a browser platform so the development page can render outside Tauri', async () => {
    const platform = createPlatform(false);

    expect(platform.metadata.isTauri).toBe(false);
    expect(await platform.metadata.getVersion()).toBe('开发版');
    expect(await platform.audio.isSystemAudioSupported()).toBe(false);
    expect(platform.updater.getStatus()).toEqual({
      checking: false,
      available: false,
      downloading: false,
      installing: false,
      readyToInstall: false,
    });
  });
});
