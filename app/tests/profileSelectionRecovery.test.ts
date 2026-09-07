import { describe, expect, test } from 'bun:test';

import {
  resolveEditingProfileQueryId,
  resolveSelectedProfileId,
} from '../src/lib/utils/profileSelection';
import { toChineseErrorMessage } from '../src/lib/utils/errorMessage';

const profiles = [{ id: 'vivian' }, { id: 'han' }];

describe('profile selection recovery', () => {
  test('waits for the profile list before choosing an id', () => {
    expect(resolveSelectedProfileId('deleted-profile', undefined)).toBeNull();
  });

  test('keeps a saved selection when it still belongs to the connected server', () => {
    expect(resolveSelectedProfileId('han', profiles)).toBe('han');
  });

  test('replaces a stale saved selection with the first profile from the connected server', () => {
    expect(resolveSelectedProfileId('deleted-profile', profiles)).toBe('vivian');
  });

  test('clears the selection when the connected server has no profiles', () => {
    expect(resolveSelectedProfileId('deleted-profile', [])).toBeNull();
  });

  test('turns a rejected stale profile id into an actionable message', () => {
    expect(toChineseErrorMessage('Profile not found')).toBe(
      '所选声音档案已失效，请重新选择声音后再试。',
    );
  });

  test('does not fetch an obsolete edit target while the persistent form dialog is closed', () => {
    expect(resolveEditingProfileQueryId(false, 'deleted-profile')).toBeNull();
  });

  test('fetches the edit target only while its dialog is open', () => {
    expect(resolveEditingProfileQueryId(true, 'existing-profile')).toBe('existing-profile');
  });
});
