import { describe, expect, test } from 'bun:test';
import {
  buildTimestampedTranscript,
  exportTimestampedTranscript,
} from '../src/lib/utils/timestampedTranscript';

describe('buildTimestampedTranscript', () => {
  test('formats original segments with stable hour-level timecodes', () => {
    expect(
      buildTimestampedTranscript([
        { start_ms: 3_599_999, end_ms: 3_601_004, text: ' 跨小时 ' },
        { start_ms: 0, end_ms: 1_250, text: '第一句' },
      ]),
    ).toBe('[00:00:00.000 --> 00:00:01.250] 第一句\n' + '[00:59:59.999 --> 01:00:01.004] 跨小时');
  });

  test('ignores empty segment text and returns an empty document when none remain', () => {
    expect(buildTimestampedTranscript([{ start_ms: 0, end_ms: 100, text: '  ' }])).toBe('');
    expect(buildTimestampedTranscript([])).toBe('');
  });
});

describe('exportTimestampedTranscript', () => {
  const segments = [{ start_ms: 0, end_ms: 1_250, text: 'Whisper 原文' }];

  test('writes the raw timestamped document with the expected filename', async () => {
    let saveOptions: unknown;
    let writeCall: unknown;
    const result = await exportTimestampedTranscript(
      '12345678-aaaa-bbbb-cccc-dddddddddddd',
      segments,
      async (options) => {
        saveOptions = options;
        return '/tmp/result.txt';
      },
      async (path, contents) => {
        writeCall = { path, contents };
      },
    );

    expect(saveOptions).toEqual({
      defaultPath: 'capture_12345678_timestamped.txt',
      filters: [{ name: 'Timestamped text', extensions: ['txt'] }],
    });
    expect(writeCall).toEqual({
      path: '/tmp/result.txt',
      contents: '[00:00:00.000 --> 00:00:01.250] Whisper 原文',
    });
    expect(result).toEqual({ status: 'exported', path: '/tmp/result.txt' });
  });

  test('does not write when the save dialog is cancelled', async () => {
    let writes = 0;
    const result = await exportTimestampedTranscript(
      '12345678',
      segments,
      async () => null,
      async () => {
        writes += 1;
      },
    );

    expect(result).toEqual({ status: 'cancelled' });
    expect(writes).toBe(0);
  });

  test('does not open the save dialog when timestamp segments are missing', async () => {
    let saves = 0;
    const result = await exportTimestampedTranscript(
      '12345678',
      [],
      async () => {
        saves += 1;
        return '/tmp/result.txt';
      },
      async () => {},
    );

    expect(result).toEqual({ status: 'missing' });
    expect(saves).toBe(0);
  });

  test('propagates write errors so the UI can display its error toast', async () => {
    await expect(
      exportTimestampedTranscript(
        '12345678',
        segments,
        async () => '/tmp/result.txt',
        async () => {
          throw new Error('disk full');
        },
      ),
    ).rejects.toThrow('disk full');
  });
});
