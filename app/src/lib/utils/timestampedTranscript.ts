import type { TranscriptSegment } from '@/lib/api/types';

function formatTimestamp(milliseconds: number): string {
  const safeMilliseconds = Math.max(0, Math.round(milliseconds));
  const hours = Math.floor(safeMilliseconds / 3_600_000);
  const afterHours = safeMilliseconds % 3_600_000;
  const minutes = Math.floor(afterHours / 60_000);
  const afterMinutes = afterHours % 60_000;
  const seconds = Math.floor(afterMinutes / 1_000);
  const millis = afterMinutes % 1_000;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

/** Export Whisper's raw segment wording; refined prose is intentionally excluded. */
export function buildTimestampedTranscript(segments: TranscriptSegment[]): string {
  return [...segments]
    .sort((left, right) => left.start_ms - right.start_ms || left.end_ms - right.end_ms)
    .map((segment) => ({ ...segment, text: segment.text.trim() }))
    .filter((segment) => segment.text.length > 0)
    .map(
      (segment) =>
        `[${formatTimestamp(segment.start_ms)} --> ${formatTimestamp(segment.end_ms)}] ${segment.text}`,
    )
    .join('\n');
}

type SaveTimestampedFile = (options: {
  defaultPath: string;
  filters: Array<{ name: string; extensions: string[] }>;
}) => Promise<string | null>;

type WriteTimestampedFile = (path: string, contents: string) => Promise<void>;

export type TimestampedTranscriptExportResult =
  | { status: 'missing' }
  | { status: 'cancelled' }
  | { status: 'exported'; path: string };

/** Run the timestamped TXT save flow with injectable desktop file operations. */
export async function exportTimestampedTranscript(
  captureId: string,
  segments: TranscriptSegment[],
  saveFile: SaveTimestampedFile,
  writeFile: WriteTimestampedFile,
): Promise<TimestampedTranscriptExportResult> {
  const contents = buildTimestampedTranscript(segments);
  if (!contents) return { status: 'missing' };

  const path = await saveFile({
    defaultPath: `capture_${captureId.slice(0, 8)}_timestamped.txt`,
    filters: [{ name: 'Timestamped text', extensions: ['txt'] }],
  });
  if (!path) return { status: 'cancelled' };

  await writeFile(path, contents);
  return { status: 'exported', path };
}
