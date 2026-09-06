/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TranscriptSegment } from './TranscriptSegment';
/**
 * Response model for transcription.
 */
export type TranscriptionResponse = {
  text: string;
  duration: number;
  timestamped_text: string;
  segments: Array<TranscriptSegment>;
};
