/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $TranscriptionResponse = {
  description: `Response model for transcription.`,
  properties: {
    text: {
      type: 'string',
      isRequired: true,
    },
    duration: {
      type: 'number',
      isRequired: true,
    },
    timestamped_text: {
      type: 'string',
      isRequired: true,
    },
    segments: {
      type: 'array',
      contains: {
        type: 'TranscriptSegment',
      },
      isRequired: true,
    },
  },
} as const;
