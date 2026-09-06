/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $TranscriptSegment = {
  description: `One Whisper-native text segment on the source audio timeline.`,
  properties: {
    start_ms: {
      type: 'number',
      isRequired: true,
    },
    end_ms: {
      type: 'number',
      isRequired: true,
    },
    text: {
      type: 'string',
      isRequired: true,
    },
  },
} as const;
