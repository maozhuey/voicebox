import { describe, expect, test } from 'bun:test';

import { toChineseErrorMessage } from '../src/lib/utils/errorMessage';

describe('generation terminal-status diagnostics', () => {
  test('shows segment progress and a retryable diagnostic for an unpublished terminal state', () => {
    expect(
      toChineseErrorMessage('GENERATION_TERMINAL_STATUS_MISSING:diag-123:1/3'),
    ).toBe('合成在第 1/3 段中断，请重试。诊断编号：diag-123');
  });

  test('distinguishes a worker exit from a local service exit', () => {
    expect(toChineseErrorMessage('GENERATION_WORKER_EXITED:diag-worker:2/3')).toBe(
      '合成任务异常结束（第 2/3 段），请重试。诊断编号：diag-worker',
    );
    expect(toChineseErrorMessage('GENERATION_SERVER_EXITED:diag-server:1/3')).toBe(
      '本地服务异常结束，合成已中断（第 1/3 段），请重试。诊断编号：diag-server',
    );
  });
});
