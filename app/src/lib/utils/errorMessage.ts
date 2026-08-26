const CHINESE_TEXT = /[\u3400-\u9fff]/;

/**
 * 将可能直接展示在弹窗或 Toast 中的底层错误转换为简体中文。
 * 原始英文错误仍应写入开发日志；面向用户时优先给出可执行、可理解的中文说明。
 */
export function toChineseErrorMessage(error: unknown, fallback = '操作失败，请稍后重试。'): string {
  const raw =
    typeof error === 'string' ? error.trim() : error instanceof Error ? error.message.trim() : '';

  if (!raw) return fallback;
  if (CHINESE_TEXT.test(raw)) return raw;

  let match = raw.match(/^Engine ['"]?([^'"]+)['"]? does not support cloned voice profiles$/i);
  if (match) return `引擎“${match[1]}”不支持克隆声音档案`;

  match = raw.match(/^No module named ['"]([^'"]+)['"]$/i);
  if (match) return `缺少运行模块“${match[1]}”，请安装依赖后重试。`;

  if (/TorchCodec is required/i.test(raw)) {
    return '当前功能需要 TorchCodec，请安装对应依赖后重试。';
  }
  if (/failed to fetch|networkerror|network request failed|load failed/i.test(raw)) {
    return '无法连接本地服务，请确认服务已启动并检查网络连接。';
  }
  if (/profile not found/i.test(raw)) return '未找到声音档案。';
  if (/generation not found/i.test(raw)) return '未找到生成任务。';
  if (/story not found/i.test(raw)) return '未找到故事。';
  if (/only failed generations can be retried/i.test(raw)) return '只能重试生成失败的任务。';
  if (/field required/i.test(raw)) return '缺少必填内容，请检查后重试。';
  if (/already downloaded/i.test(raw)) return '该资源已经下载。';
  if (/not downloaded/i.test(raw)) return '所需模型尚未下载，请先下载模型。';

  match = raw.match(/HTTP(?: error! status:)?\s*(\d{3})/i);
  if (match) return `请求失败（状态码 ${match[1]}）。`;

  return fallback;
}
