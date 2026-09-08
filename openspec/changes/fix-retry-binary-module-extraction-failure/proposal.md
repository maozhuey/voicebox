# 修复重试失败音频的安装包损坏提示

## Why

历史列表中失败的音频点击"重试"按钮后，前端再次收到通用提示"生成失败 / 生成过程中发生错误"。后端实际错误是 PyInstaller 打包的 sidecar 在加载声音引擎模块时抛出 `zlib.error: Error -3 while decompressing data: incorrect header check`，原因是构建或复制过程中 PYZ 归档损坏。原始 zlib 错误消息既不告诉用户真正发生了什么，也不可在不同 Python 版本中保持稳定；而客户端把这条消息回退到通用 toast，让用户误以为只是网络或模型问题，反复点击"重试"始终得到相同结果。

PyInstaller 打包后的 `voicebox-server` 把所有后端模块塞进同一个 PYZ 归档，加载引擎时 `import backend.backends.<x>` 就会先解压缩该归档；如果归档损坏，异常发生在推理真正开始之前，所以重试永远不可能成功。同时，`ModuleNotFoundError`、`ImportError` 这两个家族也会在 sidecar 启动期被抛出，落到相同的"模块加载失败"用户后果。

后端生成运行器已经能识别 CosyVoice、MLX 终端错误并写入安全诊断，但 zlib / import 家族被遗漏，原始异常直接进入 `Generation.error`。前端错误映射有 `CAPTURE_DIAGNOSTIC`、`GENERATION_TERMINAL_STATUS_MISSING`、`GENERATION_WORKER_EXITED` 等稳定码，唯独没有 PyInstaller 安装包损坏这一类。

## What Changes

- 在生成运行器中识别 zlib、import 家族失败，并把它映射为稳定错误码 `BINARY_MODULE_EXTRACTION_FAILED:<diagnostic_id>`，同时写入脱敏本地诊断。
- 把这一类失败和已有的 `terminal_status_missing`、`worker_exited`、`server_exited` 一样当作"可重试但需要重新安装"的提示，而不是通用"生成过程中发生错误"。
- 客户端识别 `BINARY_MODULE_EXTRACTION_FAILED` 错误码，提示用户 Voicebox 安装包损坏，需要重新安装 Voicebox；原始 zlib 字符串、模块名和堆栈不再写入历史或暴露给用户。
- 增加覆盖 `zlib.error`、`ModuleNotFoundError` 在顶层与 `__cause__` 链中、`ImportError`、不相关 `RuntimeError` 以及 cause chain 循环的回归测试，并验证前端映射优先级压倒通用回退。

## Capabilities

### New Capabilities

- `binary-module-extraction-failure`: 在 PyInstaller PYZ 损坏或 sidecar 模块加载失败时给出可操作的稳定诊断，并保证历史与 UI 不泄露 zlib / import 内部信息。

## Impact

- 后端：`backend/services/generation.py`、`backend/services/generation_diagnostics.py`，新增/扩展 zlib 与 import 家族分类器与诊断写入；新增 `backend/tests/test_generation_binary_extraction_failure.py` 和扩展 `backend/tests/test_generation_diagnostics.py`。
- 前端：`app/src/lib/utils/errorMessage.ts` 增加稳定码到中文文案映射；`app/tests/generationTerminalStatus.test.ts` 增加映射优先级回归。
- 公开接口：HTTP 与 SSE 行为不变；`Generation.error` 中保存的字段值从原始 zlib 字符串变为稳定错误码；不影响模型选择、请求参数或历史数据结构。
