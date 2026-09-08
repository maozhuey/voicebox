## 1. 后端分类与诊断

- [x] 1.1 在 `backend/services/generation_diagnostics.py` 中新增 `BINARY_MODULE_EXTRACTION_FAILED` 稳定错误码、`binary_module_extraction_failed` 诊断类型和 `BinaryFailureSubtype` 三态枚举；通过单元测试验证错误码与类型名一一对应。
- [x] 1.2 在 `write_generation_diagnostic` 中允许 `failure_subtype` 字段并把它写入诊断 JSONL；通过测试验证该字段仅在传入时出现，且永不写入原始异常文本或用户正文。
- [x] 1.3 在 `backend/services/generation_diagnostics.py` 中实现 `classify_binary_extraction_failure`，按 `ModuleNotFoundError → zlib.error → ImportError` 顺序匹配并沿 `__cause__` / `__context__` 链递归；通过参数化与循环 cause chain 测试验证优先级、链路穿透和循环终止。

## 2. 生成运行器接入

- [x] 2.1 在 `backend/services/generation.py` 的 except 分支中新增 zlib / import 家族映射：调用 `classify_binary_extraction_failure` 命中时写入诊断并以 `diagnostic.error_code` 替换 `error`；通过测试验证 `zlib.error` 顶层、`ImportError` 包裹 `zlib.error`、`ModuleNotFoundError` 顶层与不相关 `RuntimeError` 四种情形分别得到正确错误码或原始消息。
- [x] 2.2 保留"重试 / 取消 / 正常完成 / MLX / CosyVoice"既有路径的最终错误码不变；通过 `pytest tests/test_task_queue_cancellation.py` 验证既有行为未被破坏。

## 3. 客户端错误映射

- [x] 3.1 在 `app/src/lib/utils/errorMessage.ts` 中识别 `BINARY_MODULE_EXTRACTION_FAILED:<id>` 稳定码并返回中文可行动文案 + 诊断编号；通过 `app/tests/generationTerminalStatus.test.ts` 新增两个用例验证稳定码命中并压倒通用回退。
- [x] 3.2 不修改 Toast 触发链路；通过现有 `useGenerationProgress` 单测或类型检查确认新文案进入 SSE failed 分支后仍走原有 toast 路径。

## 4. 整体验证

- [x] 4.1 运行后端 `pytest tests/test_generation_diagnostics.py tests/test_generation_binary_extraction_failure.py tests/test_audio_failed_generation.py` 并确认全部通过。
- [x] 4.2 运行前端 `bun test tests/generationTerminalStatus.test.ts` 与 `bun run typecheck`，确认无新增类型错误且 4 个用例全绿。
- [x] 4.3 在本地 Docker 中构建新的 sidecar，模拟 PYZ 损坏（如重命名 PyInstaller 内部 PYZ 副本）并触发一次生成；确认历史记录 `error` 是 `BINARY_MODULE_EXTRACTION_FAILED:<id>`、SSE payload 一致、Toast 是"Voicebox 安装包损坏"中文。
- [x] 4.4 执行 `openspec validate fix-retry-binary-module-extraction-failure --strict`，确认提案、设计、规格和任务清单均通过严格校验。
