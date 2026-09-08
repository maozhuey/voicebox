## Context

PyInstaller 打包后的 `voicebox-server` 把后端所有 Python 模块塞进同一个 PYZ 归档（`pyimod02_importers` 解压 + `marshal.loads` 加载）。当归档被构建中断、磁盘 IO 错误或后续拷贝损坏时，`get_tts_backend_for_engine` 第一次导入引擎模块就会抛出 `zlib.error`。这一异常发生在推理真正开始之前；任务队列会把它当作普通生成异常，把原始 `str(e)` 直接写入 `Generation.error`。前端错误映射对此一无所知，于是 toast 回退到通用 "生成过程中发生错误"，并随 SSE 重新弹出，每次点击"重试"都看到同一条不可行动的提示。

当前生成运行器已经有针对 CosyVoice 阶段和 MLX 原生 `SIGSEGV` 的稳定诊断写入，但 zlib / `ImportError` / `ModuleNotFoundError` 这一家族既未被分类，也未生成诊断，原始异常消息会进入历史和 API 响应。同一家族里 `ModuleNotFoundError` 比 `ImportError` 更具体，必须先匹配；`zlib.error` 必须先于外层 `ImportError` 匹配，否则会被误归为普通的 import 失败。

## Goals / Non-Goals

**Goals:**

- 让后端识别 zlib 错误与 import 家族失败，写入脱敏诊断并以稳定错误码 `BINARY_MODULE_EXTRACTION_FAILED:<diagnostic_id>` 替换原始异常消息。
- 让客户端 toast 把这条稳定码映射为"Voicebox 安装包损坏，请重新安装"的中文说明，并附诊断编号供支持关联。
- 不丢失现有的重试 / 取消 / 终端状态写入链路：稳定码必须走 `update_generation_status(status="failed", error=...)` 的同一路径，任务队列、SSE 与历史 API 完全不变。
- 保证诊断 JSONL 仍然不携带用户正文、参考音频路径、绝对路径、原始异常文本或堆栈。

**Non-Goals:**

- 不试图自动重试或修复损坏的 PYZ；这超出 Voicebox 的能力范围。
- 不区分 macOS / Windows / Linux 上的不同 PyInstaller 异常路径，统一映射到同一稳定码，由支持团队按诊断编号读取本地 JSONL。
- 不修改 PyInstaller 构建参数或 sidecar 打包流程；该修复仅覆盖运行时识别与提示。
- 不重写前端 toast 系统；只在 `toChineseErrorMessage` 中追加一个稳定码分支，并保证它的优先级高于通用回退。

## Decisions

### 1. 使用稳定错误码而非原始 zlib 字符串

将 `BINARY_MODULE_EXTRACTION_FAILED:<diagnostic_id>` 作为历史与 SSE 错误字段的统一值；原始 `zlib.error: Error -3 while decompressing data: incorrect header check` 与 `No module named 'backend.backends.qwen_custom_voice_backend'` 不再进入历史。原因：

- 原始字符串在 Python 不同小版本间会变化（错误码、数字、措辞都不同），无法在客户端稳定匹配。
- 原始字符串会暴露打包内部结构（"PYZ"、"backend.backends.x" 等），对用户和支持人员都没有信息价值。
- 同一稳定码能覆盖多个用户后果相同的失败家族（zlib 损坏、模块缺失、import 错误），客户端文案保持简洁。

### 2. 在生成运行器的 except 分支中先做家族分类

`run_generation` 已经按"CosyVoice 阶段 → MLX 原生 → 其他"分流异常。新增一段 `else` 分支调用 `classify_binary_extraction_failure(e)`：命中时调用 `write_generation_diagnostic(kind="binary_module_extraction_failed", failure_subtype=...)` 并把 `error` 替换为 `diagnostic.error_code`；未命中时保留现有 `error = str(e)`，让真正无关的异常继续保留原始消息供开发日志使用。

分类器必须沿 `__cause__` / `__context__` 链递归，因为 PyInstaller 实际抛出的是 `ImportError("Loader failed")` 外层包 `zlib.error` 内层。优先级：先 `ModuleNotFoundError`，再 `zlib.error`，最后 `ImportError`，否则同一链路会被错归。同时需要 `id(current)` 去重，防止循环 cause chain 永远循环。

### 3. 诊断 JSONL 不写入任何 zlib / import 内部文本

`write_generation_diagnostic` 已经支持 `failure_subtype: str | None` 字段；只为这一类失败写入 `"failure_subtype": "zlib_corruption" | "module_not_found" | "import_error"` 三选一。绝不写入原始 `str(exc)`、模块名、绝对路径或 traceback。客户端只看 `error_code` 的稳定前缀 + 诊断编号。

### 4. 客户端映射在 SSE/Toast 调用之前必须命中

`toChineseErrorMessage(raw, fallback="生成过程中发生错误")` 在 useGenerationProgress 的 SSE failed 分支被调用。新增 `/^BINARY_MODULE_EXTRACTION_FAILED:([A-Za-z0-9-]+)$/` 分支，返回 `"Voicebox 安装包损坏，无法加载该声音引擎。请重新安装 Voicebox 后重试。诊断编号：<id>"`，并显式新增一条"稳定码必须优先于回退"的测试，避免未来有人把通用分支重新提到前面。

### 5. 不改变公共 API 与 SSE payload 形状

只替换 `Generation.error` 的字段值；`GenerationResponse`、`HistoryResponse` 的其他字段、`/generate/{id}/status` 的 SSE payload 形状、`/history` 的列表条目均保持不变。SSE 仍以 `{ "status": "failed", "error": "BINARY_MODULE_EXTRACTION_FAILED:<id>" }` 形式推送，客户端无须做 schema 调整。

## Risks / Trade-offs

- [稳定码不能区分 PYZ 损坏与真正的依赖缺失] → 用 `failure_subtype` 写诊断 JSONL；用户文案保持一致，避免把内部差异暴露给普通用户；支持人员按诊断编号读取 subtype。
- [分类器误判普通 ImportError] → 仅当异常来自 `get_tts_backend_for_engine` 加载路径才映射；其他无关 `ImportError` 不会被现有 except 拦截，行为不变。
- [诊断写入失败] → `write_generation_diagnostic` 已经吞掉 `OSError` 并继续返回错误码，任务队列不会因此挂起。
- [客户端误把已有错误码当成新码] → 正则带 `^` 与行尾锚定，并明确排除已有点号 `:` 后跟 `/N` 的 `GENERATION_*` 家族，避免被新规则截胡。

## Migration Plan

1. 实现 `classify_binary_extraction_failure` 与后端 except 分支，提交单元测试与回归测试。
2. 在 `toChineseErrorMessage` 增加新分支并提交前端测试。
3. 运行 `pytest tests/test_generation_diagnostics.py tests/test_generation_binary_extraction_failure.py` 与 `bun test tests/generationTerminalStatus.test.ts`，确认全绿。
4. 在本地 Docker 构建新的 sidecar 并复现 PYZ 损坏场景，确认历史记录的 `error` 是稳定码、SSE 推送是稳定码、Toast 是中文提示。
5. 若需回滚，保留旧 except 分支路径：`failure_subtype` 不写入，`error` 回退到 `str(e)`；前端稳定码分支删除即可，公共接口零迁移。
