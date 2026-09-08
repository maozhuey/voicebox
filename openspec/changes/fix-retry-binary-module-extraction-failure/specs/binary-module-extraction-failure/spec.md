## Purpose

保证 PyInstaller PYZ 损坏或 sidecar 模块加载失败（zlib / import 家族）被识别为可重试但需要重新安装的稳定失败，并通过脱敏本地诊断和客户端中文提示把这一类失败与普通的网络、模型加载或终端状态丢失失败区分开。

## ADDED Requirements

### Requirement: 稳定错误码与诊断写入

当生成运行器在 `run_generation` 顶层捕获到一个属于 PyInstaller 二进制模块提取家族的异常时，系统 MUST 将异常分类为以下三种 subtype 之一并把 subtype 写入本地诊断 JSONL，同时把 `Generation.error` 字段替换为稳定错误码 `BINARY_MODULE_EXTRACTION_FAILED:<diagnostic_id>`：

- `zlib_corruption`：异常链中存在 `zlib.error`（包括 `Error -3 while decompressing data` 系列消息）。
- `module_not_found`：异常链中存在 `ModuleNotFoundError`。
- `import_error`：异常链中存在 `ImportError` 但不包含以上两类。

`Generation.error` MUST NOT 保留原始异常字符串。诊断 JSONL MUST 仅包含生成 ID、引擎、模型、生命周期类别（`unexpected`）、subtype 与时间戳；MUST NOT 包含用户正文、参考音频路径、绝对路径、原始异常消息或堆栈。

#### Scenario: 顶层 zlib 错误映射为稳定码

- **WHEN** 引擎模块加载器在顶层抛出 `zlib.error("Error -3 while decompressing data: incorrect header check")`
- **THEN** `Generation.error` MUST 等于 `BINARY_MODULE_EXTRACTION_FAILED:<diagnostic_id>`，诊断 MUST 包含 `failure_subtype = "zlib_corruption"` 且 MUST NOT 包含原始 zlib 消息

#### Scenario: 异常链中的 zlib 错误仍被识别

- **WHEN** 顶层异常是 `ImportError`，其 `__cause__` 链中存在 `zlib.error`
- **THEN** 系统 MUST 把它归类为 `zlib_corruption`，`Generation.error` MUST 写入稳定错误码而非 `ImportError` 文案

#### Scenario: ModuleNotFoundError 优先级高于 ImportError

- **WHEN** 顶层异常是 `ImportError`，其 `__cause__` 链中存在 `ModuleNotFoundError`
- **THEN** 系统 MUST 把它归类为 `module_not_found`，而非退化为 `import_error`

#### Scenario: 循环 cause chain 安全返回 None

- **WHEN** 异常的 `__cause__` 链构成环（如 `a.__cause__ = b; b.__cause__ = a`）
- **THEN** 分类器 MUST 在有限步内返回 `None` 而非无限递归，原始异常文案保持不变

### Requirement: 与既有失败家族不混淆

非 zlib / import 家族的异常（例如普通 `RuntimeError`、MLX 原生 `SIGSEGV` 包装的 `RuntimeError`、CosyVoice 阶段错误） MUST 保留其原始终态行为和错误字段值。`BINARY_MODULE_EXTRACTION_FAILED` 稳定码 MUST 仅在该家族被识别时写入；不得影响既有 `GENERATION_TERMINAL_STATUS_MISSING`、`GENERATION_WORKER_EXITED`、`GENERATION_SERVER_EXITED`、`CAPTURE_DIAGNOSTIC` 等稳定码。

#### Scenario: 不相关 RuntimeError 保留原始消息

- **WHEN** 引擎模块加载抛出普通 `RuntimeError("model produced empty audio")`
- **THEN** `Generation.error` MUST 等于 `"model produced empty audio"`，MUST NOT 写入 `BINARY_MODULE_EXTRACTION_FAILED`，且 MUST NOT 写入诊断 JSONL

### Requirement: 客户端中文提示

`app/src/lib/utils/errorMessage.ts` 的 `toChineseErrorMessage` MUST 在收到 `BINARY_MODULE_EXTRACTION_FAILED:<id>` 字符串时返回包含"Voicebox 安装包损坏"和"重新安装 Voicebox 后重试"的中文文案，并附诊断编号。

`toChineseErrorMessage` 对该稳定码的匹配 MUST 优先于任何通用回退（如 `生成过程中发生错误` 或 `操作失败，请稍后重试`），确保用户始终看到可操作的安装包修复建议，而非"稍后重试"的占位文案。

#### Scenario: 稳定码命中后返回安装包损坏提示

- **WHEN** `toChineseErrorMessage("BINARY_MODULE_EXTRACTION_FAILED:diag-pyz")` 被调用
- **THEN** 返回值 MUST 包含"Voicebox 安装包损坏"、"重新安装 Voicebox"和诊断编号 `diag-pyz`

#### Scenario: 稳定码压倒通用回退

- **WHEN** `toChineseErrorMessage("BINARY_MODULE_EXTRACTION_FAILED:diag-pyz", "生成过程中发生错误")` 被调用
- **THEN** 返回值 MUST 包含"重新安装 Voicebox"且 MUST NOT 等于 `"生成过程中发生错误"`

### Requirement: 不改变公共接口

`GenerationResponse`、`HistoryResponse`、`/generate/{id}/status` SSE payload 与 `/history` 列表条目的字段集 MUST 保持不变。`Generation.error` 的字段值从原始 zlib 字符串替换为稳定错误码被视为字段值变更，不被视为 schema 变更。

模型选择、请求参数、声音档案、故事附件与已有历史记录的访问路径 MUST 不受本次变更影响。

#### Scenario: SSE payload 形状保持

- **WHEN** 后端把一条失败的生成写入稳定错误码并通过 SSE 推送
- **THEN** SSE payload 的字段集（MUST 包含 `id`、`status`、`duration`、`error`、`progress_current`、`progress_total`、`source`） MUST 与变更前保持一致，仅 `error` 字段值变化
