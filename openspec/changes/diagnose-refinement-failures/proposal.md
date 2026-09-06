## Why

一次听写可以已经完成录音保存和 Whisper 转录，却在随后自动精修时收到通用 500。当前精修路由不记录诊断，客户端又把该后续失败显示为整次听写失败，用户既无法知道原始转录仍可用，也无法定位 Qwen 模型加载、生成或保存中的真实故障点。

## What Changes

- 将捕获精修失败纳入现有的本地安全诊断机制，按模型加载、LLM 生成和结果保存记录可关联的阶段与非敏感错误摘要。
- 精修失败时保留已成功写入的原始转录、音频和捕获记录；自动精修失败不得撤销或误标记已成功的听写。
- 客户端将自动精修失败明确显示为“转录成功，精修失败，已保留原始转录”及诊断编号，而不是“请求失败（状态码 500）”。
- 自动精修失败时，以原始转录完成既有的复制/自动粘贴交付；手动精修失败时保留当前捕获内容并展示同样安全的可操作错误。
- 增加覆盖精修模型加载、LLM 生成、保存失败与客户端原始文本回退的回归测试。

## Capabilities

### New Capabilities

- `capture-refinement-recovery`: 保证成功转录后的精修失败可诊断、可恢复且不会误报为整次听写失败。

### Modified Capabilities

- 无。现有 `capture-failure-diagnostics` 仅存在于尚未归档的历史变更中，尚无可修改的主规格；本变更以独立能力定义后续精修的可靠性契约。

## Impact

- 后端：`backend/routes/captures.py`、`backend/services/captures.py`、`backend/services/refinement.py` 与 `backend/services/capture_diagnostics.py` 的精修错误分层和安全诊断。
- 前端：`app/src/lib/hooks/useCaptureRecordingSession.ts`、错误信息映射及相关捕获测试。
- 测试：扩展后端捕获诊断回归测试和前端自动精修失败回退测试；不增加云端依赖、不修改既有音频或成功捕获数据。
