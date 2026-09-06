## Purpose

确保 Apple Silicon 桌面端的 Qwen TTS 在模型首次加载后能够稳定生成语音，不因 MLX GPU stream 的线程归属而产生仅显示为“生成过程中发生错误”的失败。

## ADDED Requirements

### Requirement: MLX Qwen TTS 保持模型与推理线程一致

当系统在 MLX 后端上使用 Qwen TTS 时，系统 SHALL 在模型加载后将该模型的后续推理调度到同一条专用工作线程，且 SHALL 不因普通异步任务调度切换线程而失去 GPU stream。

#### Scenario: 首次加载后的 Qwen 生成

- **WHEN** Apple Silicon 上的用户首次使用已缓存的 Qwen TTS 模型生成语音
- **THEN** 系统 SHALL 完成模型加载并在同一工作线程执行推理，不返回 `There is no Stream(gpu, 1) in current thread.`

#### Scenario: 连续生成复用已加载模型

- **WHEN** Qwen TTS 模型已加载且用户再次提交生成请求
- **THEN** 系统 SHALL 复用同一工作线程执行推理，并保持原有异步请求响应与生成队列行为

### Requirement: MLX 线程亲和性失败可回归验证

系统 MUST 提供自动化回归测试，验证加载操作与生成操作通过同一 MLX 工作执行器调度，防止未来将二者重新分发到不受约束的线程池。

#### Scenario: 测试检测加载和生成的执行器归属

- **WHEN** 自动化测试模拟 MLX TTS 的加载和生成调用
- **THEN** 测试 MUST 断言两次调用使用同一个专用执行器，且不依赖默认线程池
