## Why

在 Apple Silicon 上，Qwen TTS 的模型加载与推理被调度到不同的后台线程。MLX 将 GPU stream 绑定到创建它的线程，因此推理会失败并返回 `There is no Stream(gpu, 1) in current thread.`，桌面端只能显示笼统的“生成过程中发生错误”。

该问题已在已安装桌面应用的本地服务上复现，必须保证模型加载和后续推理使用同一条 MLX 工作线程。

## What Changes

- 为 MLX TTS 后端提供单一、专用的工作线程，串行执行模型加载和所有 Qwen 推理。
- 保持既有异步接口和生成队列语义，避免阻塞 HTTP 事件循环或并发争用 GPU。
- 为“先加载、再于异步生成路径推理”的线程亲和性行为增加回归测试。

## Capabilities

### New Capabilities

- `mlx-tts-thread-affinity`: 保证 Apple Silicon 上的 MLX Qwen TTS 模型和推理共享同一工作线程与 GPU stream。

### Modified Capabilities

- 无。

## Impact

- 影响 `backend/backends/mlx_backend.py` 的 Qwen TTS 模型加载与生成调度。
- 影响 Apple Silicon 的 Qwen TTS 运行时，不修改请求 API、语音档案或模型文件格式。
- 新增后端回归测试；修复后的侧车需要重新打包并替换本机桌面应用。
