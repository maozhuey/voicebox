## Why

CosyVoice 3 RL 在 16 GiB Apple Silicon 桌面端与已常驻的 Qwen/MLX 模型共存时，会同时占用 CPU 模型内存和 Metal 图形内存，导致大量交换；其上游非流式推理又会无期限等待内部解码线程，最终让单条生成永久占用串行队列。现有状态只显示“生成中”，无法判断卡在前处理、LLM 解码、Flow 还是声码器。

## What Changes

- 在加载 CosyVoice 前卸载其他已加载的 TTS 后端并释放其设备缓存，避免跨引擎模型同时常驻。
- 为 CosyVoice 的每个生成分段设置可恢复的执行时限；超时后持久化失败状态并释放生成队列，不再让不可取消的上游线程无限阻塞后续任务。
- 让 CosyVoice 仅按自然语义边界分段并保留自然朗读的停顿语义；无语义边界的文本不按字符数强制截断，而由可终止时限保护。
- 在任务状态中持久化 CosyVoice 的运行阶段和阶段耗时，区分前处理、LLM 解码、Flow 与声码器，且失败信息不暴露内部实现细节。

## Capabilities

### New Capabilities

- `cosyvoice-runtime-stability`: 桌面端 CosyVoice 在受限内存环境下进行可恢复、可诊断的生成，并在切换引擎时避免模型资源叠加。

### Modified Capabilities

- 无。现有主规格中没有可修改的生成看门狗能力；本变更把 CosyVoice 的可恢复超时行为作为新运行时能力定义。

## Impact

- 受影响代码：`backend/backends/__init__.py`、`backend/backends/cosyvoice_backend.py`、`backend/services/generation.py`、`backend/services/task_queue.py`、`backend/utils/chunked_tts.py`、任务状态模型与相关测试。
- 受影响 API：生成状态 SSE 和历史任务记录将增加兼容的诊断字段；既有生成请求字段保持兼容。
- 受影响系统：macOS 桌面端冻结侧车需要重新构建、签名和本机替换；不会修改模型缓存、声音档案或线上环境。
