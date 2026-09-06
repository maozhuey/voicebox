## Why

CosyVoice 等异步合成在桌面端重启、队列工作协程异常退出或任务状态写入失败后，可能留下状态为 `loading_model` 或 `generating`、但没有实际执行者的生成记录。此类孤儿记录会长期显示“正在生成”，也会错误地占用任务队列状态，用户只能手工取消。

## What Changes

- 在后端启动和生成队列运行期间，对数据库活动记录、内存任务追踪和实际队列执行状态进行一致性恢复。
- 将无实际队列任务或运行协程的孤儿生成记录安全标记为失败，并写入稳定、可理解的恢复原因；不删除既有生成历史或已产出的音频。
- 为队列工作协程终止、任务跟踪失配和启动恢复补充可观测日志及回归测试，确保新生成任务仍可继续执行。

## Capabilities

### New Capabilities

- `generation-lifecycle-recovery`: 在桌面服务启动和队列异常后自动终结没有实际执行者的活动生成记录。

### Modified Capabilities

- 无。

## Impact

- 影响 `backend/app.py`、`backend/services/task_queue.py`、生成状态持久化及相关后端测试。
- 不新增外部依赖，不更改生成、历史或取消接口的请求格式。
