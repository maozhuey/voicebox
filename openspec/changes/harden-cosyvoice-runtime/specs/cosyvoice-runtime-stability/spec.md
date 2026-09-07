## Purpose

让 macOS 桌面端的 CosyVoice 生成在内存受限或上游推理异常时仍可恢复，并向用户和 API 调用方提供不含文本内容的阶段化诊断信息。

## ADDED Requirements

### Requirement: CosyVoice 加载前的 TTS 资源隔离
系统在开始 CosyVoice 模型加载前 SHALL 卸载所有其他已加载的 TTS 引擎，并释放其可释放的运行时与设备缓存；系统 MUST 不卸载当前待执行的 CosyVoice 实例。系统 MUST NOT 卸载持有活动执行租约的其他 TTS 引擎；此时 CosyVoice 请求必须等待该租约释放后再回收资源和加载模型。

#### Scenario: Qwen 后切换到 CosyVoice
- **WHEN** Qwen TTS 已加载且用户提交 CosyVoice 生成
- **THEN** 系统先卸载 Qwen TTS 并执行缓存回收，再开始 CosyVoice 的模型加载

#### Scenario: 没有其他 TTS 模型常驻
- **WHEN** 用户提交 CosyVoice 生成且没有其他已加载的 TTS 引擎
- **THEN** 系统继续加载 CosyVoice，且不因资源回收步骤产生错误

#### Scenario: 其他 TTS 正在生成
- **WHEN** Qwen 或其他 TTS 引擎持有活动执行租约，且用户提交 CosyVoice 生成
- **THEN** 系统不卸载该正在执行的模型、不使其任务失败，并在其租约释放后再卸载空闲模型和加载 CosyVoice

### Requirement: CosyVoice 安全分段
系统 SHALL 仅在段落、句末或强停顿等自然语义边界拆分 CosyVoice 请求，并按原始顺序合成。系统 MUST NOT 因固定字符数上限强制截断没有自然边界的文本；自然朗读模式 MUST 保留原文段落与标点产生的停顿语义。

#### Scenario: 含自然边界的普通长文本生成
- **WHEN** CosyVoice 收到包含句末或强停顿的长文本
- **THEN** 系统按这些自然语义边界切分为多个分段并顺序生成

#### Scenario: 没有自然边界的长文本
- **WHEN** CosyVoice 收到没有段落、句末或强停顿的长文本
- **THEN** 系统将原文作为一个分段执行，且以独立的可终止执行时限防止任务永久阻塞

#### Scenario: 自然朗读长文本生成
- **WHEN** CosyVoice 自然朗读收到含句末标点、分号或段落换行的长文本
- **THEN** 系统在安全分段后保留相应的句间、转折和段落停顿

### Requirement: 可终止的 CosyVoice 分段执行
系统 MUST 对每个 CosyVoice 分段设置独立、可配置且有正数默认值的执行时限。工作进程的启动就绪与分段推理 MUST 是独立阶段；同一工作进程已就绪后，后续分段 MUST 不再等待新的就绪信号。超时后系统 SHALL 终止仍在运行的 CosyVoice 工作单元、将任务标记为失败并释放串行生成队列。

#### Scenario: 分段在时限内完成
- **WHEN** CosyVoice 分段在配置时限内返回可用音频
- **THEN** 系统继续处理下一分段或完成整条生成

#### Scenario: 分段超过时限
- **WHEN** CosyVoice 分段超过其执行时限
- **THEN** 系统停止该分段的工作单元、将生成标记为可重试失败，并允许后续排队生成开始

#### Scenario: 同一工作进程连续生成多个语义分段
- **WHEN** 工作进程已完成首次启动，第一分段返回音频后系统提交第二分段
- **THEN** 系统直接向已就绪的工作进程提交第二分段，不得等待另一个启动就绪信号
- **THEN** 两个分段均使用各自的推理时限并按原始顺序返回音频

#### Scenario: 新工作进程启动与分段推理分开计时
- **WHEN** CosyVoice 模型尚未在工作进程中就绪
- **THEN** 系统先在模型启动时限内等待就绪，再从实际提交分段时开始计算该分段的推理时限

### Requirement: CosyVoice 阶段化诊断
系统 SHALL 在 CosyVoice 生成期间记录当前阶段及各阶段耗时；阶段 MUST 包含前处理、LLM 解码、Flow 与声码器。诊断信息 MUST 不包含用户输入文本、参考音频路径或底层异常堆栈。

#### Scenario: 正常完成的阶段记录
- **WHEN** CosyVoice 生成成功完成
- **THEN** 生成历史和状态 API 返回最终阶段及可用的阶段耗时记录

#### Scenario: 在某阶段超时或失败
- **WHEN** CosyVoice 在 LLM 解码、Flow 或声码器阶段失败或超时
- **THEN** 生成历史和状态 API 返回对应阶段与安全、可重试的错误信息

#### Scenario: 非 CosyVoice 生成的兼容性
- **WHEN** Qwen 或其他非 CosyVoice 引擎生成或查询历史
- **THEN** 既有生成请求和响应保持兼容，CosyVoice 专属诊断字段为空或缺省
