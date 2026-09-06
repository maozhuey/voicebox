## Purpose

确保本地语音、转录和文本模型在可用硬件上优先获得 GPU 加速，并在设备或模型加载异常时以可恢复、可解释的终态反馈给用户，而不会让生成任务长期停留在进行中。

## ADDED Requirements

### Requirement: GPU 优先的模型执行
系统 SHALL 在目标模型与当前平台兼容时优先选择可用 GPU 执行模型加载和推理；Apple Silicon SHALL 优先使用 MLX 或 MPS，其他支持的平台 SHALL 优先使用其可用的 CUDA、ROCm、XPU 或 DirectML 后端。

#### Scenario: Apple Silicon 上存在兼容 GPU
- **WHEN** 用户在具有可用 Apple GPU 的设备上加载任一支持 GPU 的本地模型
- **THEN** 系统 SHALL 使用 MLX 或 MPS，而不是预先将该模型固定为 CPU

#### Scenario: GPU 不可用
- **WHEN** 当前平台没有可用的受支持 GPU
- **THEN** 系统 SHALL 使用 CPU，并继续完成模型加载与生成流程

### Requirement: GPU 失败时的 CPU 回退
系统 MUST 在模型因 GPU 加载或首次推理失败而无法运行时释放失败的 GPU 实例，并自动以 CPU 重试一次；成功回退后，后续请求 SHALL 使用该可运行实例。

#### Scenario: GPU 模型加载失败
- **WHEN** 兼容模型在 GPU 上加载时返回错误
- **THEN** 系统 MUST 清理失败实例、以 CPU 重试一次，并在 CPU 重试成功后继续原请求

#### Scenario: GPU 与 CPU 均无法加载
- **WHEN** GPU 加载失败且 CPU 重试也失败
- **THEN** 系统 SHALL 将关联任务标记为失败，并返回不包含内部路径或敏感信息的可理解错误说明

### Requirement: 有界的模型加载
系统 SHALL 对每次模型加载应用可配置的有限超时；超时值 MUST 有安全默认值，并允许通过本地运行配置覆盖。

#### Scenario: 模型在限时内完成加载
- **WHEN** 模型在配置的加载时限内完成
- **THEN** 系统 SHALL 继续生成流程并报告正常进度

#### Scenario: 模型加载超时
- **WHEN** 模型加载超过配置的时限
- **THEN** 系统 MUST 将关联生成任务更新为 `failed`、释放串行生成队列，并向客户端提供“模型加载超时，请重试或检查模型与设备”的错误说明

### Requirement: 已缓存模型的本地优先加载
系统 MUST 在确认模型所需文件完整地存在于本地缓存后，以不依赖远端元数据请求的方式加载该模型；缓存不完整时系统 SHALL 使用现有下载流程并保留下载失败状态。

#### Scenario: 本地缓存完整且远端不可达
- **WHEN** 模型文件完整保存在本地且远端模型服务不可达
- **THEN** 系统 SHALL 从本地缓存加载模型，不得因远端元数据检查无限等待

#### Scenario: 本地缓存不完整
- **WHEN** 本地缓存缺少模型加载所需文件
- **THEN** 系统 SHALL 启动或恢复可观测的下载流程；下载或加载最终失败时 SHALL 为关联任务产生明确失败状态

### Requirement: 生成任务的终态可见性
系统 MUST 确保每个已接受的生成请求最终进入 `completed` 或 `failed`；客户端 SHALL 能通过既有历史与状态流看到失败原因，并可以在问题排除后重试失败任务。

#### Scenario: 模型加载发生不可恢复异常
- **WHEN** 模型加载抛出不可恢复异常
- **THEN** 系统 MUST 将生成记录从 `loading_model` 更新为 `failed`，并让状态流在有限时间内结束

#### Scenario: 失败任务之后有排队任务
- **WHEN** 一个生成任务因加载失败或超时结束，且队列中仍有任务
- **THEN** 系统 SHALL 继续处理后续任务，不得让队列永久阻塞
