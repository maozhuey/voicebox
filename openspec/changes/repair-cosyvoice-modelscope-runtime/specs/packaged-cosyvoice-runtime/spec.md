## Purpose

确保桌面端已缓存完整 CosyVoice 模型时，冻结发行包具备启动其推理运行时的全部依赖，并在依赖无法满足时留下安全、可重试且可诊断的任务终态。

## ADDED Requirements

### Requirement: 已缓存 CosyVoice 的桌面端可加载性
系统 SHALL 在桌面端冻结发行包中提供已缓存 CosyVoice 模型启动所需的运行时依赖；模型缓存完整时，加载流程 MUST 不因延迟解析的运行时模块缺失而失败。

#### Scenario: 冻结运行时解析 CosyVoice 依赖
- **WHEN** 桌面端服务器在已缓存完整 CosyVoice 模型的环境中开始加载 CosyVoice
- **THEN** 系统 SHALL 成功解析 CosyVoice 启动期间访问的运行时模块，并继续既有模型构造流程

#### Scenario: 构建配置回归验证
- **WHEN** 开发者执行 CosyVoice 冻结包依赖回归测试
- **THEN** 测试 SHALL 验证构建配置覆盖延迟解析的运行时导入，且不得下载或加载实际 CosyVoice 权重

### Requirement: CosyVoice 启动失败的可恢复终态
系统 MUST 将 CosyVoice 启动期间无法恢复的运行时依赖错误转换为既有的安全失败终态；失败任务 SHALL 可在问题修复后重试，且后续排队任务不得被阻塞。

#### Scenario: 冻结运行时依赖仍不可用
- **WHEN** CosyVoice 在启动中遇到不可恢复的运行时依赖错误
- **THEN** 系统 SHALL 将关联生成记录标记为 `failed`，返回不包含内部路径的可理解错误，并释放生成队列

#### Scenario: 修复后重试失败任务
- **WHEN** 运行时依赖已恢复且用户重试先前失败的 CosyVoice 任务
- **THEN** 系统 SHALL 重新进入既有模型加载与生成流程，而不要求重新创建声音档案或重新下载完整模型
