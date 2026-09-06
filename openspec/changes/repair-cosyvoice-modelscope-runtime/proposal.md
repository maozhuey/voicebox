## Why

桌面端已下载完整的 CosyVoice 3 模型后，CosyVoice 在导入 `modelscope.hub.snapshot_download` 时仍会因冻结包缺少运行时可发现的模块而失败。界面仅显示通用的模型加载失败，导致用户无法使用 CosyVoice，且无法从模型下载状态判断问题所在。

## What Changes

- 让桌面端冻结构建显式收集 CosyVoice 所需的 ModelScope 运行时代码、数据和元数据，避免依赖 PyInstaller 无法静态分析的懒加载导入。
- 为 ModelScope 导入建立轻量级回归测试，验证打包配置覆盖 CosyVoice 会访问的 `snapshot_download` 路径，而不下载或加载 7 GB 模型权重。
- 在本地 Docker/开发验证策略中保留真实 CosyVoice 已缓存模型的加载检查，并确认失败时仍按既有机制产生安全、可重试的终态。

## Capabilities

### New Capabilities

- `packaged-cosyvoice-runtime`: 桌面端冻结发行包可加载 CosyVoice 所依赖的 ModelScope 运行时，并对缺失依赖提供可诊断、可恢复的任务终态。

### Modified Capabilities

- 无。

## Impact

- 受影响代码：`backend/build_binary.py`、CosyVoice 运行时保护逻辑及其测试。
- 受影响系统：macOS 桌面端 `voicebox-server` 冻结包；不改变生成 API、模型仓库或用户声音档案格式。
- 依赖影响：仍使用固定的 `modelscope==1.20.0`；构建产物会包含该包实际运行所需的可发现模块与元数据。
