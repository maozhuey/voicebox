## 1. 冻结构建依赖修复

- [x] 1.1 在 `backend/build_binary.py` 中为 ModelScope 增加完整包收集规则，并保留其发行元数据收集；验证生成的 PyInstaller 参数同时包含 `--collect-all modelscope` 与 `--copy-metadata modelscope`。
- [x] 1.2 为 CosyVoice 的冻结构建依赖新增快速回归测试，使用模拟的 PyInstaller 调用断言 1.1 的参数；验证测试不访问网络、不下载模型权重且不启动实际构建。

## 2. 自动化验证

- [x] 2.1 运行新增构建依赖测试及既有 CosyVoice 注册测试：`python -m pytest backend/tests/test_cosyvoice_packaging.py backend/tests/test_cosyvoice_registry.py -q`，验证所有用例通过。
- [x] 2.2 运行既有模型运行时弹性测试：`python -m pytest backend/tests/test_model_runtime_resilience.py -q`，验证模型加载失败仍进入安全失败终态且队列回收行为不回归。
- [x] 2.3 运行与本次变更相关的完整后端测试策略：`python -m pytest backend/tests -q`，记录通过、跳过和失败项；结果为 277 通过、5 跳过，另有两项既有问题：`test_profile_duplicate_names.py` 的顶层导入收集错误，以及 `test_progress.py::test_hf_progress_tracker` 未捕获 tqdm 进度；均与本次构建规则变更无关。

## 3. 本地冻结发行物验收

- [x] 3.1 在确认至少保留构建临时文件和现有模型缓存所需磁盘空间后，构建本地桌面服务器：`cd backend && python build_binary.py`；验证产物存在且构建日志无 ModelScope 收集错误。验证结果：构建前可用空间 24 GB，`backend/dist/voicebox-server` 已由 569 MB 更新为 578 MB。
- [x] 3.2 使用新构建的本地服务器、现有完整 CosyVoice 缓存和独立的测试数据目录执行一条短文本 CosyVoice 生成；验证模型加载越过 ModelScope 导入、任务最终为 `completed` 且音频文件存在。验证结果：生成记录 `d05ea236-1fdf-44e8-9eda-85375b08b11f` 已完成，生成时长 3.6 秒，音频文件大小 169 KB。
- [x] 3.3 若 3.2 失败，保留构建日志与安全失败终态，回滚仅本次构建规则改动而不删除用户模型缓存；验证后续 Qwen 生成仍可成功。未触发：3.2 已通过，因此无需回滚或执行失败恢复分支。
