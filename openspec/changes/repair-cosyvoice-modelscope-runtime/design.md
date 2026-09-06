## Context

详见 proposal.md 的 Why。CosyVoice 源码在运行时从包根导入 ModelScope 的下载入口；该入口由 ModelScope 的惰性导入机制解析。当前冻结构建仅复制了 `modelscope` 的发行元数据，没有收集其 Python 包内容。开发环境可解析该导入，但桌面端单文件服务器解包到临时目录后会在这一点失败。

已有 `backend/build_binary.py` 集中声明 PyInstaller 收集规则，且 `backend/tests/test_rocm_build.py`、`backend/tests/test_audioop_python313.py` 已采用“模拟 PyInstaller 并断言参数”的轻量级构建回归模式。CosyVoice 注册测试明确避免下载或加载 7 GB 权重。

## Goals / Non-Goals

**Goals:**

- 将 ModelScope 的完整可运行包内容与保留的发行元数据一起纳入桌面端服务器构建。
- 在不执行真实冻结构建、不下载模型权重的单元测试中锁定收集规则。
- 保留既有模型加载超时、CPU 回退和安全失败终态，不改变其用户可见语义。

**Non-Goals:**

- 不改动 CosyVoice 模型仓库、缓存目录、权重内容或用户声音档案。
- 不更新 `modelscope==1.20.0`、不重写 CosyVoice 上游源码，也不将运行时网络请求替换为另一模型服务。
- 不将本次修复扩展到 Qwen、CustomVoice 或已确认的“服务关闭中断生成”历史记录。

## Decisions

### 收集整个 ModelScope 运行时包

在 PyInstaller 参数中保留 `--copy-metadata modelscope`，并增加 `--collect-all modelscope`。ModelScope 根包通过懒加载分发子模块；完整收集能够覆盖当前 `snapshot_download` 及其后续间接导入，同时收集包数据和二进制扩展，避免用少量手写 `--hidden-import` 掩盖这一次错误后在下一条延迟导入处再次失败。

备选方案是在构建脚本中仅声明 `modelscope.hub.snapshot_download` 等具体子模块。该方案体积较小，但依赖上游的动态导入图，维护成本高且不能可靠覆盖 CosyVoice 构造过程。另一备选方案是修改 CosyVoice 上游源码以彻底绕开 ModelScope；这会扩大厂商源码差异并提高升级风险，故不采用。

### 以构建参数测试防回归，以冻结端到端检查验证发行物

新增一个快速测试，模拟 PyInstaller 调用并断言 ModelScope 同时具有完整收集和元数据收集规则；该测试不创建二进制、不访问网络、不读取模型权重。保留并更新现有冻结二进制端到端检查说明：在具备已缓存 CosyVoice 模型的本机上，执行顺序模型加载和短文本生成，确认桌面端真实运行时可用。

备选方案是把 7 GB 模型加载纳入单元测试。该方案会使常规测试不可重复且过慢，故仅作为本地发行验证步骤。

### 不新增另一层异常转换

现有 `load_engine_model` 已将底层异常转换为安全的 `ModelLoadError`，生成服务会将任务标记为失败并释放队列。此次根因在构建依赖缺失，因此修复打包内容而不是修改用户错误文案；这既避免泄露临时解包路径，也不掩盖其他真正的运行时错误。

## Risks / Trade-offs

- [发行包体积增加] → ModelScope 是 CosyVoice 的必要运行时；在本地构建后记录产物大小，并只收集该单个包而非新增整套训练依赖。
- [仍存在未被快速测试覆盖的冻结环境差异] → 使用真实桌面端冻结二进制和已缓存模型的短文本检查作为发布前验收。
- [完整模型缓存被误判为可用但权重损坏] → 保持既有完整性检查与安全失败终态；不以本修复掩盖缓存损坏。

## Migration Plan

1. 更新构建规则和快速回归测试，运行目标测试与现有 CosyVoice、模型运行时测试。
2. 在本机构建桌面端服务器，使用已缓存 CosyVoice 模型完成一条短文本生成检查。
3. 若冻结二进制仍无法导入运行时，回滚构建规则改动并保留测试，依据完整错误日志补充最小缺失收集项；不删除用户模型缓存。
