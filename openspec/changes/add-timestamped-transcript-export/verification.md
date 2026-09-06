## 验证记录

验证日期：2026-09-05

### 自动测试

- 后端新增与相关回归：37 passed。
- 前端时间戳构建与保存流程：6 passed，覆盖成功、取消、缺少分段及写入失败。
- 前端 App 与 Web TypeScript：通过。
- 后端扩展回归（排除两个已确认的历史坏测试文件）：252 passed，5 skipped。
- `git diff --check` 与 OpenSpec strict validate：通过。

### 真实模型和桌面端验收

- Apple Silicon 自动选择 `MLXSTTBackend`，Whisper Large 模型已缓存并用于真实识别。
- 4.019 秒 WAV：识别为 1 个分段；模型原始 29.980 秒填充末端被限制为 4.020 秒。
- 51.690 秒 WAV：识别为 24 个升序分段，末段结束于 51.500 秒。
- 完成短音频与长音频的导入、Large 转录、Qwen 精修、时间戳 TXT 构建和重新转录；精修前后分段及格式化文稿完全一致，重新转录后清除旧精修，API 与 SQLite 分段一致。
- 音频接口的中段 Range 请求返回 `206 Partial Content` 与正确的 `Content-Range`，可供播放器定位。
- 捕获页已截图检查：菜单项位于普通 TXT 与 Markdown 之间；旧捕获点击后显示重新转录提示且未打开保存框；浏览器环境触发 Tauri 写入失败时显示错误提示。
- Debug macOS `.app` 本地构建成功。电脑锁屏导致无法继续操作原生保存框；保存成功、取消和失败分支已由依赖注入的自动测试覆盖。

### 仓库既有检查问题

- 完整 pytest 在收集 `test_profile_duplicate_names.py` 时因其顶层相对导入失败；排除该文件后，`test_progress.py::test_hf_progress_tracker` 因既有 tqdm 补丁未捕获回调失败。
- 全仓 Ruff 与 Biome 存在大量本变更前已有的问题；本次新增的 Python 与 TypeScript 文件分别通过定向 Ruff/Biome，TypeScript 全量类型检查通过。


### 2026-09-06 本地桌面安装验收

- 重新构建 Python server/MCP sidecar 及 Tauri release `.app`，安装至 `/Applications/Voicebox.app` 并启动。
- 前端导出测试：6 passed；后端时间戳与持久化测试：17 passed；App/Web TypeScript 检查通过。
- 新安装包通过 macOS ad-hoc 签名完整性检查；签名前包内 backend SHA-256 与新构建 sidecar 一致。
- 独立数据库副本验证：14 条旧捕获可读取且接口包含时间戳字段；真实 Whisper Large 对 3.960 秒音频返回 `[00:00:00.000 --> 00:00:03.960] Thank you.`。测试仅修改独立副本。
- 安装后本地后端 `http://127.0.0.1:17493/health` 健康，捕获接口包含 `transcript_segments`。
- 原生桌面捕获页导出菜单实际显示：音频 (WAV)、文字稿 (TXT)、带时间戳文稿 (TXT)、Markdown (MD)。
- 原有捕获文本、精修文本、音频路径逐项与备份一致；14 条捕获、71 条生成记录保留，SQLite 完整性检查通过。
- 原应用和数据库备份：`/Users/hanchanglin/Library/Application Support/sh.voicebox.app/backups/desktop-update-20260906-002050`。
- 本次仅安装本地 macOS 应用；未启动开发服务器、Docker 容器或线上部署。原生保存框实际写入未在本次重复验收，保存成功/取消/失败由前端自动测试覆盖。
