## 1. 后端阶段化精修与数据保护

- [x] 1.1 在 `backend/services/refinement.py` 为捕获精修显式执行 LLM 加载并以受控异常区分 `model_load` 与 `llm_generate`；补充伪后端加载失败、生成失败和成功生成测试，并验证原始异常文本不进入异常对外消息。
- [x] 1.2 在 `backend/services/captures.py` 将精修结果的写回限制为一次提交，并在提交或刷新失败时回滚、以 `persist` 阶段抛出受控异常；扩展 `backend/tests/test_refinement_language.py`，验证失败后原始转录、已有精修文本和模型元数据保持不变。

## 2. 安全诊断与 API 契约

- [x] 2.1 扩展 `backend/services/capture_diagnostics.py` 以记录精修失败的固定类型、阶段、来源、模型尺寸和脱敏异常摘要；扩展 `backend/tests/test_capture_failure_diagnostics.py`，验证诊断中不含转录正文、音频路径、文件名或堆栈。
- [x] 2.2 更新 `POST /captures/{capture_id}/refine`，对受控精修失败生成 `CAPTURE_REFINEMENT_DIAGNOSTIC:<id>` 的安全 500 响应，并保留 404 行为；以路由测试验证 `model_load`、`llm_generate`、`persist` 三个阶段都能记录可关联诊断编号。

## 3. 客户端恢复与用户提示

- [x] 3.1 在错误文案映射中识别精修诊断令牌，验证它输出包含诊断编号的中文精修失败说明，而不是 HTTP 400/500 原文。
- [x] 3.2 重构 `useCaptureRecordingSession` 的精修 mutation 输入以保留自动/手动发起上下文；自动精修失败时失效并广播已成功捕获、只调用一次原始转录交付，手动失败时不自动粘贴。提取可单测的失败恢复策略，并用 `bun test tests` 覆盖两种模式、原始转录为空与诊断错误文案。

## 4. 整体验证与本地交付

- [x] 4.1 运行精修与捕获相关后端回归测试、前端 `bun test tests`、`bun run typecheck` 和构建检查；验证所有新增用例以及现有时间戳捕获用例通过。
- [x] 4.2 在本地开发服务器和本地 Docker 环境执行“转录成功、精修失败”的验证，确认原始文本仍可查看/交付、诊断文件无敏感内容；按桌面构建流程替换本机应用并重启，确认安装版不再展示通用 500。
