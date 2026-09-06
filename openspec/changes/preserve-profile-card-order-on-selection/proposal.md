## Why

声音档案卡片会在用户点击后因当前生成引擎的变化被重新排序，导致用户刚刚选择的卡片和相邻卡片发生位置跳动。这会打断用户在同一组卡片中继续选择或操作的预期。

## What Changes

- 保持声音档案列表采用其初始数据顺序渲染，不再因点击卡片、选中状态或关联的生成引擎变化而重新排列卡片。
- 保留当前引擎不支持的卡片禁用态、提示信息和选中态，不以排序来表达兼容性。
- 为卡片点击与引擎变化的顺序稳定性补充自动化测试用例。

## Capabilities

### New Capabilities

- `profile-card-order-stability`: 定义声音档案卡片在交互过程中保持原有列表顺序的行为。

### Modified Capabilities

<!-- 无 -->

## Impact

- 受影响前端：`app/src/components/VoiceProfiles/ProfileList.tsx`。
- 将新增或扩展前端组件测试；不涉及 API、数据结构或外部依赖变更。
