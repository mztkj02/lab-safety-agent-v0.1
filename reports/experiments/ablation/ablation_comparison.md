# 评测结果

## 总体对比

| 实验组 | 有效样本 | Macro P | Macro R | Macro F1 | Micro F1 | 误报样本率 | 漏报样本率 | 弃权率 | 分析失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 完整系统（参照，策略 v0.2） | 43 | 0.533 | 0.550 | 0.535 | 0.600 | 0.093 | 0.186 | 0.046 | 0 |
| 关闭证据复核 Agent | 43 | 0.422 | 0.617 | 0.487 | 0.556 | 0.209 | 0.163 | 0.000 | 0 |
| 关闭 Agent 自主工具调用 | 43 | 0.513 | 0.567 | 0.529 | 0.562 | 0.140 | 0.186 | 0.000 | 0 |

## 处理成本对比

| 实验组 | 平均耗时 (ms) | 平均模型调用 | 平均工具调用 | 模型调用总数 |
| --- | --- | --- | --- | --- |
| 完整系统（参照，策略 v0.2） | 6971.7 | 1.98 | 1.0 | 85 |
| 关闭证据复核 Agent | 6198.0 | 1.53 | 1.0 | 66 |
| 关闭 Agent 自主工具调用 | 7595.7 | 1.88 | 1.0 | 81 |

## 完整系统（参照，策略 v0.2） — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.750 | 1.000 | 0.857 | 3 | 3 | 1 | 0 |
| food_or_drink | 0.667 | 0.667 | 0.667 | 3 | 2 | 1 | 1 |
| ppe_missing | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| reagent_management | 0.500 | 0.333 | 0.400 | 3 | 1 | 1 | 2 |
| spill_or_obstruction | 0.750 | 0.750 | 0.750 | 4 | 3 | 1 | 1 |

主要混淆对：

- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> reagent_management` × 1
- `reagent_management -> food_or_drink` × 1

## 关闭证据复核 Agent — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.600 | 1.000 | 0.750 | 3 | 3 | 2 | 0 |
| food_or_drink | 0.429 | 1.000 | 0.600 | 3 | 3 | 4 | 0 |
| ppe_missing | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| reagent_management | 0.333 | 0.333 | 0.333 | 3 | 1 | 2 | 2 |
| spill_or_obstruction | 0.750 | 0.750 | 0.750 | 4 | 3 | 1 | 1 |

主要混淆对：

- `ppe_missing -> food_or_drink` × 2
- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> reagent_management` × 1
- `reagent_management -> food_or_drink` × 1
- `spill_or_obstruction -> flammable_near_heat` × 1

## 关闭 Agent 自主工具调用 — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 1.000 | 1.000 | 1.000 | 3 | 3 | 0 | 0 |
| food_or_drink | 0.500 | 0.667 | 0.571 | 3 | 2 | 2 | 1 |
| ppe_missing | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| reagent_management | 0.400 | 0.667 | 0.500 | 3 | 2 | 3 | 1 |
| spill_or_obstruction | 0.667 | 0.500 | 0.571 | 4 | 2 | 1 | 2 |

主要混淆对：

- `ppe_missing -> reagent_management` × 1
- `ppe_missing -> food_or_drink` × 1
- `reagent_management -> food_or_drink` × 1
