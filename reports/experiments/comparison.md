# 评测结果

## 总体对比

| 实验组 | 有效样本 | Macro P | Macro R | Macro F1 | Micro F1 | 误报样本率 | 漏报样本率 | 弃权率 | 分析失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基础识别系统（单模型直出） | 43 | 0.388 | 0.783 | 0.479 | 0.491 | 0.372 | 0.093 | 0.000 | 0 |
| 多 Agent + 工具调用 | 43 | 0.533 | 0.550 | 0.535 | 0.600 | 0.093 | 0.186 | 0.046 | 0 |
| 两轮自进化后 | 43 | 0.387 | 0.517 | 0.438 | 0.485 | 0.186 | 0.209 | 0.023 | 0 |

## 处理成本对比

| 实验组 | 平均耗时 (ms) | 平均模型调用 | 平均工具调用 | 模型调用总数 |
| --- | --- | --- | --- | --- |
| 基础识别系统（单模型直出） | 4865.0 | 1.0 | 0.0 | 43 |
| 多 Agent + 工具调用 | 6971.7 | 1.98 | 1.0 | 85 |
| 两轮自进化后 | 11652.3 | 2.63 | 1.0 | 113 |

## 基础识别系统（单模型直出） — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.429 | 1.000 | 0.600 | 3 | 3 | 4 | 0 |
| food_or_drink | 0.429 | 1.000 | 0.600 | 3 | 3 | 4 | 0 |
| ppe_missing | 0.500 | 0.250 | 0.333 | 4 | 1 | 1 | 3 |
| reagent_management | 0.250 | 0.667 | 0.364 | 3 | 2 | 6 | 1 |
| spill_or_obstruction | 0.333 | 1.000 | 0.500 | 4 | 4 | 8 | 0 |

主要混淆对：

- `ppe_missing -> reagent_management` × 2
- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> spill_or_obstruction` × 1
- `ppe_missing -> food_or_drink` × 1
- `reagent_management -> food_or_drink` × 1

## 多 Agent + 工具调用 — 分类别指标

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

## 两轮自进化后 — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.750 | 1.000 | 0.857 | 3 | 3 | 1 | 0 |
| food_or_drink | 0.600 | 1.000 | 0.750 | 3 | 3 | 2 | 0 |
| ppe_missing | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| reagent_management | 0.333 | 0.333 | 0.333 | 3 | 1 | 2 | 2 |
| spill_or_obstruction | 0.250 | 0.250 | 0.250 | 4 | 1 | 3 | 3 |

主要混淆对：

- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> reagent_management` × 1
- `ppe_missing -> food_or_drink` × 1
- `reagent_management -> food_or_drink` × 1
