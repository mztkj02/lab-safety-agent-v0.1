> 本表为 2026-10-25 标注与评测修正后重算的结果（修正了 3 处错标、修了 sample_id 归一化）。模型输出未变，仅评测口径变。

# 评测结果

## 总体对比

| 实验组 | 有效样本 | Macro P | Macro R | Macro F1 | Micro F1 | 误报样本率 | 漏报样本率 | 弃权率 | 分析失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基础系统 | 43 | 0.192 | 0.813 | 0.307 | 0.298 | 0.791 | 0.093 | 0.000 | 0 |
| 多 Agent | 43 | 0.470 | 0.550 | 0.500 | 0.545 | 0.140 | 0.209 | 0.046 | 0 |
| 两轮自进化 | 43 | 0.365 | 0.557 | 0.397 | 0.400 | 0.419 | 0.209 | 0.046 | 0 |

## 处理成本对比

| 实验组 | 平均耗时 (ms) | 平均模型调用 | 平均工具调用 | 模型调用总数 |
| --- | --- | --- | --- | --- |
| 基础系统 | 4865.0 | 1.0 | 0.0 | 43 |
| 多 Agent | 6971.7 | 1.98 | 1.0 | 85 |
| 两轮自进化 | 11652.3 | 2.63 | 1.0 | 113 |

## 基础系统 — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.188 | 1.000 | 0.316 | 3 | 3 | 13 | 0 |
| food_or_drink | 0.250 | 1.000 | 0.400 | 3 | 3 | 9 | 0 |
| ppe_missing | 0.154 | 0.400 | 0.222 | 5 | 2 | 11 | 3 |
| reagent_management | 0.100 | 0.667 | 0.174 | 3 | 2 | 18 | 1 |
| spill_or_obstruction | 0.267 | 1.000 | 0.421 | 4 | 4 | 11 | 0 |

主要混淆对：

- `ppe_missing -> reagent_management` × 2
- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> spill_or_obstruction` × 1
- `ppe_missing -> food_or_drink` × 1
- `reagent_management -> food_or_drink` × 1

## 多 Agent — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.600 | 1.000 | 0.750 | 3 | 3 | 2 | 0 |
| food_or_drink | 0.667 | 0.667 | 0.667 | 3 | 2 | 1 | 1 |
| ppe_missing | 0.000 | 0.000 | 0.000 | 5 | 0 | 0 | 5 |
| reagent_management | 0.333 | 0.333 | 0.333 | 3 | 1 | 2 | 2 |
| spill_or_obstruction | 0.750 | 0.750 | 0.750 | 4 | 3 | 1 | 1 |

主要混淆对：

- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> reagent_management` × 1
- `reagent_management -> food_or_drink` × 1

## 两轮自进化 — 分类别指标

| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flammable_near_heat | 0.333 | 1.000 | 0.500 | 3 | 3 | 6 | 0 |
| food_or_drink | 0.600 | 1.000 | 0.750 | 3 | 3 | 2 | 0 |
| ppe_missing | 0.500 | 0.200 | 0.286 | 5 | 1 | 1 | 4 |
| reagent_management | 0.143 | 0.333 | 0.200 | 3 | 1 | 6 | 2 |
| spill_or_obstruction | 0.250 | 0.250 | 0.250 | 4 | 1 | 3 | 3 |

主要混淆对：

- `ppe_missing -> flammable_near_heat` × 1
- `ppe_missing -> reagent_management` × 1
- `ppe_missing -> food_or_drink` × 1
- `reagent_management -> food_or_drink` × 1
