# 数据集组织说明

## 目录建议

```text
data/
  dev/       # 开发集，用于调试提示词、规则和 Agent 流程
  test/      # 固定测试集，只用于最终评测，不参与自进化
  labels_dev.csv
  labels_test.csv
```

## 标注字段

每条样本建议包含：

- `sample_id`：样本编号。
- `file_path`：图片或视频路径。
- `split`：dev 或 test。
- `risk_type`：风险类型；正常样本填 `normal`。
- `severity`：low / medium / high / uncertain / normal。
- `evidence`：人工标注证据。
- `location`：风险位置。
- `notes`：备注。

## 风险类型

第一版建议使用：

- `normal`
- `ppe_missing`
- `reagent_management`
- `flammable_near_heat`
- `food_or_drink`
- `spill_or_obstruction`

## 注意

- 同一视频的相邻帧不要拆到 dev 和 test 两边。
- 测试集不要用于提示词修改、规则修改或案例记忆。
- 不要制造危险场景采集数据。
