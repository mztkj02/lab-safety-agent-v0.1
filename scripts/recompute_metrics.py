"""用已有的报告重新计算指标，不重新调用模型。

用途：标注或评测逻辑改动后（例如修正了错标、修了 sample_id 归一化），
不需要再花几十次 API 调用重跑，直接拿已有报告重算即可。

    # 主实验三组
    python scripts/recompute_metrics.py \
        --labels data/web_labels_test.csv \
        --group 基础系统=reports/experiments/base \
        --group 多Agent=reports/experiments/multi \
        --group 两轮自进化=reports/experiments/evolved \
        --out reports/experiments

    # 消融
    python scripts/recompute_metrics.py \
        --labels data/web_labels_test.csv \
        --group 完整系统=reports/experiments/multi \
        --group 关复核=reports/experiments/ablation/no_review \
        --group 关工具=reports/experiments/ablation/no_tools \
        --out reports/experiments/ablation

会把 comparison.md / comparison.json 写到 --out 目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import (  # noqa: E402
    evaluate_run,
    load_costs,
    load_labels,
    load_predictions,
    render_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="用已有报告重算指标")
    parser.add_argument("--labels", required=True, help="测试集标注 CSV")
    parser.add_argument("--group", action="append", default=[],
                        help="名称=报告目录，可重复")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--note", default="", help="写进表头的说明")
    args = parser.parse_args()

    if not args.group:
        print("至少给一个 --group 名称=目录")
        return

    labels_path = PROJECT_ROOT / args.labels
    labels = load_labels(labels_path)
    print(f"标注样本：{len(labels)} 条（{args.labels}）")

    runs = []
    for item in args.group:
        if "=" not in item:
            print(f"  ! 格式应为 名称=目录，跳过：{item}")
            continue
        name, raw_dir = item.split("=", 1)
        directory = PROJECT_ROOT / raw_dir
        if not directory.exists():
            print(f"  ! 目录不存在，跳过：{raw_dir}")
            continue
        preds = load_predictions(directory)
        if not preds:
            print(f"  ! 没有报告文件，跳过：{raw_dir}")
            continue

        missing = [k for k in labels if k not in preds]
        if missing:
            print(f"  ⚠ {name}：{len(missing)} 个标注样本没有对应报告，"
                  f"这部分会被当成「未预测」"

                  f"（例：{missing[:3]}）")

        metrics = evaluate_run(labels, preds, load_costs(directory))
        runs.append((name, directory, metrics))
        print(f"  {name:16} macro F1={metrics['macro']['f1']:.3f}"
              f"  误报={metrics['false_positive_rate']:.1%}"
              f"  漏报={metrics['miss_rate']:.1%}"
              f"  报告={len(preds)}")

    if not runs:
        print("没有可评测的组")
        return

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    header = ""
    if args.note:
        header = f"> {args.note}\n\n"

    md_path = out_dir / "comparison.md"
    md_path.write_text(header + render_markdown(runs), encoding="utf-8")

    json_path = out_dir / "comparison.json"
    json_path.write_text(
        json.dumps({name: m for name, _, m in runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写入 {md_path.relative_to(PROJECT_ROOT)} 与 comparison.json")


if __name__ == "__main__":
    main()
