"""评测模块。

支持：
- 多标签样本（一个样本可同时包含多种风险）
- per-class 与 macro / micro 的 Precision / Recall / F1
- 混淆对统计（把 A 误判成 B 的次数）
- “证据不足时是否正确输出不确定”的弃权率统计
- 多组结果横向对比（基础版 / 多 Agent 版 / 自进化版 / 消融实验）

用法：
    python src/evaluate.py --labels data/labels_test.csv --reports reports/run_base reports/run_multi \
        --names 基础版 多Agent版 --out-md reports/eval.md
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

NORMAL = "normal"
#: 计入“系统给出了风险结论”的状态
POSITIVE_STATUSES = {"confirmed", "simulated", "analyzed"}
#: 计入“系统主动弃权”的状态
ABSTAIN_STATUSES = {"uncertain"}


# ------------------------------------------------------------------ 数据读取


def load_labels(labels_path: Path) -> Dict[str, Set[str]]:
    """读取标注 CSV。同一 sample_id 出现多行即视为多标签样本。"""
    labels: Dict[str, Set[str]] = defaultdict(set)
    with labels_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = (row.get("sample_id") or "").strip()
            risk_type = (row.get("risk_type") or NORMAL).strip() or NORMAL
            if sample_id:
                labels[sample_id].add(risk_type)
    return dict(labels)


def load_predictions(reports_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """从报告目录读取预测。返回 sample_id -> {positive, abstain}。"""
    predictions: Dict[str, Dict[str, Set[str]]] = {}
    for path in sorted(reports_dir.glob("*_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sample_id = data.get("sample_id") or path.stem.replace("_report", "")
        positive, abstain = set(), set()
        for finding in data.get("findings", []):
            status = finding.get("status", "")
            risk_type = finding.get("risk_type")
            if not risk_type:
                continue
            if status in POSITIVE_STATUSES:
                positive.add(risk_type)
            elif status in ABSTAIN_STATUSES:
                abstain.add(risk_type)
        predictions[sample_id] = {"positive": positive, "abstain": abstain}
    return predictions


def load_costs(reports_dir: Path) -> Dict[str, Dict[str, float]]:
    """读取每个样本的处理耗时与调用成本，用于成本对比。"""
    costs: Dict[str, Dict[str, float]] = {}
    for path in sorted(reports_dir.glob("*_report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sample_id = data.get("sample_id") or path.stem.replace("_report", "")
        costs[sample_id] = {
            "duration_ms": float(data.get("duration_ms") or 0),
            "model_calls": float(data.get("model_calls") or 0),
            "tool_calls": float(data.get("tool_calls") or 0),
        }
    return costs


# ------------------------------------------------------------------ 指标计算


def evaluate_run(
    labels: Dict[str, Set[str]],
    predictions: Dict[str, Dict[str, Set[str]]],
    costs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion: Counter = Counter()
    abstained_samples = 0
    missed_samples: List[str] = []
    false_positive_samples: List[str] = []
    evaluated = 0

    for sample_id, true_set in labels.items():
        true_risks = {t for t in true_set if t != NORMAL}
        pred_all = predictions.get(sample_id)
        pred_set = set(pred_all["positive"]) if pred_all else set()
        abstain_set = set(pred_all["abstain"]) if pred_all else set()

        evaluated += 1
        if not pred_set and abstain_set:
            abstained_samples += 1

        for risk_type in true_risks:
            if risk_type in pred_set:
                tp[risk_type] += 1
            else:
                fn[risk_type] += 1
                missed_samples.append(sample_id)
                for wrong in pred_set - true_risks:
                    confusion[f"{risk_type} -> {wrong}"] += 1

        for risk_type in pred_set - true_risks:
            fp[risk_type] += 1
            false_positive_samples.append(sample_id)

    classes = sorted(set(tp) | set(fp) | set(fn))
    per_class = []
    for risk_type in classes:
        p = _div(tp[risk_type], tp[risk_type] + fp[risk_type])
        r = _div(tp[risk_type], tp[risk_type] + fn[risk_type])
        per_class.append({
            "risk_type": risk_type,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(_f1(p, r), 4),
            "support": tp[risk_type] + fn[risk_type],
            "tp": tp[risk_type],
            "fp": fp[risk_type],
            "fn": fn[risk_type],
        })

    macro_p = _mean([c["precision"] for c in per_class])
    macro_r = _mean([c["recall"] for c in per_class])
    macro_f1 = _mean([c["f1"] for c in per_class])

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = _div(total_tp, total_tp + total_fp)
    micro_r = _div(total_tp, total_tp + total_fn)
    total_samples = evaluated
    fp_samples = len(set(false_positive_samples))
    fn_samples = len(set(missed_samples))

    cost_block = _summarize_costs(costs)

    return {
        "sample_count": total_samples,
        "predicted_samples": len(predictions),
        "per_class": per_class,
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(_f1(micro_p, micro_r), 4)},
        "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
        "false_positive_samples": fp_samples,
        "missed_samples": fn_samples,
        "false_positive_rate": round(_div(fp_samples, total_samples), 4),
        "miss_rate": round(_div(fn_samples, total_samples), 4),
        "abstain_samples": abstained_samples,
        "abstain_rate": round(_div(abstained_samples, total_samples), 4),
        "confusions": [{"pair": k, "count": v} for k, v in confusion.most_common()],
        "cost": cost_block,
    }


def _summarize_costs(costs: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    if not costs:
        return {}
    n = len(costs)
    if not n:
        return {}
    return {
        "avg_duration_ms": round(sum(c["duration_ms"] for c in costs.values()) / n, 1),
        "avg_model_calls": round(sum(c["model_calls"] for c in costs.values()) / n, 2),
        "avg_tool_calls": round(sum(c["tool_calls"] for c in costs.values()) / n, 2),
        "total_model_calls": int(sum(c["model_calls"] for c in costs.values())),
        "total_tool_calls": int(sum(c["tool_calls"] for c in costs.values())),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------------------ 输出


def render_markdown(runs: List[Tuple[str, Path, Dict[str, Any]]]) -> str:
    lines = ["# 评测结果", "", "## 总体对比", ""]
    lines.append("| 实验组 | 样本数 | Macro P | Macro R | Macro F1 | Micro F1 | 误报样本率 | 漏报样本率 | 弃权率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, _, m in runs:
        lines.append(
            f"| {name} | {m['sample_count']} | {m['macro']['precision']:.3f} | {m['macro']['recall']:.3f} "
            f"| {m['macro']['f1']:.3f} | {m['micro']['f1']:.3f} | {m['false_positive_rate']:.3f} "
            f"| {m['miss_rate']:.3f} | {m['abstain_rate']:.3f} |"
        )

    if any(m.get("cost") for _, _, m in runs):
        lines.extend(["", "## 处理成本对比", ""])
        lines.append("| 实验组 | 平均耗时 (ms) | 平均模型调用 | 平均工具调用 | 模型调用总数 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, _, m in runs:
            c = m.get("cost") or {}
            lines.append(
                f"| {name} | {c.get('avg_duration_ms', '—')} | {c.get('avg_model_calls', '—')} "
                f"| {c.get('avg_tool_calls', '—')} | {c.get('total_model_calls', '—')} |"
            )

    for name, _, m in runs:
        lines.extend(["", f"## {name} — 分类别指标", ""])
        lines.append("| 风险类型 | Precision | Recall | F1 | 支持数 | TP | FP | FN |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in m["per_class"]:
            lines.append(
                f"| {c['risk_type']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} "
                f"| {c['support']} | {c['tp']} | {c['fp']} | {c['fn']} |"
            )
        if m["confusions"]:
            lines.extend(["", "主要混淆对：", ""])
            for row in m["confusions"][:8]:
                lines.append(f"- `{row['pair']}` × {row['count']}")

    lines.append("")
    return "\n".join(lines)


def render_console(runs: List[Tuple[str, Path, Dict[str, Any]]]) -> str:
    out = []
    for name, _, m in runs:
        out.append(f"=== {name} ===")
        out.append(f"samples={m['sample_count']} predicted={m['predicted_samples']}")
        out.append(
            f"macro  P={m['macro']['precision']:.3f} R={m['macro']['recall']:.3f} F1={m['macro']['f1']:.3f}"
            f" | micro F1={m['micro']['f1']:.3f}"
        )
        out.append(
            f"误报样本率={m['false_positive_rate']:.3f} 漏报样本率={m['miss_rate']:.3f} 弃权率={m['abstain_rate']:.3f}"
        )
        c = m.get("cost") or {}
        if c:
            out.append(
                f"成本：平均耗时 {c.get('avg_duration_ms')} ms｜平均模型调用 {c.get('avg_model_calls')} 次"
                f"｜平均工具调用 {c.get('avg_tool_calls')} 次"
            )
        for cc in m["per_class"]:
            out.append(
                f"  {cc['risk_type']:<24} P={cc['precision']:.3f} R={cc['recall']:.3f} F1={cc['f1']:.3f} "
                f"(TP={cc['tp']} FP={cc['fp']} FN={cc['fn']})"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测风险识别结果（支持多标签与多组对比）")
    parser.add_argument("--labels", required=True, help="标注 CSV，例如 data/labels_test.csv")
    parser.add_argument("--reports", required=True, nargs="+", help="一个或多个报告目录")
    parser.add_argument("--names", default=None, help="与 --reports 对应的实验组名称，逗号分隔")
    parser.add_argument("--out-json", default=None, help="把完整指标写入 JSON")
    parser.add_argument("--out-md", default=None, help="把对比表写入 Markdown")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    names = args.names.split(",") if args.names else None

    runs: List[Tuple[str, Path, Dict[str, Any]]] = []
    for idx, reports in enumerate(args.reports):
        reports_dir = Path(reports)
        name = names[idx] if names and idx < len(names) else reports_dir.name
        predictions = load_predictions(reports_dir)
        runs.append((name, reports_dir, evaluate_run(labels, predictions, load_costs(reports_dir))))

    print(render_console(runs))

    if args.out_json:
        payload = {name: metrics for name, _, metrics in runs}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"指标已写入 {args.out_json}")

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_markdown(runs), encoding="utf-8")
        print(f"对比表已写入 {args.out_md}")


if __name__ == "__main__":
    main()
