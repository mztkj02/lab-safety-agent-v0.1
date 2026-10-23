"""消融实验：逐个关掉关键功能，看指标怎么变。

课程要求「关闭一项关键功能做对比」。这里比要求多做几个，因为只有逐个关掉，
才能回答「到底是哪个机制在起作用」而不是笼统地说「多 Agent 有用」。

配置：
    full        完整系统（多 Agent + 工具 + 规则 + 记忆 + 复核）
    no_review   关掉证据复核 Agent —— 检验「质疑者」到底抑制了多少误报
    no_rules    关掉安全规则检索 —— 检验规则依据对定级的影响
    no_memory   关掉案例记忆 —— 检验历史纠错能否复用
    no_tools    关掉 Agent 自主工具调用 —— 检验裁剪/检索是否真的被用上

用法：
    python scripts/run_ablation.py --labels data/web_labels_test.csv
    python scripts/run_ablation.py --only no_review          # 只跑一个
    python scripts/run_ablation.py --resume                   # 断点续跑
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for p in (SRC_DIR, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate import evaluate_run, load_costs, load_labels, load_predictions, render_markdown  # noqa: E402
from llm import describe_backend  # noqa: E402
from run_experiment import read_samples, run_group  # noqa: E402

#: 配置文件 -> (显示名, 传给 pipeline 的 ablation 开关)
ABLATIONS = {
    "full": ("完整系统", {}),
    "no_review": ("关闭证据复核 Agent", {"use_review_agent": False}),
    "no_rules": ("关闭安全规则检索", {"use_rules": False}),
    "no_memory": ("关闭案例记忆", {"use_memory": False}),
    "no_tools": ("关闭 Agent 自主工具调用", {"use_agent_tools": False}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验：关闭关键功能做对比")
    parser.add_argument("--labels", default="data/web_labels_test.csv", help="测试集标注 CSV")
    parser.add_argument("--out", default="reports/experiments/ablation", help="输出目录")
    parser.add_argument("--only", action="append", help="只跑指定的配置（可重复）")
    parser.add_argument("--sleep", type=float, default=0.5, help="样本间隔秒数")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--mock", action="store_true", help="强制模拟模式")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本")
    parser.add_argument("--strategy", default="",
                        help="策略文件名，如 configs/strategy_v0.2.json（默认用当前策略）")
    parser.add_argument("--reference", default="reports/experiments/multi",
                        help="参照组报告目录（完整系统），会一并计入对比表；留空则不加入")
    args = parser.parse_args()

    backend = describe_backend()
    use_mock = args.mock or backend["mode"] == "mock"
    mode_label = "模拟模式" if use_mock else f"真实模型 {backend['model']}"
    print(f"运行模式：{mode_label}")

    samples = read_samples(PROJECT_ROOT / args.labels)
    if not samples:
        print(f"错误：{args.labels} 没有样本")
        sys.exit(1)
    if args.limit > 0:
        samples = samples[: args.limit]
        print(f"按 --limit 只取前 {len(samples)} 个样本")

    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    keys = args.only or list(ABLATIONS)
    strategy_path = (PROJECT_ROOT / args.strategy) if args.strategy else None
    if strategy_path:
        print(f"策略版本：{strategy_path.name}")
    runs = []
    for key in keys:
        if key not in ABLATIONS:
            print(f"未知配置 {key}，可选：{', '.join(ABLATIONS)}")
            continue
        label, ablation = ABLATIONS[key]
        run_dir = out_root / key
        print(f"\n>>> 消融配置：{label}")
        run_group(label, samples, run_dir, use_mock, "multi_agent",
                  sleep=args.sleep, resume=args.resume, ablation=ablation,
                  strategy_path=strategy_path)
        runs.append((label, run_dir, key))

    # 参照组：完整系统的既有结果，直接纳入对比，避免重跑一遍浪费调用成本。
    # 只有在策略版本一致时才可比 —— multi 组是在 v0.2 下跑的。
    ref_dir = PROJECT_ROOT / args.reference if args.reference else None
    if ref_dir and ref_dir.exists() and any(ref_dir.glob("*_report.json")):
        same_strategy = not args.strategy or "v0.2" in args.strategy
        if same_strategy:
            runs.insert(0, ("完整系统（参照，策略 v0.2）", ref_dir, "reference"))
        else:
            print(f"提示：参照组 {args.reference} 是 v0.2 的结果，与本次策略不同，不纳入对比。")

    labels = load_labels(PROJECT_ROOT / args.labels)
    results = []
    for label, directory, _key in runs:
        preds = load_predictions(directory)
        results.append((label, directory, evaluate_run(labels, preds, load_costs(directory))))

    md = render_markdown(results)
    md_path = out_root / "ablation_comparison.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = out_root / "ablation_comparison.json"
    json_path.write_text(
        json.dumps({name: m for name, _, m in results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 消融结果 ===")
    for name, _, m in results:
        print(f"{name}：macro F1={m['macro']['f1']:.3f}"
              f"｜误报率={m['false_positive_rate']:.3f}｜漏报率={m['miss_rate']:.3f}"
              f"｜失败={m.get('failed_samples', 0)}")
    print(f"\n对比表：{md_path}")
    print(f"指标 JSON：{json_path}")
    print("\n注意：消融组与完整系统要用同一个测试集、同一份策略，否则不可比。")


if __name__ == "__main__":
    main()
