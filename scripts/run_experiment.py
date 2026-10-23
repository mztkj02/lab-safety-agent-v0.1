"""一键跑通课题要求的三组对比实验。

对应课程要求：“建议比较：基础识别系统 / 加入多智能体与工具调用的系统 / 两轮自进化后的系统”，
并报告精确率、召回率、误报与漏报、处理时间和调用成本。

用法：
    # 只跑前两组
    python scripts/run_experiment.py --labels data/demo_labels_test.csv --groups base,multi

    # 完整三组：先在开发集上完成两轮自进化，再在测试集上评第三组
    python scripts/run_experiment.py --labels data/demo_labels_test.csv --auto-evolve --rounds 2

注意：
- 自进化只使用开发集（--dev-labels），测试集绝不参与策略更新或案例记忆，符合课程要求。
- 输出目录下会生成 comparison.md 与 comparison.json，可直接放进项目报告。
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_run, load_costs, load_labels, load_predictions, render_markdown  # noqa: E402
from llm import describe_backend, get_llm  # noqa: E402
from pipeline import SafetyAnalysisPipeline  # noqa: E402
from self_evolution import list_versions, propose_strategy_version, promote_strategy  # noqa: E402
from tools import CaseMemory  # noqa: E402

CURRENT_STRATEGY = PROJECT_ROOT / "configs" / "strategy_current.json"


def read_samples(labels_path: Path) -> List[Dict[str, str]]:
    rows = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("file_path"):
                rows.append(row)
    return rows


def run_group(
    name: str,
    samples: List[Dict[str, str]],
    out_dir: Path,
    mock: bool,
    analysis_mode: str,
    sleep: float = 0.0,
    resume: bool = False,
) -> Dict[str, int]:
    """跑一组样本。

    `resume=True` 时不清空目录、跳过已有报告的样本——真实模型跑 80 张要几十分钟，
    中途断线（429、断网、手滑 Ctrl+C）后不该从头再来。
    """
    if out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SafetyAnalysisPipeline(
        PROJECT_ROOT, CURRENT_STRATEGY, force_mock=mock, analysis_mode=analysis_mode
    )
    print(f"\n--- 运行实验组：{name}（{len(samples)} 个样本，模式 {analysis_mode}）---")

    done = skipped = 0
    failures: List[str] = []
    started = time.time()

    for idx, row in enumerate(samples, 1):
        sample_id = row.get("sample_id") or Path(row["file_path"]).stem
        report_path = out_dir / f"{sample_id}_report.json"

        if resume and report_path.exists():
            done += 1
            skipped += 1
            continue

        input_path = PROJECT_ROOT / row["file_path"]
        if not input_path.exists():
            print(f"  [{idx}/{len(samples)}] 跳过（文件不存在）：{row['file_path']}")
            failures.append(row["file_path"])
            continue

        try:
            pipeline.analyze(input_path, row.get("scene") or "chemistry_lab", out_dir)
        except KeyboardInterrupt:
            print(f"\n  已中断。已完成的 {done} 个样本可直接用 `--resume` 续跑。")
            raise
        except Exception as err:  # noqa: BLE001
            print(f"  [{idx}/{len(samples)}] 失败 {input_path.name}：{err}")
            failures.append(input_path.name)
        else:
            done += 1
            elapsed = time.time() - started
            # 估算剩余时间：真跑 80 张时，能不能去吃饭全看这个数字
            eta = (elapsed / done) * (len(samples) - idx) if done else 0
            print(f"  [{idx}/{len(samples)}] {input_path.name}"
                  f"  (已用 {elapsed/60:.1f} min，预计还需 {eta/60:.1f} min)")

        if sleep > 0 and idx < len(samples):
            time.sleep(sleep)

    print(f"  完成 {done} 个（其中续跑跳过 {skipped} 个），失败 {len(failures)} 个")
    return {"done": done, "skipped": skipped, "failed": len(failures)}



def auto_evolve(samples: List[Dict[str, str]], rounds: int, mock: bool, work_dir: Path,
                sleep: float = 0.0, resume: bool = False) -> None:
    """在开发集上完成若干轮“纠错 -> 提案 -> 晋升”。"""
    memory = CaseMemory(PROJECT_ROOT / "memory" / "cases.json")
    for round_no in range(1, rounds + 1):
        round_dir = work_dir / f"dev_round{round_no}"
        run_group(f"开发集第 {round_no} 轮（进化前）", samples, round_dir, mock, "multi_agent",
                  sleep=sleep, resume=resume)

        new_cases = _collect_cases_from_predictions(samples, round_dir)
        for case in new_cases:
            memory.add_case(case)
        print(f"  第 {round_no} 轮：从开发集自动生成 {len(new_cases)} 条纠错案例（模拟人工纠错）")

        draft = propose_strategy_version(
            PROJECT_ROOT, CURRENT_STRATEGY,
            f"第 {round_no} 轮自进化：修正开发集上暴露的误报与漏报",
            llm=get_llm(force_mock=mock),
        )
        data = json.loads(draft.read_text(encoding="utf-8"))
        promote_strategy(PROJECT_ROOT, data["version"])
        latest = (data.get("change_log") or [{}])[-1]
        print(f"  已晋升策略 {data['version']}（提案来源：{latest.get('proposal_source')}）")
        for change in latest.get("changes", []):
            print(f"    - {change}")


def _collect_cases_from_predictions(samples: List[Dict[str, str]], reports_dir: Path) -> List[Dict]:
    cases = []
    for row in samples:
        sample_id = row["sample_id"]
        truth = row.get("risk_type") or "normal"
        report_path = reports_dir / f"{sample_id}_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        predicted: Set[str] = {
            f["risk_type"] for f in report.get("findings", [])
            if f.get("status") in ("confirmed", "simulated", "analyzed")
        }

        if truth == "normal":
            for wrong in predicted:
                cases.append(_case(sample_id, report, wrong, "normal",
                                   "开发集标注为正常场景，系统却报了风险，属于误报。"))
        elif truth not in predicted:
            others = sorted(predicted)
            cases.append(_case(
                sample_id, report, others[0] if others else truth, truth,
                "开发集标注的真实风险与系统输出不一致，需要修正。",
            ))
    return cases


def _case(sample_id: str, report: Dict, original: str, corrected: str, note: str) -> Dict:
    return {
        "report_sample_id": sample_id,
        "input_path": report.get("input_path"),
        "strategy_version": report.get("strategy_version"),
        "original_risk_type": original,
        "correct_risk_type": corrected,
        "note": note,
        "reviewer": "auto_dev_loop",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="三组对比实验（基础 / 多 Agent / 两轮自进化）")
    parser.add_argument("--labels", default="data/demo_labels_test.csv", help="测试集标注 CSV")
    parser.add_argument("--dev-labels", default="data/demo_labels_dev.csv", help="开发集标注 CSV（自进化用）")
    parser.add_argument("--groups", default="base,multi", help="要跑的组：base,multi,evolved")
    parser.add_argument("--auto-evolve", action="store_true", help="自动完成两轮自进化后再评 evolved 组")
    parser.add_argument("--rounds", type=int, default=2, help="自进化轮数")
    parser.add_argument("--out", default="reports/experiments", help="输出根目录")
    parser.add_argument("--mock", action="store_true", help="强制模拟模式")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="每个样本之间的间隔秒数（默认 1.0；免费模型并发低，调大可减少 429）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：不清空输出目录，跳过已有报告的样本")
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 个样本（0 表示全部），用于先小样本试跑验证链路")
    args = parser.parse_args()

    backend = describe_backend()
    use_mock = args.mock or backend["mode"] == "mock"
    mode_label = "模拟模式" if use_mock else f"真实模型 {backend['model']}"
    print(f"运行模式：{mode_label}")
    if not use_mock:
        print(f"接口：{backend['base_url']}｜图片传参：{backend.get('image_format')}")
    if mode_label == "模拟模式":
        print("提示：模拟模式下所有结论都会标记为 simulated，指标只能证明评测管线可用，不能作为实验结论。")

    test_samples = read_samples(PROJECT_ROOT / args.labels)
    if not test_samples:
        print(f"错误：{args.labels} 中没有样本，请先执行 python scripts/make_demo_dataset.py")
        sys.exit(1)
    if args.limit > 0:
        test_samples = test_samples[: args.limit]
        print(f"已按 --limit 只取前 {len(test_samples)} 个样本")

    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    names = {"base": "基础识别系统（单模型直出）", "multi": "多 Agent + 工具调用", "evolved": "两轮自进化后"}
    run_dirs: List[tuple] = []
    kw = {"sleep": args.sleep, "resume": args.resume}

    if "base" in groups:
        run_group(names["base"], test_samples, out_root / "base", args.mock, "baseline", **kw)
        run_dirs.append((names["base"], out_root / "base"))

    if "multi" in groups:
        run_group(names["multi"], test_samples, out_root / "multi", args.mock, "multi_agent", **kw)
        run_dirs.append((names["multi"], out_root / "multi"))

    if "evolved" in groups or args.auto_evolve:
        dev_samples = read_samples(PROJECT_ROOT / args.dev_labels)
        if not dev_samples:
            print(f"警告：{args.dev_labels} 为空，跳过自进化。")
        else:
            if args.limit > 0:
                dev_samples = dev_samples[: args.limit]
            print("\n=== 开始两轮自进化（仅使用开发集）===")
            auto_evolve(dev_samples, args.rounds, args.mock, out_root / "_dev", **kw)
            run_group(names["evolved"], test_samples, out_root / "evolved", args.mock, "multi_agent", **kw)
            run_dirs.append((names["evolved"], out_root / "evolved"))

    # ---------------- 评测 ----------------
    labels = load_labels(PROJECT_ROOT / args.labels)
    runs = []
    for name, directory in run_dirs:
        preds = load_predictions(directory)
        runs.append((name, directory, evaluate_run(labels, preds, load_costs(directory))))

    header = ""
    if mode_label == "模拟模式":
        header = (
            "> ⚠️ **本表由模拟模式生成**：样本结论均标记为 simulated，只能验证评测管线是否跑通，\n"
            "> **不能作为实验结果**。接入真实模型 API 后重新运行本脚本才会得到有效数据。\n\n"
        )
    elif args.limit > 0:
        header = (
            f"> ⚠️ **本表仅用前 {len(test_samples)} 个样本**（`--limit` 试跑），"
            "指标只用于确认链路可用，**不是最终实验结果**。\n\n"
        )

    md_path = out_root / "comparison.md"
    md_path.write_text(header + render_markdown(runs), encoding="utf-8")
    json_path = out_root / "comparison.json"
    json_path.write_text(
        json.dumps({name: m for name, _, m in runs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 对比结果 ===")
    for name, _, m in runs:
        print(f"{name}：macro F1={m['macro']['f1']:.3f}｜micro F1={m['micro']['f1']:.3f}"
              f"｜误报样本率={m['false_positive_rate']:.3f}｜漏报样本率={m['miss_rate']:.3f}")
    print(f"\n对比表：{md_path}")
    print(f"指标 JSON：{json_path}")
    print(f"当前策略版本：{', '.join(list_versions(PROJECT_ROOT))}")


if __name__ == "__main__":
    main()
