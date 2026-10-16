"""命令行入口。

无论从哪个目录调用都不会因 sys.path 问题失败。
"""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import (  # noqa: E402
    evaluate_run,
    load_costs,
    load_labels,
    load_predictions,
    render_console,
    render_markdown,
)
from llm import describe_backend  # noqa: E402
from pipeline import SafetyAnalysisPipeline  # noqa: E402
from self_evolution import (  # noqa: E402
    analyze_cases,
    list_versions,
    promote_strategy,
    propose_strategy_version,
    record_feedback,
)

CURRENT_STRATEGY = PROJECT_ROOT / "configs" / "strategy_current.json"


# ------------------------------------------------------------------ 子命令


def cmd_analyze(args) -> None:
    ablation = {}
    if args.no_rules:
        ablation["use_rules"] = False
    if args.no_memory:
        ablation["use_memory"] = False
    if args.no_review:
        ablation["use_review_agent"] = False
    if args.no_agent_tools:
        ablation["use_agent_tools"] = False

    pipeline = SafetyAnalysisPipeline(
        PROJECT_ROOT,
        CURRENT_STRATEGY,
        force_mock=args.mock,
        ablation=ablation or None,
        max_video_frames=args.max_frames,
        analysis_mode=args.mode,
    )
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "reports"
    report = pipeline.analyze(Path(args.input), args.scenario, out_dir)

    s = report.summary()
    print(f"系统形态：{report.analysis_mode}｜模式：{report.mode}｜策略：{report.strategy_version}｜帧数：{report.frame_count}")
    print(f"风险条目：{s['total']}（高风险 {s['high_risk']}，需人工优先复核 {s['needs_human_review']}）")
    print(f"状态：{s['by_status']}｜Agent 分歧：{s['conflict_count']} 处")
    print(f"成本：模型调用 {report.model_calls} 次｜工具调用 {report.tool_calls} 次｜耗时 {report.duration_ms} ms")
    print(f"报告：{out_dir / (report.sample_id + '_report.md')}")
    if report.mode == "mock":
        print("注意：当前为模拟模式，结果不可作为真实识别结论。")


def cmd_feedback(args) -> None:
    memory_path = record_feedback(
        PROJECT_ROOT,
        Path(args.report),
        args.original_risk_type,
        args.correct_risk_type,
        args.correct_severity,
        args.note,
        reviewer=args.reviewer or "",
    )
    print(f"纠错案例已写入：{memory_path}")
    analysis = analyze_cases(PROJECT_ROOT)
    print(f"案例聚合：共 {analysis['total_cases']} 条，误报 {analysis['false_positive_count']} 条，"
          f"重复误判模式 {len(analysis['confusion_pairs'])} 组")

    version_path = propose_strategy_version(
        PROJECT_ROOT, CURRENT_STRATEGY, args.note, llm=get_llm(force_mock=args.mock)
    )
    draft = json.loads(version_path.read_text(encoding="utf-8"))
    latest = (draft.get("change_log") or [{}])[-1]
    print(f"新策略草案已生成：{version_path}")
    print(f"提案来源：{latest.get('proposal_source')}")
    for change in latest.get("changes", []):
        print(f"  - {change}")
    print(f"确认无误后执行：python src/main.py promote --version {draft.get('version')}")


def cmd_promote(args) -> None:
    target = promote_strategy(PROJECT_ROOT, args.version)
    print(f"已晋升为当前生效策略：{target}")


def cmd_rollback(args) -> None:
    target = promote_strategy(PROJECT_ROOT, args.version)
    print(f"已回滚到 {args.version}：{target}")


def cmd_versions(args) -> None:
    versions = list_versions(PROJECT_ROOT)
    current = json.loads(CURRENT_STRATEGY.read_text(encoding="utf-8")).get("version")
    print(f"当前生效版本：{current}")
    for v in versions:
        mark = " <- current" if v == current else ""
        print(f"  {v}{mark}")


def cmd_evaluate(args) -> None:
    labels = load_labels(Path(args.labels))
    names = args.names.split(",") if args.names else None
    runs = []
    for idx, reports in enumerate(args.reports):
        reports_dir = Path(reports)
        name = names[idx] if names and idx < len(names) else reports_dir.name
        predictions = load_predictions(reports_dir)
        runs.append((name, reports_dir, evaluate_run(labels, predictions, load_costs(reports_dir))))
    print(render_console(runs))
    if args.out_json:
        payload = {name: metrics for name, _, metrics in runs}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"指标已写入 {args.out_json}")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_markdown(runs), encoding="utf-8")
        print(f"对比表已写入 {args.out_md}")


def cmd_status(args) -> None:
    backend = describe_backend()
    print("=== 运行环境 ===")
    print(f"  模型模式：{backend['mode']}")
    print(f"  接口地址：{backend['base_url']}")
    print(f"  模型名称：{backend['model']}")
    print(f"  API Key：{'已配置' if backend['has_api_key'] else '未配置（将使用模拟模式）'}")
    try:
        import cv2  # noqa: F401
        print("  视频支持：已安装 opencv-python")
    except Exception:
        print("  视频支持：未安装 opencv-python（视频输入不可用）")
    print("=== 项目状态 ===")
    cases = analyze_cases(PROJECT_ROOT)
    print(f"  纠错案例：{cases['total_cases']} 条（误报 {cases['false_positive_count']} 条）")
    print(f"  策略版本：{', '.join(list_versions(PROJECT_ROOT)) or '无'}")


def cmd_serve(args) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.server import create_app  # noqa: PLC0415

    app = create_app(PROJECT_ROOT)
    print(f"启动服务：http://{args.host}:{args.port}")
    print("浏览器打开该地址即可使用完整界面。")
    app.run(host=args.host, port=args.port, debug=args.debug)


# ------------------------------------------------------------------ 解析器


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-agent",
        description="实验室危险场景识别与自进化 Agent 原型",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="分析图片或视频")
    p.add_argument("--input", required=True, help="图片或视频路径")
    p.add_argument("--scenario", default="chemistry_lab", help="场景名称")
    p.add_argument("--mock", action="store_true", help="强制使用模拟模式")
    p.add_argument("--mode", default="multi_agent", choices=["multi_agent", "baseline"],
                   help="multi_agent=多 Agent 协作（默认）；baseline=基础对照系统（单模型直出）")
    p.add_argument("--out-dir", default=None, help="报告输出目录")
    p.add_argument("--max-frames", type=int, default=4, help="视频最多抽帧数")
    p.add_argument("--no-rules", action="store_true", help="消融：关闭规则检索")
    p.add_argument("--no-memory", action="store_true", help="消融：关闭案例记忆")
    p.add_argument("--no-review", action="store_true", help="消融：关闭证据复核 Agent")
    p.add_argument("--no-agent-tools", action="store_true", help="消融：关闭 Agent 自主工具调用")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("feedback", help="记录人工纠错并生成策略草案")
    p.add_argument("--report", required=True, help="分析报告 JSON 路径")
    p.add_argument("--original-risk-type", required=True, help="系统原始风险类型")
    p.add_argument("--correct-risk-type", default=None, help="人工修正后的风险类型（填 normal 表示应为正常）")
    p.add_argument("--correct-severity", default=None, help="人工修正后的风险等级")
    p.add_argument("--note", required=True, help="纠错说明")
    p.add_argument("--reviewer", default=None, help="纠错人")
    p.add_argument("--mock", action="store_true", help="不调用模型，使用确定性规则提案")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("promote", help="把草案晋升为当前生效策略")
    p.add_argument("--version", required=True, help="例如 v0.2")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("rollback", help="回退策略版本")
    p.add_argument("--version", required=True, help="例如 v0.1")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("versions", help="列出所有策略版本")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("evaluate", help="评测识别结果")
    p.add_argument("--labels", required=True)
    p.add_argument("--reports", required=True, nargs="+")
    p.add_argument("--names", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("status", help="查看运行环境与项目状态")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("serve", help="启动本地 Web 演示界面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
