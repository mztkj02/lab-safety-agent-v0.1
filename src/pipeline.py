"""分析主流程：输入 -> 预处理 -> 多 Agent 协作（含自主工具调用）-> 时间融合 -> 报告落盘。"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents import (
    BaselineAnalyzer,
    EvidenceReviewAgent,
    PerceptionAgent,
    RiskAnalysisAgent,
)
from llm import get_llm
from models import AgentTrace, AnalysisReport, RiskFinding
from toolkit import ToolRegistry
from tools import CaseMemory, SafetyRuleRetriever, ToolLogger, VideoFrameExtractor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#: 风险在少数帧中“一闪而过”时的置信度折扣
TRANSIENT_PENALTY = 0.7
#: 风险在多数帧中持续出现时的置信度加成
PERSISTENT_BONUS = 0.05


class SafetyAnalysisPipeline:
    def __init__(
        self,
        project_root: Path,
        strategy_path: Path,
        force_mock: bool = False,
        ablation: Dict[str, bool] | None = None,
        max_video_frames: int = 4,
        analysis_mode: str = "multi_agent",
    ):
        self.project_root = project_root
        self.strategy_path = strategy_path
        self.strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        self.llm = get_llm(force_mock=force_mock)
        self.max_video_frames = max_video_frames
        self.analysis_mode = analysis_mode

        flags = dict(self.strategy.get("ablation", {}))
        flags.update(ablation or {})
        self.flags = {
            "use_rules": bool(flags.get("use_rules", True)),
            "use_memory": bool(flags.get("use_memory", True)),
            "use_review_agent": bool(flags.get("use_review_agent", True)),
            "use_agent_tools": bool(flags.get("use_agent_tools", True)),
        }

        self.tool_logger = ToolLogger(project_root / "reports" / "tool_calls.jsonl")
        self.rules = SafetyRuleRetriever(project_root / "rules" / "lab_safety_rules.json")
        self.memory = CaseMemory(project_root / "memory" / "cases.json")

    # ------------------------------------------------------------ 主入口
    def analyze(self, input_path: Path, scenario: str, output_dir: Path) -> AnalysisReport:
        started = time.time()
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在：{input_path}")

        # 复用同一个 pipeline 连续分析多个样本时，成本要按本次增量计算
        calls_before = int(getattr(self.llm, "calls", 0))
        tools_before = len(self.tool_logger.calls)

        output_dir.mkdir(parents=True, exist_ok=True)
        sample_id = input_path.stem
        traces: List[AgentTrace] = []

        image_paths = self._resolve_inputs(input_path, output_dir)
        frame_results: List[Tuple[int, str, List[RiskFinding]]] = []

        if self.analysis_mode == "baseline":
            analyzer = BaselineAnalyzer(self.llm, self.strategy)
            for idx, image_path in enumerate(image_paths):
                findings, trace = analyzer.run(image_path, scenario)
                traces.append(trace)
                frame_results.append((idx, image_path.name, findings))
        else:
            registry = self._build_registry(output_dir) if self.flags["use_agent_tools"] else None
            for idx, image_path in enumerate(image_paths):
                findings, frame_traces = self._analyze_frame(
                    image_path, scenario, registry, frame_index=idx
                )
                traces.extend(frame_traces)
                frame_results.append((idx, image_path.name, findings))

        total_frames = len(image_paths)
        merged = self._temporal_fuse(frame_results, total_frames)
        conflicts = self._collect_conflicts(merged)

        report = AnalysisReport(
            sample_id=sample_id,
            input_path=str(input_path),
            scenario=scenario,
            mode=self.llm.mode,
            analysis_mode=self.analysis_mode,
            strategy_version=self.strategy.get("version", "unknown"),
            findings=merged,
            traces=traces,
            config={
                "ablation": self.flags,
                "max_video_frames": self.max_video_frames,
                "model": getattr(self.llm, "model", "mock"),
                "analysis_mode": self.analysis_mode,
            },
            frame_count=total_frames,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_ms=int((time.time() - started) * 1000),
            model_calls=int(getattr(self.llm, "calls", 0)) - calls_before,
            tool_calls=len(self.tool_logger.calls) - tools_before,
            conflicts=conflicts,
        )
        self._save_report(report, output_dir)
        self.tool_logger.record(
            "report.save",
            {"sample": sample_id, "dir": str(output_dir)},
            f"落盘 {sample_id}_report.json / .md",
        )
        return report

    # ------------------------------------------------------------ 单帧处理
    def _analyze_frame(
        self, image_path: Path, scenario: str, registry: ToolRegistry | None, frame_index: int
    ) -> Tuple[List[RiskFinding], List[AgentTrace]]:
        traces: List[AgentTrace] = []

        perception_agent = PerceptionAgent(self.llm, self.strategy, registry=registry)
        perception, trace = perception_agent.run(image_path, scenario)
        traces.append(trace)
        self.tool_logger.record(
            "llm.analyze_image",
            {"image": str(image_path)},
            perception.get("scene_description", "")[:200],
        )

        risk_agent = RiskAnalysisAgent(
            self.llm, self.rules, self.memory, self.strategy,
            use_rules=self.flags["use_rules"], use_memory=self.flags["use_memory"],
        )
        analyzed, trace = risk_agent.run(perception)
        traces.append(trace)

        review_agent = EvidenceReviewAgent(self.llm, self.strategy, enabled=self.flags["use_review_agent"])
        reviewed, trace = review_agent.run(analyzed)
        traces.append(trace)

        for item in reviewed:
            item.source_image = image_path.name
        return reviewed, traces

    def _build_registry(self, output_dir: Path) -> ToolRegistry:
        return ToolRegistry(
            self.project_root,
            self.tool_logger,
            self.rules,
            self.memory,
            output_dir / "tools",
            enabled={
                "search_rules": self.flags["use_rules"],
                "search_cases": self.flags["use_memory"],
            },
        )

    # ------------------------------------------------------------ 输入解析
    def _resolve_inputs(self, input_path: Path, output_dir: Path) -> List[Path]:
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return [input_path]
        if suffix in VIDEO_SUFFIXES:
            frames_dir = output_dir / "frames"
            extractor = VideoFrameExtractor(max_frames=self.max_video_frames)
            frames = extractor.extract(input_path, frames_dir)
            if not frames:
                if not extractor.available():
                    raise RuntimeError("视频抽帧失败：未安装 opencv-python，请先 `pip install opencv-python`。")
                raise RuntimeError("视频抽帧失败：无法读取该视频文件。")
            self.tool_logger.record(
                "video.extract_frames",
                {"video": str(input_path), "max_frames": self.max_video_frames},
                f"按时间轴均匀抽取 {len(frames)} 帧",
            )
            return frames
        raise ValueError(
            f"不支持的文件类型：{suffix}（支持图片 {'/'.join(sorted(IMAGE_SUFFIXES))} "
            f"与视频 {'/'.join(sorted(VIDEO_SUFFIXES))}）"
        )

    # ------------------------------------------------------------ 时间融合
    @staticmethod
    def _temporal_fuse(
        frame_results: List[Tuple[int, str, List[RiskFinding]]], total_frames: int
    ) -> List[RiskFinding]:
        """跨帧融合：跟踪同一风险在时间轴上的持续情况，合并重复预警。

        课题要求“视频分析应结合多个时间点的信息”，这里体现为：
        - 记录该风险出现在哪些帧、持续帧数与出现比例；
        - 只在极少数帧一闪而过 -> 判定更可能是瞬时误检，降置信度并降级；
        - 在多数帧中持续出现   -> 证据更强，升置信度。
        """
        buckets: Dict[str, List[Tuple[int, str, RiskFinding]]] = {}
        for frame_index, frame_name, findings in frame_results:
            for finding in findings:
                buckets.setdefault(finding.risk_type, []).append((frame_index, frame_name, finding))

        merged: List[RiskFinding] = []
        for risk_type, items in buckets.items():
            best = max(items, key=lambda x: x[2].confidence)[2]
            item = RiskFinding.from_dict(best.to_dict())

            frames_seen = len({idx for idx, _, _ in items})
            item.frames_seen = frames_seen
            item.frames_total = total_frames
            item.persistence = round(frames_seen / total_frames, 3) if total_frames else 0.0

            frame_ids = sorted({idx for idx, _, _ in items})
            if total_frames > 1:
                span = f"第 {frame_ids[0] + 1}~{frame_ids[-1] + 1} 帧" if len(frame_ids) > 1 \
                    else f"仅第 {frame_ids[0] + 1} 帧"
                item.temporal_note = f"{span}（持续 {frames_seen}/{total_frames} 帧，占 {item.persistence:.0%}）"

                if item.persistence <= 1 / total_frames and total_frames >= 3:
                    item.confidence = round(item.confidence * TRANSIENT_PENALTY, 3)
                    item.uncertainty = _join(item.uncertainty, "该风险仅在单帧出现，可能是瞬时误检，已降低置信度")
                    if item.status == "confirmed":
                        item.status = "uncertain"
                        item.severity = "uncertain"
                        item.needs_human_review = True
                elif item.persistence >= 0.5:
                    item.confidence = round(min(item.confidence + PERSISTENT_BONUS, 1.0), 3)
                    item.uncertainty = _join(item.uncertainty, "该风险在多数帧中持续出现，证据较强")
            else:
                item.temporal_note = "单帧图像输入"

            merged.append(item)

        merged.sort(key=lambda f: (f.severity != "high", f.severity != "medium", -f.confidence))
        return merged

    # ------------------------------------------------------------ 冲突汇总
    @staticmethod
    def _collect_conflicts(findings: List[RiskFinding]) -> List[Dict[str, Any]]:
        records = []
        for f in findings:
            if not f.disagreement:
                continue
            records.append({
                "risk_type": f.risk_type,
                "analysis_view": f.analysis_severity or f.severity,
                "review_verdict": f.review_verdict,
                "disagreement": f.disagreement,
                "resolution": {
                    "rejected": "采纳复核意见，判定为误报，交人工确认",
                    "uncertain": "按保守原则降级为不确定" + ("，并列入人工优先复核" if f.needs_human_review else ""),
                    "confirmed": "采纳复核意见，但列入人工抽查",
                }.get(f.review_verdict, "记录待人工判定"),
                "needs_human_review": f.needs_human_review,
            })
        return records

    # ------------------------------------------------------------ 报告落盘
    def _save_report(self, report: AnalysisReport, output_dir: Path) -> None:
        json_path = output_dir / f"{report.sample_id}_report.json"
        md_path = output_dir / f"{report.sample_id}_report.md"
        json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path.write_text(self._render_markdown(report), encoding="utf-8")

    @staticmethod
    def _render_markdown(report: AnalysisReport) -> str:
        s = report.summary()
        mode_label = "模拟模式（不可作为真实识别结论）" if report.mode == "mock" else "真实模型 API"
        approach = {
            "baseline": "基础识别系统（单模型直出，无 Agent / 无工具 / 无规则）",
            "multi_agent": "多 Agent 协作 + 自主工具调用",
        }.get(report.analysis_mode, report.analysis_mode)

        lines = [
            f"# 安全分析报告：{report.sample_id}",
            "",
            f"- 输入：`{report.input_path}`",
            f"- 场景：{report.scenario}",
            f"- 系统形态：{approach}",
            f"- 运行模式：{mode_label}",
            f"- 策略版本：{report.strategy_version}",
            f"- 处理帧数：{report.frame_count}",
            f"- 生成时间：{report.generated_at}",
            f"- 处理耗时：{report.duration_ms} ms",
            f"- 模型调用：{report.model_calls} 次｜工具调用：{report.tool_calls} 次",
            "",
            "## 结论概览",
            "",
            f"- 风险条目：{s['total']}（其中高风险 {s['high_risk']}）",
            f"- 状态分布：{_fmt_counter(s['by_status'])}",
            f"- 等级分布：{_fmt_counter(s['by_severity'])}",
            f"- 需人工优先复核：{s['needs_human_review']} 条",
            f"- Agent 意见分歧：{s['conflict_count']} 处",
            "",
            "## 风险明细",
        ]

        effective = [f for f in report.findings if f.status != "rejected"]
        rejected = [f for f in report.findings if f.status == "rejected"]

        if not effective:
            lines.extend(["", "未识别到明确风险。注意：这不等于场景安全。"])

        for idx, item in enumerate(effective, 1):
            flags = []
            if item.needs_human_review:
                flags.append("⚠ 需人工优先复核")
            lines.extend([
                "",
                f"### {idx}. {item.risk_type}" + ("　" + " ".join(flags) if flags else ""),
                f"- 等级：{item.severity}",
                f"- 状态：{item.status}",
                f"- 置信度：{item.confidence:.2f}",
                f"- 位置 / 来源帧：{item.location} / {item.time_range}",
                f"- 时间维度：{item.temporal_note or '不适用'}",
                f"- 证据：{item.evidence}",
                f"- 建议：{item.recommendation}",
                f"- 不确定性：{item.uncertainty or '无'}",
                f"- 判定依据：规则 {item.rule_ids or '无'}；案例 {item.case_ids or '无'}",
            ])
            if item.disagreement:
                lines.append(f"- **Agent 分歧**：{item.disagreement}（复核判定：{item.review_verdict}）")

        if rejected:
            lines.extend(["", "## 复核驳回的误报"])
            for item in rejected:
                lines.append(f"- {item.risk_type}：{item.uncertainty}")

        if report.conflicts:
            lines.extend(["", "## Agent 意见冲突与仲裁", ""])
            lines.append("| 风险类型 | 风险分析 Agent | 证据复核 Agent | 分歧 | 仲裁结论 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for c in report.conflicts:
                lines.append(
                    f"| {c['risk_type']} | {c['analysis_view']} | {c['review_verdict']} "
                    f"| {c['disagreement']} | {c['resolution']} |"
                )

        lines.extend(["", "## Agent 执行过程"])
        for trace in report.traces:
            ms = f"（{trace.duration_ms} ms）" if trace.duration_ms else ""
            lines.append(f"- **{trace.agent}**{ms}：{trace.output_summary}")
            for call in trace.tool_calls:
                if call.get("tool") and "enabled" not in call:
                    lines.append(f"  - 工具调用：{call['tool']} → {'成功' if call.get('ok') else '失败'}")
            for conflict in trace.conflicts:
                lines.append(f"  - {conflict}")

        lines.extend([
            "",
            "## 运行配置",
            "",
            "```json",
            json.dumps(report.config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 免责声明",
            "",
            report.disclaimer,
            "",
        ])
        return "\n".join(lines)


def _fmt_counter(counter: Dict[str, int]) -> str:
    if not counter:
        return "无"
    return "，".join(f"{k} {v}" for k, v in sorted(counter.items()))


def _join(a: str, b: str) -> str:
    parts = [p for p in [str(a).strip(), str(b).strip()] if p]
    return "；".join(parts)
