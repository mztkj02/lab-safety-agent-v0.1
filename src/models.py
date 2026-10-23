"""数据结构定义。

约定（重要）：
- severity：low / medium / high / uncertain
- status：
    candidate  感知阶段产生的候选风险，尚未分析
    analyzed   风险分析 Agent 已完成定级
    confirmed  证据复核 Agent 确认成立
    uncertain  证据不足，无法可靠判断
    rejected   复核判定为误报
    simulated  模拟(mock)模式产生的结果 —— 永远不能冒充真实识别结论
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SEVERITIES = ("low", "medium", "high", "uncertain")
STATUSES = ("candidate", "analyzed", "confirmed", "uncertain", "rejected", "simulated")

#: 只有这几种状态才被视为“系统给出的有效风险结论”
EFFECTIVE_STATUSES = ("confirmed", "simulated", "uncertain")


@dataclass
class RiskFinding:
    risk_type: str
    severity: str = "uncertain"
    evidence: str = ""
    location: str = "unknown"
    time_range: str = "unknown"
    confidence: float = 0.0
    recommendation: str = ""
    uncertainty: str = ""
    status: str = "candidate"
    #: 该结论来自哪张图/哪一帧
    source_image: str = ""
    #: 命中的规则 id 与案例 id，便于报告里追溯依据
    rule_ids: List[str] = field(default_factory=list)
    case_ids: List[str] = field(default_factory=list)

    # ---- Agent 意见冲突（课题要求“处理意见不一致的情况”） ----
    #: 风险分析 Agent 给出的原始等级（复核前的判定），用于展示分歧
    analysis_severity: str = ""
    #: 复核 Agent 的原始判定：confirmed / uncertain / rejected
    review_verdict: str = ""
    #: 两个 Agent 结论不一致时的分歧说明；一致时为空
    disagreement: str = ""
    #: 分歧严重到必须人工优先复核（例如分析判高风险、复核却因证据不足降级）
    needs_human_review: bool = False

    # ---- 视频时间维度（课题要求“视频分析应结合多个时间点的信息”） ----
    #: 在多少帧中被检测到
    frames_seen: int = 0
    #: 本次分析总帧数（图片输入为 1）
    frames_total: int = 0
    #: 出现比例 = frames_seen / frames_total
    persistence: float = 0.0
    #: 形如 “第 2~4 帧（持续 3/4 帧）”
    temporal_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskFinding":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AgentTrace:
    agent: str
    input_summary: str
    output_summary: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None
    #: 该 Agent 执行中出现的致命错误（调用失败等）。非空表示这一步没真正完成，
    #: 不能把「没输出风险」解释成「场景安全」。
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    sample_id: str
    input_path: str
    scenario: str
    mode: str
    strategy_version: str
    findings: List[RiskFinding]
    traces: List[AgentTrace]
    #: multi_agent（默认） / baseline（基础对照系统，单模型直出）
    analysis_mode: str = "multi_agent"
    config: Dict[str, Any] = field(default_factory=dict)
    frame_count: int = 1
    generated_at: str = ""
    duration_ms: Optional[int] = None
    #: 模型调用次数与工具调用次数，用于成本对比
    model_calls: int = 0
    tool_calls: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    #: 分析失败记录。任何一帧的模型调用彻底失败都会记在这里——
    #: 失败绝不能被当成「未发现风险」静默吞掉，否则漏报率会被系统性低估。
    analysis_errors: List[str] = field(default_factory=list)
    disclaimer: str = (
        "本结果用于教学与辅助分析，必须人工复核；“未识别到风险”不等于“场景安全”。"
        "模拟(mock)模式的输出不构成任何真实识别结论。"
    )

    @property
    def failed(self) -> bool:
        """整体分析是否失败（有帧彻底没跑成）。"""
        return bool(self.analysis_errors)

    def summary(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for f in self.findings:
            by_status[f.status] = by_status.get(f.status, 0) + 1
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
        return {
            "total": len(self.findings),
            "effective": sum(1 for f in self.findings if f.status not in ("rejected",)),
            "by_status": by_status,
            "by_severity": by_severity,
            "high_risk": sum(1 for f in self.findings if f.severity == "high"),
            "needs_human_review": sum(1 for f in self.findings if f.needs_human_review),
            "conflict_count": len(self.conflicts),
            "analysis_failed": self.failed,
            "analysis_error_count": len(self.analysis_errors),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "input_path": self.input_path,
            "scenario": self.scenario,
            "mode": self.mode,
            "analysis_mode": self.analysis_mode,
            "strategy_version": self.strategy_version,
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
            "frame_count": self.frame_count,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "config": self.config,
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.findings],
            "conflicts": self.conflicts,
            "analysis_errors": self.analysis_errors,
            "traces": [t.to_dict() for t in self.traces],
            "disclaimer": self.disclaimer,
        }
