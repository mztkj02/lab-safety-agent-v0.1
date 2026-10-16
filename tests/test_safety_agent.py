"""核心逻辑单元测试。

覆盖：JSON 解析、中文规则检索、时间维度融合、Agent 冲突仲裁、自进化版本管理、端到端流程。
运行：python -m pytest tests -q
"""

import json
from pathlib import Path

import pytest

from agents import EvidenceReviewAgent, PerceptionAgent, RiskAnalysisAgent
from llm import MockLLM, extract_json
from models import RiskFinding
from pipeline import SafetyAnalysisPipeline
from self_evolution import list_versions, next_version, propose_strategy_version
from tools import CaseMemory, SafetyRuleRetriever


# ------------------------------------------------------------------ JSON 解析


def test_extract_json_plain():
    assert extract_json('{"a": 1}')["a"] == 1


def test_extract_json_with_code_fence():
    text = '模型回答如下：\n```json\n{"findings": []}\n```\n以上。'
    assert extract_json(text) == {"findings": []}


def test_extract_json_with_trailing_comma():
    assert extract_json('{"a": [1, 2,], }')["a"] == [1, 2]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("完全没有 JSON")


# ------------------------------------------------------------------ 规则检索


@pytest.fixture()
def retriever(project_root: Path) -> SafetyRuleRetriever:
    return SafetyRuleRetriever(project_root / "rules" / "lab_safety_rules.json")


def test_retriever_hits_chinese_keyword(retriever):
    hits = retriever.retrieve("人员没有戴护目镜在做实验")
    assert hits, "中文查询应当命中规则"
    assert hits[0]["risk_type"] == "ppe_missing"


def test_retriever_for_finding_prefers_exact_type(retriever):
    hits = retriever.retrieve_for_finding("flammable_near_heat", "酒精灯旁边放着纸张")
    assert hits and hits[0]["risk_type"] == "flammable_near_heat"
    assert hits[0]["default_severity"] == "high"


def test_retriever_returns_empty_for_unrelated_query(retriever):
    assert retriever.retrieve("今天天气不错") == []


def test_every_rule_has_source(retriever):
    for rule in retriever.rules:
        assert rule.get("source"), f"规则 {rule.get('rule_id')} 缺少来源"


# ------------------------------------------------------------------ 时间融合


def _finding(risk_type: str, confidence: float, status: str = "confirmed") -> RiskFinding:
    return RiskFinding(
        risk_type=risk_type, severity="medium", evidence="可见证据描述足够长",
        confidence=confidence, status=status,
    )


def test_temporal_fuse_merges_duplicate_frames():
    frame_results = [
        (0, "f1.jpg", [_finding("spill_or_obstruction", 0.6)]),
        (1, "f2.jpg", [_finding("spill_or_obstruction", 0.8)]),
        (2, "f3.jpg", [_finding("spill_or_obstruction", 0.7)]),
        (3, "f4.jpg", []),
    ]
    merged = SafetyAnalysisPipeline._temporal_fuse(frame_results, 4)
    assert len(merged) == 1, "同类风险应合并为一条"
    item = merged[0]
    assert item.frames_seen == 3 and item.frames_total == 4
    assert item.persistence == 0.75
    assert "3/4" in item.temporal_note
    assert item.confidence >= 0.8, "持续出现应获得置信度加成"


def test_temporal_fuse_downgrades_transient_hit():
    frame_results = [
        (0, "f1.jpg", []),
        (1, "f2.jpg", [_finding("ppe_missing", 0.8)]),
        (2, "f3.jpg", []),
        (3, "f4.jpg", []),
    ]
    merged = SafetyAnalysisPipeline._temporal_fuse(frame_results, 4)
    item = merged[0]
    assert item.frames_seen == 1
    assert item.status == "uncertain", "仅单帧出现的风险应被降级"
    assert item.confidence < 0.8
    assert item.needs_human_review is True


def test_temporal_fuse_single_image():
    merged = SafetyAnalysisPipeline._temporal_fuse([(0, "a.jpg", [_finding("food_or_drink", 0.5)])], 1)
    assert merged[0].temporal_note == "单帧图像输入"
    assert merged[0].persistence == 1.0


# ------------------------------------------------------------------ 冲突仲裁


class _StubLLM(MockLLM):
    """可控的假模型：让复核 Agent 返回指定判定，用于测试冲突处理。"""

    mode = "api"          # 冒充真实模型，绕过 simulated 分支
    is_simulated = False

    def __init__(self, verdict: str, severity: str = ""):
        self.verdict = verdict
        self.severity = severity
        self.calls = 0

    def complete_json(self, system: str, user: str):
        payload = extract_json(user)
        findings = payload.get("findings", [])
        return {"results": [
            {"index": i, "verdict": self.verdict, "severity": self.severity, "reason": "测试理由"}
            for i, _ in enumerate(findings)
        ]}


def _strategy(min_confidence: float = 0.45) -> dict:
    return {"review": {"min_confidence": min_confidence, "min_evidence_len": 8}, "prompts": {}, "agents": {}}


def test_review_marks_false_positive_conflict():
    agent = EvidenceReviewAgent(_StubLLM("rejected"), _strategy())
    findings = [RiskFinding(risk_type="ppe_missing", severity="medium",
                            evidence="水杯被当成护目镜缺失", confidence=0.9, status="analyzed")]
    reviewed, trace = agent.run(findings)
    item = reviewed[0]
    assert item.status == "rejected"
    assert item.review_verdict == "rejected"
    assert item.disagreement, "误报应记录 Agent 之间的分歧"
    assert item.needs_human_review is True
    assert any("冲突" in c for c in trace.conflicts)


def test_review_downgrades_high_risk_and_flags_human():
    agent = EvidenceReviewAgent(_StubLLM("uncertain"), _strategy())
    findings = [RiskFinding(risk_type="flammable_near_heat", severity="high",
                            evidence="加热板旁的纸张", confidence=0.9, status="analyzed")]
    reviewed, _ = agent.run(findings)
    item = reviewed[0]
    assert item.status == "uncertain" and item.severity == "uncertain"
    assert item.needs_human_review is True, "高风险被降级必须交人工优先复核"


def test_review_fallback_when_model_unavailable():
    class _Broken(_StubLLM):
        def complete_json(self, system, user):
            raise RuntimeError("网络不可用")

    agent = EvidenceReviewAgent(_Broken("confirmed"), _strategy())
    findings = [RiskFinding(risk_type="food_or_drink", severity="low",
                            evidence="实验台右侧边缘放着一个透明水杯", confidence=0.9, status="analyzed")]
    reviewed, trace = agent.run(findings)
    assert reviewed[0].status == "confirmed", "模型不可用时应回退阈值复核"
    assert any("阈值" in c for c in trace.conflicts)


def test_review_disabled_by_ablation():
    agent = EvidenceReviewAgent(_StubLLM("rejected"), _strategy(), enabled=False)
    findings = [RiskFinding(risk_type="ppe_missing", severity="medium",
                            evidence="证据足够长的一段描述", confidence=0.9, status="analyzed")]
    reviewed, trace = agent.run(findings)
    assert reviewed[0].status == "confirmed"
    assert any("已关闭" in c for c in trace.conflicts)


# ------------------------------------------------------------------ 案例记忆


def test_case_memory_roundtrip(tmp_path: Path):
    memory = CaseMemory(tmp_path / "cases.json")
    memory.add_case({"original_risk_type": "ppe_missing", "correct_risk_type": "food_or_drink",
                     "note": "水杯误判"})
    hits = memory.retrieve("food_or_drink")
    assert len(hits) == 1 and hits[0]["note"] == "水杯误判"
    assert memory.stats()["total"] == 1


# ------------------------------------------------------------------ 版本管理


def test_version_increment_no_collision(tmp_path: Path):
    (tmp_path / "versions").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "memory").mkdir()
    current = tmp_path / "configs" / "strategy_current.json"
    current.write_text(json.dumps({
        "version": "v0.2", "risk_types": ["ppe_missing"],
        "prompts": {"perception": "感知 {scenario} {risk_types}"},
    }, ensure_ascii=False), encoding="utf-8")

    assert next_version(tmp_path) == "v0.3"
    first = propose_strategy_version(tmp_path, current, "第一轮")
    assert json.loads(first.read_text(encoding="utf-8"))["version"] == "v0.3"

    # 第二次提案必须递增到 v0.4，而不是覆盖 v0.3
    second = propose_strategy_version(tmp_path, first, "第二轮")
    assert json.loads(second.read_text(encoding="utf-8"))["version"] == "v0.4"
    assert list_versions(tmp_path) == ["v0.2", "v0.3", "v0.4"]


# ------------------------------------------------------------------ 端到端


def test_pipeline_end_to_end_mock(project_root: Path, tmp_path: Path):
    sample = tmp_path / "no_goggle_demo.jpg"
    sample.write_bytes((project_root / "data" / "dev" / "ppe_example.jpg").read_bytes())

    pipeline = SafetyAnalysisPipeline(project_root, project_root / "configs" / "strategy_current.json",
                                      force_mock=True)
    report = pipeline.analyze(sample, "chemistry_lab", tmp_path / "out")

    assert report.mode == "mock"
    assert report.findings, "文件名含 no_goggle 的样本在模拟模式下应给出候选风险"
    assert all(f.status == "simulated" for f in report.findings), \
        "模拟模式的结果绝不能标记为 confirmed"
    assert report.model_calls > 0
    assert (tmp_path / "out" / f"{sample.stem}_report.json").exists()
    assert (tmp_path / "out" / f"{sample.stem}_report.md").exists()


def test_pipeline_baseline_mode_has_no_review(project_root: Path, tmp_path: Path):
    sample = tmp_path / "flammable_heater_demo.jpg"
    sample.write_bytes((project_root / "data" / "dev" / "ppe_example.jpg").read_bytes())

    pipeline = SafetyAnalysisPipeline(project_root, project_root / "configs" / "strategy_current.json",
                                      force_mock=True, analysis_mode="baseline")
    report = pipeline.analyze(sample, "chemistry_lab", tmp_path / "out")
    assert report.analysis_mode == "baseline"
    assert all(t.agent.startswith("基础识别系统") for t in report.traces), \
        "基础对照组不应出现多 Agent 的执行记录"


def test_agent_tools_are_actually_called(project_root: Path, tmp_path: Path):
    """验证 Agent 自主工具调用链路：模拟模型会先申请工具，再给出结论。"""
    from toolkit import ToolRegistry
    from tools import ToolLogger

    logger = ToolLogger(tmp_path / "tool_calls.jsonl")
    registry = ToolRegistry(
        project_root, logger,
        SafetyRuleRetriever(project_root / "rules" / "lab_safety_rules.json"),
        CaseMemory(tmp_path / "cases.json"),
        tmp_path / "tools",
    )
    agent = PerceptionAgent(MockLLM(), {"risk_types": ["ppe_missing"],
                                        "prompts": {"perception": "场景 {scenario} 类型 {risk_types}"}},
                            registry=registry)
    result, trace = agent.run(project_root / "data" / "dev" / "ppe_example.jpg", "chemistry_lab")

    assert trace.tool_calls, "感知 Agent 应当自主发起工具调用"
    names = {c["tool"] for c in trace.tool_calls}
    assert {"search_rules", "search_cases"} <= names
    assert result.get("findings"), "工具调用后应给出最终候选风险"
    log_lines = (tmp_path / "tool_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) >= len(trace.tool_calls), "每次工具调用都要留日志"


def test_ablation_flag_recorded(project_root: Path, tmp_path: Path):
    sample = tmp_path / "drink_cup_demo.jpg"
    sample.write_bytes((project_root / "data" / "dev" / "ppe_example.jpg").read_bytes())
    pipeline = SafetyAnalysisPipeline(project_root, project_root / "configs" / "strategy_current.json",
                                      force_mock=True, ablation={"use_rules": False})
    report = pipeline.analyze(sample, "chemistry_lab", tmp_path / "out")
    assert report.config["ablation"]["use_rules"] is False
