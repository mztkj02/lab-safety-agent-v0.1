"""多 Agent 协作层。

包含四个角色：
    PerceptionAgent      场景感知：看图、提候选风险，并按需自主申请工具
    RiskAnalysisAgent    风险分析：结合规则与历史案例定级
    EvidenceReviewAgent  证据复核：以“质疑者”立场压制误报，并与分析结论做冲突仲裁
    BaselineAnalyzer     基础对照系统：单模型直出，无 Agent、无工具、无规则，用于实验对比

设计原则：
1. 每个 Agent 都要真正做出判断，而不是只做阈值过滤。
2. 每个 Agent 都有确定性回退：模型调用失败不会让整条链路崩掉。
3. 模拟(mock)模式下，最终结论一律标记为 simulated，绝不冒充真实识别结果。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm import BaseLLM
from models import AgentTrace, RiskFinding
from toolkit import ToolRegistry
from tools import CaseMemory, SafetyRuleRetriever

ANALYSIS_SYSTEM = (
    "你是高校实验室安全风险分析专家。给定候选风险与检索到的安全规则，"
    "请判定风险等级并给出可执行的处理建议。只输出 JSON。"
)
REVIEW_SYSTEM = (
    "你是实验室安全结论复核专家，职责是挑毛病而不是附和。"
    "只有当证据确实支持该风险时才确认；证据模糊、被遮挡、"
    "或仅凭推测时必须降级为 uncertain。只输出 JSON。"
)

#: 老版本策略配置里可能缺字段，这里给兜底提示词，保证回滚到旧版本也不会崩。
DEFAULT_PERCEPTION_PROMPT = (
    "你是高校实验室安全场景感知 Agent。请分析输入图像，场景为 {scenario}。只允许输出 JSON，"
    "字段包括 scene_description 和 findings。findings 中每项包含 risk_type、severity、evidence、"
    "location、time_range、confidence、uncertainty。risk_type 必须从以下类型中选择：{risk_types}。"
    "如果证据不足，请明确写 uncertainty，不要编造。severity 只能是 low、medium、high、uncertain。"
)
DEFAULT_ANALYSIS_PROMPT = (
    "请为每个候选风险判定 severity（low/medium/high/uncertain）、给出 recommendation、"
    "confidence（0~1）与 uncertainty，并简述 reasoning。\n"
    "风险等级定义：{risk_levels}\n"
    "输出 JSON，格式：{\"results\": [{\"index\": 0, \"severity\": \"...\", "
    "\"recommendation\": \"...\", \"confidence\": 0.0, \"uncertainty\": \"...\", \"reasoning\": \"...\"}]}"
)
DEFAULT_REVIEW_PROMPT = (
    "请逐条复核下列风险结论，判断证据是否支持该结论。verdict 取值："
    "confirmed（证据支持）/ uncertain（证据不足）/ rejected（明显误报）。"
    "输出 JSON，格式：{\"results\": [{\"index\": 0, \"verdict\": \"...\", \"severity\": \"...\", "
    "\"reason\": \"...\"}]}"
)
DEFAULT_BASELINE_PROMPT = (
    "分析这张高校实验室照片，找出其中的安全风险。场景为 {scenario}。"
    "risk_type 只能从以下类型中选择：{risk_types}。只输出 JSON，格式："
    "{\"scene_description\": \"...\", \"findings\": [{\"risk_type\": \"...\", \"severity\": \"low|medium|high\", "
    "\"evidence\": \"...\", \"location\": \"...\", \"recommendation\": \"...\"}]}"
)

MAX_TOOL_ROUNDS = 2
MAX_IMAGES_PER_CALL = 4


def _clip(text: Any, limit: int = 160) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _prompt(strategy: Dict[str, Any], key: str, default: str) -> str:
    return (strategy.get("prompts") or {}).get(key) or default


def _format_prompt(template: str, **values: Any) -> str:
    """只替换已知占位符。

    提示词里带有 JSON 输出示例（含大量花括号），直接用 str.format 会被当成占位符而报错。
    这里改成定向替换，只有 {scenario} / {risk_types} / {risk_levels} 这类占位符会被替换。
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _normalize_requests(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict) and (item.get("tool") or item.get("name")):
            out.append({"tool": item.get("tool") or item.get("name"), "args": item.get("args") or {}})
        elif isinstance(item, str) and item.strip():
            out.append({"tool": item.strip(), "args": {}})
    return out


def _tool_context(observations: str, round_no: int, max_rounds: int) -> str:
    return (
        f"\n\n【工具调用结果】第 {round_no} 轮\n{observations}\n\n"
        f"请结合以上新信息重新判断。如果还需要更多信息，可以继续在 tool_requests 中申请工具"
        f"（剩余 {max(0, max_rounds - round_no)} 轮）；如果信息已足够，请给出最终结论并把 tool_requests 置为空数组。"
    )


# ------------------------------------------------------------------ 场景感知


class PerceptionAgent:
    """场景感知 Agent：看图、提候选风险，并可按需自主申请工具。"""

    name = "场景感知 Agent"

    def __init__(self, llm: BaseLLM, strategy: Dict[str, Any], registry: Optional[ToolRegistry] = None):
        self.llm = llm
        self.strategy = strategy
        self.registry = registry

    def _build_prompt(self, scenario: str) -> str:
        prompt = _format_prompt(
            _prompt(self.strategy, "perception", DEFAULT_PERCEPTION_PROMPT),
            scenario=scenario,
            risk_types=json.dumps(self.strategy.get("risk_types", []), ensure_ascii=False),
        )
        notes = self.strategy.get("risk_type_notes") or {}
        if notes:
            lines = [f"- {k}：{v}" for k, v in notes.items()]
            prompt += "\n\n历史纠错得到的易混淆说明（务必遵守）：\n" + "\n".join(lines)
        if self.registry is not None:
            prompt += "\n\n" + self.registry.manifest()
            prompt += (
                "\n\n输出中额外增加 tool_requests 字段；不需要工具时填空数组。"
                "\n当前你只能看到随本次请求提供的图像。"
            )
        return prompt

    def run(self, image_path: Path, scenario: str) -> Tuple[Dict[str, Any], AgentTrace]:
        started = time.time()
        prompt = self._build_prompt(scenario)
        images: List[Path] = [image_path]
        tool_calls: List[Dict[str, Any]] = []

        try:
            result = self.llm.analyze_images(prompt, images, task="perception")
        except Exception as err:  # noqa: BLE001
            result = {"scene_description": f"感知失败：{err}", "findings": [], "tool_requests": []}

        rounds = 0
        max_rounds = self.registry.max_rounds if self.registry else 0
        while self.registry is not None and rounds < max_rounds:
            requests = _normalize_requests(result.get("tool_requests"))
            if not requests:
                break
            observations, new_images, calls = self.registry.execute_all(requests, image_path)
            tool_calls.extend(calls)
            for img in new_images:
                if img not in images and len(images) < MAX_IMAGES_PER_CALL:
                    images.append(img)
            rounds += 1
            try:
                result = self.llm.analyze_images(
                    prompt + _tool_context(observations, rounds, max_rounds), images, task="perception"
                )
            except Exception as err:  # noqa: BLE001
                result = {"scene_description": f"感知失败：{err}", "findings": [], "tool_requests": []}
                break

        findings = result.get("findings") or []
        conflicts = []
        if not findings:
            conflicts.append("未提出任何候选风险（可能画面正常，也可能模型未识别到）")

        trace = AgentTrace(
            agent=self.name,
            input_summary=f"图片 {image_path.name}；场景 {scenario}",
            output_summary=f"场景描述完成，提出候选风险 {len(findings)} 个"
                           + (f"，自主工具调用 {len(tool_calls)} 次" if tool_calls else "，未申请工具"),
            tool_calls=tool_calls,
            conflicts=conflicts,
            duration_ms=int((time.time() - started) * 1000),
        )
        for item in findings:
            item.setdefault("source_image", image_path.name)
        return result, trace


# ------------------------------------------------------------------ 风险分析


class RiskAnalysisAgent:
    """风险分析 Agent：把候选风险 + 规则 + 历史案例 合成定级与处置建议。"""

    name = "风险分析 Agent"

    def __init__(
        self,
        llm: BaseLLM,
        rule_retriever: SafetyRuleRetriever,
        memory: CaseMemory,
        strategy: Dict[str, Any],
        use_rules: bool = True,
        use_memory: bool = True,
    ):
        self.llm = llm
        self.rule_retriever = rule_retriever
        self.memory = memory
        self.strategy = strategy
        self.use_rules = use_rules
        self.use_memory = use_memory

    def _collect_context(self, item: Dict[str, Any]) -> Dict[str, Any]:
        risk_type = item.get("risk_type", "unknown")
        evidence = item.get("evidence", "")

        rules: List[Dict[str, Any]] = []
        if self.use_rules:
            rules = self.rule_retriever.retrieve_for_finding(risk_type, evidence, top_k=2)

        cases: List[Dict[str, Any]] = []
        if self.use_memory:
            cases = self.memory.retrieve(risk_type, limit=2)

        primary = rules[0] if rules else {}
        return {
            "rules": rules,
            "cases": cases,
            "default_severity": primary.get("default_severity", "uncertain"),
            "default_recommendation": primary.get("recommendation", ""),
            "rule_source": primary.get("source", ""),
            "rule_ids": [r.get("rule_id") or r.get("risk_type") for r in rules],
            "case_ids": [c.get("case_id") for c in cases if c.get("case_id")],
        }

    def run(self, perception: Dict[str, Any]) -> Tuple[List[RiskFinding], AgentTrace]:
        started = time.time()
        candidates = perception.get("findings") or []
        conflicts: List[str] = []
        contexts: List[Dict[str, Any]] = []
        payload_candidates: List[Dict[str, Any]] = []

        for idx, item in enumerate(candidates):
            ctx = self._collect_context(item)
            contexts.append(ctx)
            payload_candidates.append({
                "index": idx,
                "risk_type": item.get("risk_type", "unknown"),
                "evidence": item.get("evidence", ""),
                "location": item.get("location", "unknown"),
                "candidate_severity": item.get("severity", "uncertain"),
                "confidence": item.get("confidence", 0.0),
                "uncertainty": item.get("uncertainty", ""),
                "default_severity": ctx["default_severity"],
                "default_recommendation": ctx["default_recommendation"],
                "rules": [{"title": r.get("title"), "description": r.get("description"),
                           "source": r.get("source")} for r in ctx["rules"]],
                "historical_cases": [{"note": c.get("note"), "corrected_to": c.get("correct_risk_type")}
                                     for c in ctx["cases"]],
            })

        llm_results = self._ask_llm(payload_candidates)
        if llm_results is None and payload_candidates:
            conflicts.append("风险分析 Agent 的模型调用失败，已回退为规则默认等级。")

        findings: List[RiskFinding] = []
        for idx, item in enumerate(candidates):
            ctx = contexts[idx]
            got = llm_results.get(idx, {}) if llm_results else {}
            severity = got.get("severity") or ctx["default_severity"]
            recommendation = got.get("recommendation") or ctx["default_recommendation"] \
                or "请人工复核现场情况，并按实验室安全规范处理。"
            uncertainty = got.get("uncertainty") or item.get("uncertainty", "")
            confidence = _safe_float(got.get("confidence"), _safe_float(item.get("confidence"), 0.0))

            if self.use_rules and not ctx["rules"]:
                uncertainty = (uncertainty + "；规则库中缺少对应规则，等级依据不足").strip("；")
                conflicts.append(f"{item.get('risk_type')} 未检索到明确规则，已标记为不确定。")
            if self.use_memory and ctx["cases"]:
                confidence = min(confidence + 0.05, 1.0)

            findings.append(RiskFinding(
                risk_type=item.get("risk_type", "unknown"),
                severity=severity,
                evidence=item.get("evidence", ""),
                location=item.get("location", "unknown"),
                time_range=item.get("time_range", "unknown"),
                confidence=round(confidence, 3),
                recommendation=recommendation,
                uncertainty=uncertainty,
                status="analyzed",
                source_image=item.get("source_image", ""),
                rule_ids=[rid for rid in ctx["rule_ids"] if rid],
                case_ids=ctx["case_ids"],
            ))

        trace = AgentTrace(
            agent=self.name,
            input_summary=f"候选风险 {len(candidates)} 个；规则检索={'开' if self.use_rules else '关'}；"
                          f"案例记忆={'开' if self.use_memory else '关'}",
            output_summary=f"完成定级 {len(findings)} 个",
            tool_calls=[
                {"tool": "rules.retrieve", "enabled": self.use_rules, "calls": len(candidates)},
                {"tool": "memory.retrieve", "enabled": self.use_memory, "calls": len(candidates)},
            ],
            conflicts=conflicts,
            duration_ms=int((time.time() - started) * 1000),
        )
        return findings, trace

    def _ask_llm(self, payload_candidates: List[Dict[str, Any]]) -> Optional[Dict[int, Dict[str, Any]]]:
        if not payload_candidates:
            return {}
        if not self.strategy.get("agents", {}).get("analysis", {}).get("use_llm", True):
            return None
        prompt_template = _prompt(self.strategy, "analysis", DEFAULT_ANALYSIS_PROMPT)
        user = _format_prompt(
            prompt_template, risk_levels=json.dumps(_risk_level_definition(), ensure_ascii=False)
        ) + "\n\n候选风险（JSON）：\n" + json.dumps(
            {"candidates": payload_candidates}, ensure_ascii=False, indent=2
        )
        try:
            data = self.llm.complete_json(ANALYSIS_SYSTEM, user)
        except Exception:  # noqa: BLE001
            return None
        return _index_results(data.get("results"))


# ------------------------------------------------------------------ 证据复核


class EvidenceReviewAgent:
    """证据复核 Agent：充当“反方”压制误报，并显式记录与风险分析 Agent 的意见冲突。"""

    name = "证据复核 Agent"

    def __init__(self, llm: BaseLLM, strategy: Dict[str, Any], enabled: bool = True):
        self.llm = llm
        self.strategy = strategy
        self.enabled = enabled
        self.min_confidence = float(strategy.get("review", {}).get("min_confidence", 0.45))
        self.min_evidence_len = int(strategy.get("review", {}).get("min_evidence_len", 8))

    def run(self, findings: List[RiskFinding]) -> Tuple[List[RiskFinding], AgentTrace]:
        started = time.time()
        conflicts: List[str] = []
        disagreements = 0

        if not self.enabled:
            reviewed = []
            for f in findings:
                item = RiskFinding.from_dict(f.to_dict())
                item.status = "simulated" if self.llm.is_simulated else "confirmed"
                item.analysis_severity = item.severity
                item.review_verdict = "skipped"
                reviewed.append(item)
            conflicts.append("消融实验：证据复核 Agent 已关闭，所有分析结论直接放行（预期误报上升）。")
            trace = AgentTrace(
                agent=self.name,
                input_summary=f"待复核风险 {len(findings)} 个",
                output_summary="（已关闭）全部直接放行",
                conflicts=conflicts,
                duration_ms=int((time.time() - started) * 1000),
            )
            return reviewed, trace

        verdicts = self._ask_llm(findings)

        reviewed: List[RiskFinding] = []
        for idx, finding in enumerate(findings):
            item = RiskFinding.from_dict(finding.to_dict())
            verdict = verdicts.get(idx) if verdicts else None

            if self.llm.is_simulated:
                item.status = "simulated"
                item.analysis_severity = item.severity
                item.review_verdict = "uncertain"
                item.uncertainty = _join(item.uncertainty, "模拟模式：结果不可作为真实识别结论")
                reviewed.append(item)
                continue

            if verdict is None:
                verdict = self._fallback_verdict(item)
                conflicts.append(f"{item.risk_type} 复核模型不可用，改用阈值规则复核。")

            analysis_view = item.severity
            item.analysis_severity = analysis_view
            v = str(verdict.get("verdict", "")).lower()
            reason = verdict.get("reason", "")
            item.review_verdict = v or "uncertain"

            # ---- 冲突检测与仲裁 ----
            if v == "rejected":
                item.status = "rejected"
                item.disagreement = (
                    f"风险分析 Agent 判定为 {analysis_view}，证据复核 Agent 认为属误报（{reason}）"
                )
                item.uncertainty = _join(item.uncertainty, f"复核判定为误报：{reason}")
                item.needs_human_review = True
                disagreements += 1
                conflicts.append(f"[冲突] {item.risk_type}：{item.disagreement} → 仲裁：采纳复核意见，标记为误报并交人工确认")
            elif v == "confirmed":
                item.status = "confirmed"
                if verdict.get("severity"):
                    item.severity = verdict["severity"]
                if analysis_view == "uncertain" and item.severity != "uncertain":
                    item.disagreement = (
                        f"风险分析 Agent 因证据不足判为 uncertain，证据复核 Agent 认为证据充分并定为 {item.severity}"
                    )
                    disagreements += 1
                    conflicts.append(f"[冲突] {item.risk_type}：{item.disagreement} → 仲裁：采纳复核意见，但需人工抽查")
                if reason:
                    item.uncertainty = _join(item.uncertainty, f"复核说明：{reason}")
            else:
                item.status = "uncertain"
                item.severity = "uncertain"
                item.uncertainty = _join(item.uncertainty, f"复核认为证据不足：{reason or '证据不充分'}")
                if analysis_view in ("medium", "high"):
                    item.disagreement = (
                        f"风险分析 Agent 定为 {analysis_view}，证据复核 Agent 认为证据不足"
                    )
                    item.needs_human_review = analysis_view == "high"
                    disagreements += 1
                    conflicts.append(
                        f"[冲突] {item.risk_type}：{item.disagreement} → 仲裁："
                        + ("按保守原则降级为不确定，并列入人工优先复核" if item.needs_human_review
                           else "按保守原则降级为不确定")
                    )

            reviewed.append(item)

        confirmed = sum(1 for f in reviewed if f.status == "confirmed")
        uncertain = sum(1 for f in reviewed if f.status == "uncertain")
        rejected = sum(1 for f in reviewed if f.status == "rejected")
        simulated = sum(1 for f in reviewed if f.status == "simulated")
        parts = [f"确认 {confirmed} 个", f"不确定 {uncertain} 个", f"驳回误报 {rejected} 个"]
        if simulated:
            parts.append(f"模拟标记 {simulated} 个")
        parts.append(f"与风险分析 Agent 存在 {disagreements} 处分歧" if disagreements else "无意见分歧")
        trace = AgentTrace(
            agent=self.name,
            input_summary=f"待复核风险 {len(findings)} 个",
            output_summary="，".join(parts),
            conflicts=conflicts,
            duration_ms=int((time.time() - started) * 1000),
        )
        return reviewed, trace

    def _fallback_verdict(self, item: RiskFinding) -> Dict[str, Any]:
        if item.confidence < self.min_confidence:
            return {"verdict": "uncertain", "reason": f"置信度 {item.confidence:.2f} 低于阈值 {self.min_confidence}"}
        if len(item.evidence.strip()) < self.min_evidence_len:
            return {"verdict": "uncertain", "reason": "证据描述过短，不足以支撑结论"}
        return {"verdict": "confirmed", "reason": "通过阈值复核"}

    def _ask_llm(self, findings: List[RiskFinding]) -> Optional[Dict[int, Dict[str, Any]]]:
        if not findings:
            return {}
        if not self.strategy.get("agents", {}).get("review", {}).get("use_llm", True):
            return None

        payload = [{
            "index": i,
            "risk_type": f.risk_type,
            "proposed_severity": f.severity,
            "evidence": f.evidence,
            "location": f.location,
            "confidence": f.confidence,
            "recommendation": f.recommendation,
        } for i, f in enumerate(findings)]

        user = _prompt(self.strategy, "review", DEFAULT_REVIEW_PROMPT) + "\n\n待复核结论（JSON）：\n" + json.dumps(
            {"findings": payload}, ensure_ascii=False, indent=2
        )
        try:
            data = self.llm.complete_json(REVIEW_SYSTEM, user)
        except Exception:  # noqa: BLE001
            return None
        return _index_results(data.get("results"))


# ------------------------------------------------------------------ 基础对照


class BaselineAnalyzer:
    """基础识别系统（对照组）。

    单次模型调用直接给出结论：不拆角色、不检索规则、不做证据复核、不调工具。
    它的存在是为了在实验章节里回答“多 Agent 到底有没有用”。
    """

    name = "基础识别系统（单模型直出）"

    def __init__(self, llm: BaseLLM, strategy: Dict[str, Any]):
        self.llm = llm
        self.strategy = strategy

    def run(self, image_path: Path, scenario: str) -> Tuple[List[RiskFinding], AgentTrace]:
        started = time.time()
        prompt = _format_prompt(
            _prompt(self.strategy, "baseline", DEFAULT_BASELINE_PROMPT),
            scenario=scenario,
            risk_types=json.dumps(self.strategy.get("risk_types", []), ensure_ascii=False),
        )
        try:
            raw = self.llm.analyze_images(prompt, [image_path], task="baseline")
        except Exception as err:  # noqa: BLE001
            raw = {"findings": []}
            note = f"基础系统调用失败：{err}"
        else:
            note = ""

        findings: List[RiskFinding] = []
        for item in raw.get("findings", []) or []:
            findings.append(RiskFinding(
                risk_type=item.get("risk_type", "unknown"),
                severity=item.get("severity", "uncertain"),
                evidence=item.get("evidence", ""),
                location=item.get("location", "unknown"),
                time_range=item.get("time_range", "unknown"),
                confidence=_safe_float(item.get("confidence"), 0.5),
                recommendation=item.get("recommendation", ""),
                uncertainty=item.get("uncertainty", ""),
                status="simulated" if self.llm.is_simulated else "confirmed",
                source_image=image_path.name,
            ))

        trace = AgentTrace(
            agent=self.name,
            input_summary=f"图片 {image_path.name}；场景 {scenario}",
            output_summary=f"单次调用直接输出风险 {len(findings)} 个（无规则、无复核、无工具）",
            conflicts=[note] if note else [],
            duration_ms=int((time.time() - started) * 1000),
        )
        return findings, trace


# ---------------- 内部小工具 ----------------


def _index_results(results: Any) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    if not isinstance(results, list):
        return indexed
    for pos, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", pos))
        except (TypeError, ValueError):
            idx = pos
        indexed[idx] = item
    return indexed


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _join(a: str, b: str) -> str:
    parts = [p for p in [str(a).strip(), str(b).strip()] if p]
    return "；".join(parts)


def _risk_level_definition() -> Dict[str, str]:
    return {
        "high": "可能立即导致伤害或事故，例如加热设备旁有易燃溶剂、强酸试剂敞口且靠近人员手部",
        "medium": "明显违规、可能发展为事故，例如未戴护目镜进行液体操作、试剂瓶标签缺失",
        "low": "不规范但短期风险较低，例如水杯放在实验台边缘",
        "uncertain": "证据不足，不能可靠判断，例如手部被遮挡无法确认是否戴手套",
    }
