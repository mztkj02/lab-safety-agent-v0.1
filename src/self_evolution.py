"""自进化模块（人工确认式）。

完整闭环：
    人工纠错 -> 案例入库 -> 聚合分析 -> 策略进化 Agent 提出修改方案
    -> 生成策略草案 -> 组员确认晋升 -> 效果复测 -> 必要时回退

进化方案有三种载体（对应课题要求的“提示词、规则或案例记忆”）：
    1. 提示词补丁（prompt_patch）——改感知提示词
    2. 易混淆说明（risk_type_notes）——把误判对变成显式约束
    3. 新增规则（proposed_rules）——晋升时写入规则库

与旧版的区别：
- 旧版每次纠错都追加同一句固定话术，版本号一律 v0.2 会互相覆盖。
- 新版由 Agent 基于案例聚合结果提出具体方案，版本号按 versions/ 目录递增，
  且 change_log 里保留“为什么这么改”的证据。
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION_RE = re.compile(r"^strategy_v(\d+)\.(\d+)\.json$")

EVOLUTION_SYSTEM = (
    "你是实验室安全系统的策略进化专家。你的任务是根据人工纠错记录，"
    "提出对系统提示词、规则库或案例说明的具体修改方案，使系统不再犯同类错误。"
    "要求修改必须具体、可执行、可验证；不要输出空泛建议。只输出 JSON。"
)

EVOLUTION_PROMPT = """下面是最近的人工纠错聚合结果：

{cases_json}

当前系统已定义的风险类型：
{risk_types}

当前已有的易混淆说明：
{existing_notes}

请提出具体的改进方案，输出 JSON，字段如下：
{{
  "diagnosis": "用 2~3 句话说明这批纠错暴露出的系统性问题",
  "risk_type_notes": {{"误判对或风险类型": "要写进提示词的易混淆说明"}},
  "prompt_patch": "追加到感知提示词末尾的补充要求（3~6 条，用换行分隔，每条以 - 开头）",
  "proposed_rules": [
    {{
      "rule_id": "R-XXX-01",
      "risk_type": "必须是上面已定义的风险类型之一",
      "title": "规则标题",
      "description": "判定依据",
      "keywords": ["关键词"],
      "default_severity": "low|medium|high",
      "recommendation": "处理建议",
      "source": "依据来源；如不确定请写“待补充权威来源”"
    }}
  ]
}}

约束：
- 只针对确实存在重复误判的问题提方案；没有新问题的字段留空（空对象 / 空字符串 / 空数组）。
- proposed_rules 最多 2 条，且必须能通过数据判断，不要写抽象原则。
- 不要编造国家标准编号，来源不确定就写“待补充权威来源”。
"""


# ---------------------------------------------------------------- 案例记录


def record_feedback(
    project_root: Path,
    report_path: Path,
    original_risk_type: str,
    correct_risk_type: Optional[str],
    correct_severity: Optional[str],
    note: str,
    reviewer: str = "",
) -> Path:
    """把一次人工纠错写入 memory/cases.json。"""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = next(
        (f for f in report.get("findings", []) if f.get("risk_type") == original_risk_type), {}
    )

    case = {
        "case_id": f"case_{int(time.time() * 1000)}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_sample_id": report.get("sample_id"),
        "input_path": report.get("input_path"),
        "strategy_version": report.get("strategy_version"),
        "original_risk_type": original_risk_type,
        "original_severity": target.get("severity"),
        "original_status": target.get("status"),
        "original_confidence": target.get("confidence"),
        "correct_risk_type": correct_risk_type,
        "correct_severity": correct_severity,
        "note": note,
        "reviewer": reviewer,
        "original_findings": report.get("findings", []),
    }

    memory_path = project_root / "memory" / "cases.json"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else []
    cases.append(case)
    memory_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return memory_path


# ---------------------------------------------------------------- 聚合分析


def analyze_cases(project_root: Path) -> Dict[str, Any]:
    """把散落的纠错案例聚合成可执行的进化依据。"""
    memory_path = project_root / "memory" / "cases.json"
    cases: List[Dict[str, Any]] = []
    if memory_path.exists():
        try:
            cases = json.loads(memory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cases = []

    confusions: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    severity_fixes: List[Dict[str, Any]] = []
    false_positives = 0

    for case in cases:
        orig = case.get("original_risk_type") or "unknown"
        corrected = case.get("correct_risk_type")
        if corrected and corrected != orig:
            confusions.setdefault((orig, corrected), []).append(case)
            if corrected == "normal":
                false_positives += 1
        if case.get("correct_severity") and case.get("original_severity") \
                and case["correct_severity"] != case["original_severity"]:
            severity_fixes.append(case)

    rows = []
    for (orig, corrected), items in confusions.items():
        rows.append({
            "original": orig,
            "corrected": corrected,
            "count": len(items),
            "latest_note": items[-1].get("note", ""),
        })
    rows.sort(key=lambda r: r["count"], reverse=True)

    return {
        "total_cases": len(cases),
        "confusion_pairs": rows,
        "false_positive_count": false_positives,
        "severity_fix_count": len(severity_fixes),
    }


# ---------------------------------------------------------------- 进化 Agent


class EvolutionAgent:
    """策略进化 Agent：根据纠错案例提出具体的提示词 / 规则修改方案。"""

    name = "策略进化 Agent"

    def __init__(self, strategy: Dict[str, Any], llm: Any = None):
        self.strategy = strategy
        self.llm = llm

    def propose(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is not None and not getattr(self.llm, "is_simulated", True):
            proposal = self._propose_with_llm(analysis)
            if proposal:
                proposal["source"] = "llm_agent"
                return proposal
            fallback = self._propose_rule_based(analysis)
            fallback["source"] = "rule_based_fallback（模型提案失败，已回退）"
            return fallback
        fallback = self._propose_rule_based(analysis)
        fallback["source"] = "rule_based_fallback（模拟模式未调用模型）"
        return fallback

    # -------------------------------------------------- 模型提案
    def _propose_with_llm(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user = EVOLUTION_PROMPT.format(
            cases_json=json.dumps(analysis, ensure_ascii=False, indent=2),
            risk_types=json.dumps(self.strategy.get("risk_types", []), ensure_ascii=False),
            existing_notes=json.dumps(self.strategy.get("risk_type_notes", {}), ensure_ascii=False),
        )
        try:
            data = self.llm.complete_json(EVOLUTION_SYSTEM, user)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict):
            return None

        valid_types = set(self.strategy.get("risk_types", []))
        rules = [
            r for r in (data.get("proposed_rules") or [])
            if isinstance(r, dict) and r.get("risk_type") in valid_types
        ]
        return {
            "diagnosis": str(data.get("diagnosis") or ""),
            "risk_type_notes": {
                str(k): str(v) for k, v in (data.get("risk_type_notes") or {}).items()
            },
            "prompt_patch": str(data.get("prompt_patch") or ""),
            "proposed_rules": rules[:2],
        }

    # -------------------------------------------------- 确定性回退
    def _propose_rule_based(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        notes: Dict[str, str] = {}
        for row in analysis.get("confusion_pairs", []):
            orig, corrected, count = row["original"], row["corrected"], row["count"]
            key = f"{orig}↔{corrected}"
            basis = _clean_sentence(row.get("latest_note", ""))
            if corrected == "normal":
                notes[key] = (
                    f"已有 {count} 次把「{orig}」误报为风险（实际正常）。依据：{basis} "
                    f"证据不足时应输出 uncertain 而不是硬报风险。"
                )
            else:
                notes[key] = (
                    f"已有 {count} 次把「{orig}」误判为「{corrected}」，二者需明确区分。依据：{basis}"
                )

        diagnosis = (
            f"累计 {analysis.get('total_cases', 0)} 条纠错，其中误报 "
            f"{analysis.get('false_positive_count', 0)} 条、等级纠正 {analysis.get('severity_fix_count', 0)} 条，"
            f"识别出 {len(analysis.get('confusion_pairs', []))} 组重复误判模式。"
        )
        patch = (
            "- 逐条检查历史纠错中出现的易混淆说明，避免重复同类误判。\n"
            "- 画面存在遮挡、模糊、角度受限时，必须输出 severity=uncertain 并说明原因。\n"
            "- 证据描述必须具体到可见物体与位置，禁止使用“可能存在”这类无信息量的表述。"
        )
        return {"diagnosis": diagnosis, "risk_type_notes": notes,
                "prompt_patch": patch, "proposed_rules": []}


# ---------------------------------------------------------------- 版本管理


def list_versions(project_root: Path) -> List[str]:
    versions_dir = project_root / "versions"
    found = set()
    if versions_dir.exists():
        for path in versions_dir.glob("strategy_v*.json"):
            m = VERSION_RE.match(path.name)
            if m:
                found.add(f"v{m.group(1)}.{m.group(2)}")
    current = project_root / "configs" / "strategy_current.json"
    if current.exists():
        try:
            found.add(json.loads(current.read_text(encoding="utf-8")).get("version", ""))
        except json.JSONDecodeError:
            pass
    return sorted((v for v in found if v), key=_version_key)


def _version_key(version: str) -> Tuple[int, int]:
    m = re.match(r"v?(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _clean_sentence(text: str) -> str:
    """去掉句尾多余标点，避免拼接出“。。 ”。"""
    return str(text or "").strip().rstrip("。；;，, ")


def next_version(project_root: Path) -> str:
    existing = list_versions(project_root)
    if not existing:
        return "v0.2"
    major, minor = _version_key(existing[-1])
    return f"v{major}.{minor + 1}"


# ---------------------------------------------------------------- 策略生成


def propose_strategy_version(
    project_root: Path,
    current_strategy_path: Path,
    proposal_note: str,
    llm: Any = None,
) -> Path:
    """由策略进化 Agent 提案，生成一份有实质差异的新策略草案。"""
    strategy = json.loads(current_strategy_path.read_text(encoding="utf-8"))
    from_version = strategy.get("version", "v0.1")
    new_version = next_version(project_root)

    analysis = analyze_cases(project_root)
    agent = EvolutionAgent(strategy, llm=llm)
    proposal = agent.propose(analysis)

    changes: List[str] = []

    # 1) 易混淆说明（提示词上下文）
    notes: Dict[str, str] = dict(strategy.get("risk_type_notes") or {})
    added = 0
    for key, text in (proposal.get("risk_type_notes") or {}).items():
        if key not in notes:
            notes[key] = text
            added += 1
            changes.append(f"新增易混淆说明：{key}")
    strategy["risk_type_notes"] = notes

    # 2) 提示词补丁：替换上一轮的进化块，避免提示词无限膨胀
    patch = (proposal.get("prompt_patch") or "").strip()
    header = f"【策略进化补充要求 {new_version}】"
    perception_prompt = strategy["prompts"]["perception"]
    marker = "【策略进化补充要求"
    if marker in perception_prompt:
        perception_prompt = perception_prompt[: perception_prompt.index(marker)].rstrip()
    if patch:
        strategy["prompts"]["perception"] = perception_prompt.rstrip() + f"\n\n{header}\n{patch}"
        changes.append("更新感知提示词：写入本轮的补充要求")
    else:
        strategy["prompts"]["perception"] = perception_prompt

    # 3) 新增规则：先存进草案，晋升时才真正写进规则库
    proposed_rules = proposal.get("proposed_rules") or []
    if proposed_rules:
        strategy["proposed_rules"] = proposed_rules
        changes.append(f"提出 {len(proposed_rules)} 条新规则，将在晋升时写入规则库")

    if not changes:
        changes.append("本轮未发现新的重复误判模式，仅更新案例记忆")

    strategy["version"] = new_version
    strategy.setdefault("change_log", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "from": from_version,
        "to": new_version,
        "proposal": proposal_note,
        "proposal_source": proposal.get("source", "unknown"),
        "diagnosis": proposal.get("diagnosis", ""),
        "changes": changes,
        "evidence": {
            "total_cases": analysis["total_cases"],
            "confusion_pairs": analysis["confusion_pairs"],
            "false_positive_count": analysis["false_positive_count"],
        },
        "status": "pending_human_confirmation",
    })

    versions_dir = project_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    out = versions_dir / f"strategy_{new_version}.json"
    out.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def promote_strategy(project_root: Path, version: str) -> Path:
    """把某个已确认的草案晋升为当前生效策略，并把新规则写入规则库。"""
    source = project_root / "versions" / f"strategy_{version}.json"
    if not source.exists():
        raise FileNotFoundError(f"找不到策略版本: {version}")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["version"] = version

    applied_rules = _merge_rules_into_library(project_root, data.get("proposed_rules") or [])

    for entry in data.get("change_log", []):
        if entry.get("to") == version:
            entry["status"] = "active"
            if applied_rules:
                entry.setdefault("changes", []).append(
                    f"已把 {len(applied_rules)} 条新规则写入规则库：{'、'.join(applied_rules)}"
                )

    target = project_root / "configs" / "strategy_current.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _merge_rules_into_library(project_root: Path, new_rules: List[Dict[str, Any]]) -> List[str]:
    if not new_rules:
        return []
    rules_path = project_root / "rules" / "lab_safety_rules.json"
    existing = json.loads(rules_path.read_text(encoding="utf-8"))
    known_ids = {r.get("rule_id") for r in existing}
    applied = []
    for rule in new_rules:
        rid = rule.get("rule_id")
        if not rid or rid in known_ids:
            continue
        existing.append({**rule, "origin": "自进化新增"})
        known_ids.add(rid)
        applied.append(rid)
    if applied:
        rules_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return applied


def rollback_strategy(project_root: Path, version: str) -> Path:
    """回退到历史版本。"""
    return promote_strategy(project_root, version)
