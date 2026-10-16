"""Agent 可调用的工具层：规则检索、案例记忆、工具日志、视频抽帧、局部裁剪。"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _bigrams(text: str) -> set:
    """中文没有空格分词，用字符二元组做粗粒度相似度。"""
    clean = "".join(ch for ch in text.lower() if not ch.isspace())
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


class ToolLogger:
    """记录每一次工具调用，供报告中“工具调用日志”章节使用。"""

    def __init__(self, log_path: Path, session_id: Optional[str] = None):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.calls: List[Dict[str, Any]] = []

    def record(self, tool_name: str, args: Dict[str, Any], result_summary: str) -> Dict[str, Any]:
        item = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "tool": tool_name,
            "args": args,
            "result_summary": result_summary,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.calls.append(item)
        return item


class SafetyRuleRetriever:
    """安全规则检索。针对中文做了字符级相似度补强。"""

    def __init__(self, rules_path: Path):
        self.rules_path = rules_path
        self.rules: List[Dict[str, Any]] = json.loads(rules_path.read_text(encoding="utf-8"))
        self._by_type = {r.get("risk_type"): r for r in self.rules}

    def get_by_type(self, risk_type: str) -> Optional[Dict[str, Any]]:
        return self._by_type.get(risk_type)

    def _score(self, query: str, rule: Dict[str, Any]) -> float:
        q = query.lower()
        score = 0.0

        risk_type = (rule.get("risk_type") or "").lower()
        if risk_type and risk_type in q:
            score += 10.0

        for kw in rule.get("keywords", []):
            if kw and kw.lower() in q:
                score += 3.0

        for alias in rule.get("aliases", []):
            if alias and alias.lower() in q:
                score += 4.0

        text = " ".join([
            rule.get("title", ""),
            rule.get("description", ""),
            rule.get("scope", ""),
        ])
        q_bigrams = _bigrams(q)
        if q_bigrams:
            overlap = _bigrams(text) & q_bigrams
            score += 3.0 * len(overlap) / len(q_bigrams)
        return score

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.8) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rule in self.rules:
            s = self._score(query, rule)
            if s >= min_score:
                scored.append((s, rule))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in scored[:top_k]]

    def retrieve_for_finding(self, risk_type: str, evidence: str = "", top_k: int = 2) -> List[Dict[str, Any]]:
        """优先按 risk_type 直配，命中不足再用描述文本补充。"""
        exact = self.get_by_type(risk_type)
        results = [exact] if exact else []
        if len(results) < top_k:
            for rule in self.retrieve(f"{risk_type} {evidence}", top_k=top_k):
                if rule not in results:
                    results.append(rule)
                if len(results) >= top_k:
                    break
        return results


class CaseMemory:
    """人工纠错案例库。检索结果会作为风险分析 Agent 的上下文。"""

    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        if not memory_path.exists():
            memory_path.write_text("[]", encoding="utf-8")

    def load(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def add_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        case.setdefault("case_id", f"case_{int(time.time() * 1000)}")
        case.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        cases = self.load()
        cases.append(case)
        self.memory_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return case

    def retrieve(self, risk_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        matched = [
            c for c in self.load()
            if c.get("correct_risk_type") == risk_type or c.get("original_risk_type") == risk_type
        ]
        return matched[-limit:]

    def stats(self) -> Dict[str, Any]:
        """给自进化模块用：统计哪些风险类型最常被纠错。"""
        from collections import Counter

        cases = self.load()
        confusions = Counter()
        by_type = Counter()
        for c in cases:
            orig = c.get("original_risk_type") or "unknown"
            corrected = c.get("correct_risk_type") or "unknown"
            by_type[orig] += 1
            if corrected != "unknown":
                confusions[f"{orig} -> {corrected}"] += 1
        return {
            "total": len(cases),
            "by_original_type": dict(by_type),
            "confusions": dict(confusions),
        }


class VideoFrameExtractor:
    """视频抽帧工具。按时间轴均匀采样，避免只取到开头几帧。"""

    def __init__(self, max_frames: int = 4):
        self.max_frames = max_frames

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except Exception:
            return False

    def extract(self, video_path: Path, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore
        except Exception:
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                positions = list(range(self.max_frames))
            else:
                positions = [
                    min(int(frame_count * (i + 0.5) / self.max_frames), max(frame_count - 1, 0))
                    for i in range(self.max_frames)
                ]

            saved: List[Path] = []
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok:
                    continue
                out = output_dir / f"{video_path.stem}_frame_{len(saved) + 1:02d}.jpg"
                if cv2.imwrite(str(out), frame):
                    saved.append(out)
            return saved
        finally:
            cap.release()


class ImageCropper:
    """局部裁剪工具。用于对可疑区域放大后交给大模型细看。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        try:
            from PIL import Image  # noqa: F401
            return True
        except Exception:
            return False

    def crop(self, image_path: Path, box: Tuple[float, float, float, float], tag: str = "crop") -> Optional[Path]:
        """box 为相对坐标 (left, top, right, bottom)，取值 0~1。"""
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None

        with Image.open(image_path) as img:
            w, h = img.size
            left, top, right, bottom = box
            region = img.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))
            out = self.output_dir / f"{image_path.stem}_{tag}.jpg"
            region.save(out, "JPEG", quality=90)
            return out
