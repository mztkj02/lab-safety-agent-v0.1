"""工具注册表：Agent 可自主选择的工具集合。

与旧版的区别：
- 旧版是流水线写死调用顺序（先抽帧、再检索规则、再检索案例），Agent 没有选择权。
- 新版把工具暴露成清单交给 Agent，由 Agent 在分析过程中主动提出 tool_requests，
  系统执行后把观察结果回灌给 Agent，形成“观察 -> 决策 -> 调工具 -> 再观察”的闭环。
  所有调用都会写入 reports/tool_calls.jsonl 并计入报告的调用成本。

可用工具：
    crop_region    局部裁剪放大，用于细看可疑区域
    detect_objects 目标检测（需要 ultralytics，可选）
    search_rules   安全规则检索
    search_cases   历史纠错案例检索
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools import CaseMemory, ImageCropper, SafetyRuleRetriever, ToolLogger

#: 给模型看的工具清单
TOOL_MANIFEST = """可用工具（需要时通过 tool_requests 主动申请调用，一次可申请多个）：
1. crop_region(left, top, right, bottom, tag)
   裁剪并放大图像中的某个区域后再仔细看。坐标是 0~1 的相对值，(0,0) 在左上角。
   适用场景：某个区域太小、太远、或细节看不清，需要放大确认。
   例：{"tool": "crop_region", "args": {"left": 0.1, "top": 0.3, "right": 0.5, "bottom": 0.9, "tag": "hands"}}
2. detect_objects()
   对当前图像做目标检测，返回检测到的物体类别、置信度与位置框。
   适用场景：需要确认画面里到底有没有某个具体物体。
3. search_rules(query)
   检索实验室安全规则库。适用场景：不确定某行为是否违规、需要找判定依据。
4. search_cases(risk_type)
   检索历史人工纠错案例。适用场景：想知道同类风险过去是怎么被纠正的。

如果当前信息已经足够，就不要申请工具，直接把 tool_requests 置为空数组。
最多允许 {max_rounds} 轮工具调用。"""


@dataclass
class ToolResult:
    tool: str
    args: Dict[str, Any]
    ok: bool
    observation: str
    summary: str
    new_images: List[Path] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(
        self,
        project_root: Path,
        logger: ToolLogger,
        rules: SafetyRuleRetriever,
        memory: CaseMemory,
        work_dir: Path,
        max_rounds: int = 2,
        enabled: Optional[Dict[str, bool]] = None,
    ):
        self.project_root = project_root
        self.logger = logger
        self.rules = rules
        self.memory = memory
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self.cropper = ImageCropper(work_dir / "crops")
        #: 允许用配置关闭个别工具（用于消融实验）
        self.enabled = {
            "crop_region": True,
            "detect_objects": True,
            "search_rules": True,
            "search_cases": True,
            **(enabled or {}),
        }

    # ------------------------------------------------------------ 对外

    def manifest(self) -> str:
        # 清单里带 JSON 示例，用定向替换而不是 str.format，避免花括号被当成占位符
        lines = [TOOL_MANIFEST.replace("{max_rounds}", str(self.max_rounds))]
        disabled = [name for name, on in self.enabled.items() if not on]
        if disabled:
            lines.append(f"\n注意：以下工具本次不可用：{'、'.join(disabled)}。")
        return "\n".join(lines)

    def execute(self, request: Dict[str, Any], current_image: Path) -> ToolResult:
        name = str(request.get("tool") or request.get("name") or "").strip()
        args = request.get("args") or {}

        if not self.enabled.get(name, False):
            return self._finish(ToolResult(
                tool=name or "unknown", args=args, ok=False,
                observation=f"工具 {name} 当前不可用，请改用其他方式判断。",
                summary="工具被禁用",
            ))

        handler = {
            "crop_region": self._crop_region,
            "detect_objects": self._detect_objects,
            "search_rules": self._search_rules,
            "search_cases": self._search_cases,
        }.get(name)

        if handler is None:
            return self._finish(ToolResult(
                tool=name or "unknown", args=args, ok=False,
                observation=f"不存在名为 {name} 的工具。可用工具见工具清单。",
                summary="未知工具",
            ))

        try:
            return self._finish(handler(args, current_image))
        except Exception as err:  # noqa: BLE001
            return self._finish(ToolResult(
                tool=name, args=args, ok=False,
                observation=f"工具 {name} 执行失败：{err}",
                summary=f"执行失败：{err}",
            ))

    def execute_all(
        self, requests: List[Dict[str, Any]], current_image: Path
    ) -> Tuple[str, List[Path], List[Dict[str, Any]]]:
        """执行一批工具申请，返回（回灌给模型的观察文本, 新增图片, 调用记录）。"""
        observations: List[str] = []
        new_images: List[Path] = []
        calls: List[Dict[str, Any]] = []
        for req in requests[:4]:  # 单轮最多 4 个，防止模型刷调用
            result = self.execute(req, current_image)
            observations.append(f"[{result.tool}] {result.observation}")
            new_images.extend(result.new_images)
            calls.append({
                "tool": result.tool,
                "args": result.args,
                "ok": result.ok,
                "summary": result.summary,
            })
        return "\n".join(observations), new_images, calls

    # ------------------------------------------------------------ 具体工具

    def _crop_region(self, args: Dict[str, Any], current_image: Path) -> ToolResult:
        if not self.cropper.available():
            return ToolResult(
                tool="crop_region", args=args, ok=False,
                observation="局部裁剪工具不可用（缺少 Pillow），请直接基于原图判断。",
                summary="Pillow 未安装",
            )

        box = (
            _clamp(args.get("left", 0.0)),
            _clamp(args.get("top", 0.0)),
            _clamp(args.get("right", 1.0)),
            _clamp(args.get("bottom", 1.0)),
        )
        if box[2] - box[0] < 0.02 or box[3] - box[1] < 0.02:
            return ToolResult(
                tool="crop_region", args=args, ok=False,
                observation="裁剪区域过小，请给出至少占画面 2% 的区域。",
                summary="裁剪区域过小",
            )

        tag = str(args.get("tag") or "region")[:24]
        out = self.cropper.crop(current_image, box, tag=tag)
        if out is None:
            return ToolResult(
                tool="crop_region", args=args, ok=False,
                observation="裁剪失败，请基于原图判断。",
                summary="裁剪失败",
            )
        return ToolResult(
            tool="crop_region", args=args, ok=True,
            observation=(
                f"已裁剪区域 left={box[0]:.2f} top={box[1]:.2f} right={box[2]:.2f} bottom={box[3]:.2f}，"
                f"放大后的局部图已附在本次请求中，请基于它重新判断该区域。"
            ),
            summary=f"裁剪 {tag} 区域并放大",
            new_images=[out],
            raw={"box": box, "path": str(out)},
        )

    def _detect_objects(self, args: Dict[str, Any], current_image: Path) -> ToolResult:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            return ToolResult(
                tool="detect_objects", args=args, ok=False,
                observation="目标检测工具当前不可用（未安装 ultralytics），请改用视觉理解或局部裁剪。",
                summary="ultralytics 未安装",
            )

        weights = os.getenv("SAFETY_YOLO_WEIGHTS", "yolov8n.pt")
        try:
            model = YOLO(weights)
            results = model.predict(str(current_image), verbose=False)
        except Exception as err:  # noqa: BLE001
            return ToolResult(
                tool="detect_objects", args=args, ok=False,
                observation=f"目标检测执行失败：{err}",
                summary=f"检测失败：{err}",
            )

        detections = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            for box in getattr(result, "boxes", []) or []:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                detections.append({
                    "label": names.get(cls_id, str(cls_id)),
                    "confidence": round(float(box.conf[0]), 3) if box.conf is not None else None,
                })

        if not detections:
            text = "目标检测未发现任何已标注类别的物体。"
        else:
            text = "目标检测结果：" + json.dumps(detections[:15], ensure_ascii=False)
        return ToolResult(
            tool="detect_objects", args=args, ok=True,
            observation=text,
            summary=f"检出 {len(detections)} 个目标",
            raw={"detections": detections, "weights": weights},
        )

    def _search_rules(self, args: Dict[str, Any], current_image: Path) -> ToolResult:
        query = str(args.get("query") or "")
        if not query:
            return ToolResult(
                tool="search_rules", args=args, ok=False,
                observation="缺少 query 参数，请给出要查询的风险类型或行为的描述。",
                summary="缺少 query",
            )
        hits = self.rules.retrieve(query, top_k=3)
        if not hits:
            return ToolResult(
                tool="search_rules", args=args, ok=True,
                observation=f"未检索到与「{query}」相关的规则，该情形缺少明确依据。",
                summary="未命中规则",
            )
        text = "；".join(
            f"{r.get('rule_id')} {r.get('title')}（默认等级 {r.get('default_severity')}）" for r in hits
        )
        return ToolResult(
            tool="search_rules", args=args, ok=True,
            observation=f"检索到规则：{text}",
            summary=f"命中 {len(hits)} 条规则",
            raw={"rule_ids": [r.get("rule_id") for r in hits]},
        )

    def _search_cases(self, args: Dict[str, Any], current_image: Path) -> ToolResult:
        risk_type = str(args.get("risk_type") or "")
        if not risk_type:
            return ToolResult(
                tool="search_cases", args=args, ok=False,
                observation="缺少 risk_type 参数。",
                summary="缺少 risk_type",
            )
        cases = self.memory.retrieve(risk_type, limit=3)
        if not cases:
            return ToolResult(
                tool="search_cases", args=args, ok=True,
                observation=f"案例库中暂无 {risk_type} 的历史纠错记录。",
                summary="无历史案例",
            )
        text = "；".join(
            f"{c.get('original_risk_type')}→{c.get('correct_risk_type') or '未改'}（{c.get('note', '')[:40]}）"
            for c in cases
        )
        return ToolResult(
            tool="search_cases", args=args, ok=True,
            observation=f"历史纠错案例：{text}",
            summary=f"命中 {len(cases)} 条案例",
            raw={"case_ids": [c.get("case_id") for c in cases]},
        )

    # ------------------------------------------------------------ 日志

    def _finish(self, result: ToolResult) -> ToolResult:
        self.logger.record(
            f"agent_tool.{result.tool}",
            {"args": result.args, "ok": result.ok},
            result.summary,
        )
        return result


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low
