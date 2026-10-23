"""大模型客户端。

两种实现：
- MockLLM：无 API 时的确定性模拟，用于跑通流程与课堂演示。
  它的输出一律标记为 simulated，绝不会冒充真实识别结论。
- OpenAICompatibleVisionLLM：任意 OpenAI 兼容的视觉模型接口。

任务类型（task）：
- "perception"：图像 -> 场景描述 + 候选风险
- "review"    ：图像 + 单条风险 -> 证据是否支持该结论
- 文本任务（analysis / review_text）走 complete_json()
"""

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balanced_objects(text: str) -> List[str]:
    """扫描出文本中所有括号配平的顶层 JSON 对象片段（按出现顺序）。"""
    spans: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : i + 1])
                    start = -1
    return spans


def _try_load(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def extract_json(text: str) -> Dict[str, Any]:
    """从模型返回里尽力抽出 JSON 对象。

    模型常见毛病：包在 ```json 围栏里、前后带解释性文字、结尾多逗号。
    另外，当我们自己拼的提示词里也带有 JSON 示例时（模拟模式需要回读提示词），
    真正的载荷通常在最末尾，所以候选顺序是：围栏块（取最后一个）→ 整段文本 → 配平对象（倒序）。
    """
    if not text:
        raise ValueError("模型返回为空")

    candidates: List[str] = [m.group(1) for m in JSON_BLOCK_RE.finditer(text)]
    candidates.append(text)
    candidates.extend(reversed(_balanced_objects(text)))

    for raw in candidates:
        data = _try_load(raw)
        if data is not None:
            return data
    raise ValueError(f"无法从模型返回中解析 JSON：{text[:200]}")


#: 文件名关键词 -> 模拟识别结果。仅用于演示流程，不代表真实识别。
_MOCK_NAME_RULES = [
    (["no_goggle", "noglasses", "noglass", "ppe", "no_ppe", "goggle"], {
        "risk_type": "ppe_missing", "severity": "medium", "location": "画面中央",
        "evidence": "模拟：文件名提示该样本涉及个人防护装备缺失场景。",
    }),
    (["open_bottle", "reagent", "bottle", "label"], {
        "risk_type": "reagent_management", "severity": "medium", "location": "实验台区域",
        "evidence": "模拟：文件名提示该样本涉及试剂管理异常场景。",
    }),
    (["flammable", "heater", "alcohol", "fire", "hotplate"], {
        "risk_type": "flammable_near_heat", "severity": "high", "location": "加热设备附近",
        "evidence": "模拟：文件名提示该样本涉及热源附近易燃物场景。",
    }),
    (["drink", "cup", "food", "water", "beverage"], {
        "risk_type": "food_or_drink", "severity": "low", "location": "实验台边缘",
        "evidence": "模拟：文件名提示该样本涉及实验区饮食场景。",
    }),
    (["spill", "leak", "obstruct", "wire", "cable", "aisle"], {
        "risk_type": "spill_or_obstruction", "severity": "medium", "location": "地面/通道",
        "evidence": "模拟：文件名提示该样本涉及洒漏或通道阻塞场景。",
    }),
]

_MOCK_DISCLAIMER = "模拟模式输出：未接入真实视觉模型，不能作为实际识别证据。"


class BaseLLM:
    mode = "base"
    #: 累计模型调用次数，用于报告中统计调用成本
    calls: int = 0

    def analyze_image(self, prompt: str, image_path: Path, task: str = "perception") -> Dict[str, Any]:
        return self.analyze_images(prompt, [image_path], task=task)

    def analyze_images(self, prompt: str, image_paths: List[Path], task: str = "perception") -> Dict[str, Any]:
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def is_simulated(self) -> bool:
        return self.mode == "mock"

    def _count_call(self) -> None:
        self.calls = int(getattr(self, "calls", 0)) + 1


class MockLLM(BaseLLM):
    """确定性模拟，用于无 API 环境跑通全流程。

    为了能在课堂上演示“Agent 自主调用工具”，第一轮会先申请工具、
    拿到观察结果后再给出结论 —— 与真实模型的行为路径一致。
    """

    mode = "mock"

    def analyze_images(self, prompt: str, image_paths: List[Path], task: str = "perception") -> Dict[str, Any]:
        self._count_call()
        primary = image_paths[0]
        name = primary.name.lower()

        if task == "review":
            # 模拟复核：一律判定证据不足，避免模拟结果被当成确认结论。
            return {
                "verdict": "uncertain",
                "reason": "模拟模式无法真正核验图像证据。",
                "supported": False,
            }

        findings = []
        for keywords, payload in _MOCK_NAME_RULES:
            if any(k in name for k in keywords):
                findings.append({
                    "risk_type": payload["risk_type"],
                    "severity": payload["severity"],
                    "evidence": payload["evidence"],
                    "location": payload["location"],
                    "time_range": "unknown",
                    "confidence": 0.40,
                    "uncertainty": _MOCK_DISCLAIMER,
                })
                break  # 文件名只对应一个演示场景，避免模拟数据过度触发

        # 第一轮：先申请工具，模拟“Agent 根据分析需要选择工具”
        # 仅对场景感知任务生效；基础对照系统(baseline)不做工具调用
        if task == "perception" and findings and "【工具调用结果】" not in prompt:
            requests = [
                {"tool": "search_rules", "args": {"query": findings[0]["risk_type"]}},
                {"tool": "search_cases", "args": {"risk_type": findings[0]["risk_type"]}},
            ]
            if len(image_paths) == 1:
                requests.append({
                    "tool": "crop_region",
                    "args": {"left": 0.25, "top": 0.10, "right": 0.85, "bottom": 0.95, "tag": "key_region"},
                })
            return {
                "scene_description": f"模拟模式：已读取 {primary.name}，需要补充信息后再定级。",
                "findings": [],
                "tool_requests": requests,
            }

        return {
            "scene_description": f"模拟模式：已读取 {primary.name}，未进行真实视觉理解。",
            "findings": findings,
            "tool_requests": [],
        }

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        """模拟文本任务。把 user 里的候选风险原样回填，等级沿用规则默认值。"""
        self._count_call()
        try:
            payload = extract_json(user)
        except ValueError:
            return {"results": []}

        candidates = payload.get("candidates", [])
        results = []
        for item in candidates:
            results.append({
                "index": item.get("index"),
                "severity": item.get("default_severity", "uncertain"),
                "recommendation": item.get("default_recommendation", ""),
                "confidence": item.get("confidence", 0.0),
                "uncertainty": item.get("uncertainty", "") or _MOCK_DISCLAIMER,
                "reasoning": "模拟模式：沿用规则库默认等级，未进行真实推理。",
            })
        return {"results": results}


class OpenAICompatibleVisionLLM(BaseLLM):
    mode = "api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("SAFETY_LLM_API_KEY")
        self.base_url = (
            base_url or os.getenv("SAFETY_LLM_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("SAFETY_LLM_MODEL") or "gpt-4o-mini"
        # 图片传参格式：data_uri（OpenAI 标准，"data:image/jpeg;base64,..."）
        # 或 base64（只给裸 base64 串）。智谱 GLM-4V 等国内接口有的只吃后者。
        self.image_format = (os.getenv("SAFETY_LLM_IMAGE_FORMAT") or "data_uri").strip().lower()
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError("缺少 SAFETY_LLM_API_KEY 环境变量")

    # ---------- 底层请求 ----------
    def _post(self, messages: list, response_format: bool = True, temperature: float = 0.1) -> Dict[str, Any]:
        self._count_call()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", "ignore")[:500]
                last_err = RuntimeError(f"HTTP {err.code}: {detail}")
                # 400 通常是该模型不支持 response_format，降级重试一次
                if err.code == 400 and response_format:
                    response_format = False
                    payload.pop("response_format", None)
                    body = json.dumps(payload).encode("utf-8")
                    continue
                if err.code in (401, 403):
                    raise RuntimeError(f"API 鉴权失败（{err.code}），请检查 SAFETY_LLM_API_KEY。") from err
            except Exception as err:  # noqa: BLE001 - 网络类异常统一重试
                last_err = err
        raise RuntimeError(f"调用模型失败：{last_err}")

    @staticmethod
    def _content(data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as err:
            raise RuntimeError(f"模型返回结构异常：{str(data)[:300]}") from err

    # ---------- 对外接口 ----------
    def analyze_images(self, prompt: str, image_paths: List[Path], task: str = "perception") -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            url = (image_b64 if self.image_format == "base64"
                   else f"data:{mime};base64,{image_b64}")
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        data = self._post([{"role": "user", "content": content}])
        return extract_json(self._content(data))

    def analyze_image(self, prompt: str, image_path: Path, task: str = "perception") -> Dict[str, Any]:
        return self.analyze_images(prompt, [image_path], task=task)

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        data = self._post(messages)
        return extract_json(self._content(data))


def describe_backend() -> Dict[str, Any]:
    """给界面/报告用：当前会走真实 API 还是模拟模式。"""
    has_key = bool(os.getenv("SAFETY_LLM_API_KEY"))
    return {
        "has_api_key": has_key,
        "base_url": os.getenv("SAFETY_LLM_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("SAFETY_LLM_MODEL", "gpt-4o-mini"),
        "image_format": (os.getenv("SAFETY_LLM_IMAGE_FORMAT") or "data_uri").strip().lower(),
        "mode": "api" if has_key else "mock",
    }


def get_llm(force_mock: bool = False) -> BaseLLM:
    if force_mock:
        return MockLLM()
    try:
        return OpenAICompatibleVisionLLM()
    except Exception:
        return MockLLM()
