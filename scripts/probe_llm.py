"""单次真实调用探针：验证智谱 GLM-4V 的连通性与图片传参格式。

只发一次请求，不跑完整多 Agent 流程，用来快速定位配置问题。
用法：
    python scripts/probe_llm.py <图片路径>
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm import get_llm  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python scripts/probe_llm.py <图片路径>")
        sys.exit(1)

    image = Path(sys.argv[1])
    if not image.exists():
        print(f"图片不存在：{image}")
        sys.exit(1)

    size_kb = image.stat().st_size / 1024
    print(f"图片：{image.name}（{size_kb:.1f} KB）")

    llm = get_llm()
    print(f"模式：{llm.mode}｜模型：{getattr(llm, 'model', '—')}"
          f"｜传参格式：{getattr(llm, 'image_format', '—')}")
    if llm.is_simulated:
        print("当前是模拟模式，请先配置 .env 里的 SAFETY_LLM_API_KEY")
        sys.exit(1)

    prompt = (
        "请描述这张图片的内容。然后判断它是否是高校实验室场景（有实验台、"
        "仪器、试剂、穿实验服的人员等）。\n"
        "只输出 JSON，格式："
        '{"scene_description": "画面描述", "is_lab": true 或 false, '
        '"reason": "判断依据"}'
    )

    started = time.time()
    try:
        result = llm.analyze_image(prompt, image, task="perception")
    except Exception as exc:  # noqa: BLE001 - 探针要如实暴露任何失败
        elapsed = time.time() - started
        print(f"❌ 调用失败（{elapsed:.1f}s）：{type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = time.time() - started
    print(f"✅ 调用成功（{elapsed:.1f}s）")
    print(f"场景描述：{result.get('scene_description', '（无）')}")
    print(f"是否实验室：{result.get('is_lab')}")
    print(f"依据：{result.get('reason', '（无）')}")


if __name__ == "__main__":
    main()
