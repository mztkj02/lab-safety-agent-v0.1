"""生成用于冒烟测试与课堂演示的「占位数据集」。

重要说明：
    这个脚本生成的图片是纯色块 + 文字标注的程序化占位图，**不是真实实验室照片**，
    只能用来验证“系统能不能跑通”，绝不能拿来当实验数据或训练/评测依据。
    真实数据必须按课题要求自行采集（≥80 个独立样本，含正常/危险/易混淆三类）。

用法：
    python scripts/make_demo_dataset.py --count 24
    python scripts/make_demo_dataset.py --count 24 --video      # 额外生成一段演示视频
"""

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 风险类型 -> (文件名关键词, 画面主色, 中文说明)
#: 关键词刻意与 llm.MockLLM 的识别关键词对齐，保证无 API 时也能跑通端到端流程。
SCENARIOS = {
    "normal": (["lab_normal"], (226, 232, 240), "正常实验室场景"),
    "ppe_missing": (["no_goggle", "no_ppe", "ppe_missing"], (255, 228, 230), "PPE 缺失"),
    "reagent_management": (["reagent_open", "bottle_nolabel"], (254, 243, 199), "试剂管理异常"),
    "flammable_near_heat": (["flammable_heater", "alcohol_fire"], (254, 215, 215), "热源附近易燃物"),
    "food_or_drink": (["drink_cup", "food_table"], (220, 252, 231), "实验区饮食"),
    "spill_or_obstruction": (["spill_floor", "cable_aisle"], (219, 234, 254), "洒漏或通道阻塞"),
}


#: 容易混淆样本：(文件名线索, 真实标注, 说明)
#: 课程要求数据集中包含“容易混淆的场景”。这里用误导性的文件名线索来模拟
#: “模型只看表面特征就会判错”的情形，用于验证证据复核与自进化能否把误报压下来。
CONFUSABLE = [
    ("no_goggle", "food_or_drink", "实验台上其实是透明水杯，容易被误判为未戴护目镜"),
    ("no_goggle", "normal", "拍摄角度导致护目镜被头部轮廓遮挡，实际佩戴完整"),
    ("bottle_nolabel", "normal", "试剂瓶标签被手挡住，实际标签完整清晰"),
    ("flammable_heater", "normal", "加热板已断电冷却，旁边的纸张不构成热源风险"),
    ("spill_floor", "normal", "地面反光与阴影被误判为液体洒漏"),
    ("drink_cup", "normal", "台面上是透明洗瓶与纯水试剂瓶，不是饮用容器"),
]


def _draw(path: Path, title: str, subtitle: str, color, index: int) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False

    width, height = 640, 420
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # 画一些几何形状，避免纯色图被某些模型直接判为不可用
    random.seed(index)
    for _ in range(12):
        x1, y1 = random.randint(0, width - 100), random.randint(60, height - 80)
        x2, y2 = x1 + random.randint(30, 120), y1 + random.randint(30, 90)
        draw.rectangle([x1, y1, x2, y2], outline=(120, 130, 145), width=2)

    draw.rectangle([0, 0, width, 56], fill=(255, 255, 255))
    draw.text((16, 14), f"PLACEHOLDER #{index}  {title}", fill=(30, 41, 59))
    draw.text((16, 34), subtitle, fill=(100, 116, 139))
    draw.rectangle([0, 0, width - 1, height - 1], outline=(148, 163, 184), width=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=88)
    return True


SEVERITY_BY_TYPE = {
    "normal": "normal", "ppe_missing": "medium", "reagent_management": "medium",
    "flammable_near_heat": "high", "food_or_drink": "low", "spill_or_obstruction": "medium",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成占位数据集（仅用于跑通流程）")
    parser.add_argument("--count", type=int, default=30, help="生成样本总数（课程要求真实数据 ≥80）")
    parser.add_argument("--out", default="data/demo", help="输出目录")
    parser.add_argument("--confusable-every", type=int, default=6,
                        help="每隔多少个样本插入一个易混淆样本")
    parser.add_argument("--video", action="store_true", help="额外生成一段演示视频用于测试时间维度融合")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _draw(out_dir / "_probe.jpg", "probe", "检测 Pillow 是否可用", (240, 240, 240), 0):
        print("错误：未安装 Pillow，请先执行 pip install pillow")
        sys.exit(1)
    (out_dir / "_probe.jpg").unlink()

    kinds = list(SCENARIOS)
    rows = []
    confusable_idx = 0

    for i in range(1, args.count + 1):
        if args.confusable_every and i % args.confusable_every == 0:
            keyword, true_type, note = CONFUSABLE[confusable_idx % len(CONFUSABLE)]
            confusable_idx += 1
            name = f"demo_{i:04d}_confuse_{keyword}.jpg"
            path = out_dir / name
            _draw(path, keyword, f"占位图 · 容易混淆样本 · {note}", (243, 244, 246), i)
            rows.append({
                "sample_id": path.stem,
                "file_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "split": "dev" if (i % 10) < 7 else "test",
                "risk_type": true_type,
                "severity": SEVERITY_BY_TYPE[true_type],
                "evidence": note,
                "location": "见说明",
                "notes": "容易混淆样本：文件名线索会误导模型，用于验证误报抑制能力",
            })
            continue

        kind = kinds[(i - 1) % len(kinds)]
        keywords, color, cn = SCENARIOS[kind]
        keyword = keywords[i % len(keywords)]
        name = f"demo_{i:04d}_{keyword}.jpg"
        path = out_dir / name
        _draw(path, keyword, f"占位图 · 类别 {cn} · 非真实数据", color, i)
        rows.append({
            "sample_id": path.stem,
            "file_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "split": "dev" if (i % 10) < 7 else "test",
            "risk_type": kind,
            "severity": SEVERITY_BY_TYPE[kind],
            "evidence": f"占位数据：{cn}（程序化生成，非真实标注）",
            "location": "全画面",
            "notes": "占位数据，仅用于验证流程可运行",
        })

    fields = ["sample_id", "file_path", "split", "risk_type", "severity", "evidence", "location", "notes"]
    for split in ("dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        csv_path = PROJECT_ROOT / "data" / f"demo_labels_{split}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        print(f"已写入 {csv_path}（{len(subset)} 条）")

    total_confusable = sum(1 for r in rows if "confuse" in r["sample_id"])
    print(f"占位样本共 {len(rows)} 张（其中易混淆 {total_confusable} 张），输出目录 {out_dir}")
    print("提醒：这些是程序化占位图，不是真实实验数据，仅用于验证流程可跑通。")

    if args.video:
        _make_video(out_dir)


def _make_video(out_dir: Path, frames: int = 24) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        print("未安装 opencv-python，跳过视频生成。pip install opencv-python")
        return

    video_path = out_dir / "demo_spill_floor.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 420))
    for i in range(frames):
        frame = np.full((420, 640, 3), 235, dtype=np.uint8)
        # 第 8~20 帧画出一片"洒漏液体"，用于验证跨帧持续跟踪
        if 8 <= i <= 20:
            cv2.rectangle(frame, (150, 260), (480, 380), (120, 150, 190), -1)
        cv2.putText(frame, f"PLACEHOLDER VIDEO frame {i + 1}/{frames}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        writer.write(frame)
    writer.release()
    print(f"已生成演示视频 {video_path}（含 13 帧持续出现的液体区域，用于测试时间维度融合）")


if __name__ == "__main__":
    main()
