"""构造长时序测试视频：长时间稳定画面 + 一个持续数秒的异常事件。

用途：验证当前 `VideoFrameExtractor`（固定抽 N 帧、均匀分布）在长时间监控下
会不会直接把异常漏掉，以及要覆盖住需要多少帧。

    python scripts/make_long_video.py --duration 240 --event-at 120 --event-len 3

产出：.cache/long_video/<name>.mp4 与一份 event.json（记录异常真实时间段，
方便回头核对"抽到的帧里到底有没有事件"）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / ".cache" / "long_video"


def _load(path: Path):
    """读图。用 np.fromfile + imdecode，cv2.imread 不认中文路径。"""
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _fit(img, size):
    import cv2

    w, h = size
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale) + 1, int(ih * scale) + 1))
    y0 = max(0, (resized.shape[0] - h) // 2)
    x0 = max(0, (resized.shape[1] - w) // 2)
    return resized[y0:y0 + h, x0:x0 + w]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成长时序测试视频")
    parser.add_argument("--duration", type=float, default=240.0, help="视频总时长（秒）")
    parser.add_argument("--event-at", type=float, default=120.0, help="异常开始时间（秒）")
    parser.add_argument("--event-len", type=float, default=3.0, help="异常持续时长（秒）")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--name", default="", help="输出文件名（不含扩展名）")
    parser.add_argument("--normal", nargs="*", default=[], help="正常画面图片路径")
    parser.add_argument("--event", default="", help="异常画面图片路径")
    args = parser.parse_args()

    import cv2

    # 默认素材：用 dev 集里的正常实验室图 + 一张明火风险图，避免碰到 test 集
    normal_paths = [Path(p) for p in args.normal] or [
        PROJECT_ROOT / "data" / "web_raw" / "38093862.jpg",
        PROJECT_ROOT / "data" / "web_raw" / "8851786.jpg",
        PROJECT_ROOT / "data" / "web_raw" / "9243559.jpg",
    ]
    event_path = Path(args.event) if args.event else (
        PROJECT_ROOT / "data" / "synth" / "synth_flammable_near_heat_01.jpg"
    )

    missing = [str(p) for p in normal_paths + [event_path] if not p.exists()]
    if missing:
        print("缺少素材：")
        for m in missing:
            print("  ", m)
        return

    normals = [_fit(_load(p), (args.width, args.height)) for p in normal_paths]
    event_img = _fit(_load(event_path), (args.width, args.height))
    normals = [n for n in normals if n is not None]
    if not normals or event_img is None:
        print("图片读取失败")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or f"long_{int(args.duration)}s_event{int(args.event_at)}"
    out_path = OUT_DIR / f"{name}.mp4"

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width, args.height),
    )

    total_frames = int(args.duration * args.fps)
    event_start = int(args.event_at * args.fps)
    event_end = int((args.event_at + args.event_len) * args.fps)

    # 正常画面每秒轮换一张，避免整段视频是完全静态（那样帧去重会失真）
    for i in range(total_frames):
        if event_start <= i < event_end:
            writer.write(event_img)
        else:
            idx = (i // args.fps) % len(normals)
            writer.write(normals[idx])
    writer.release()

    meta = {
        "file": str(out_path),
        "duration_sec": args.duration,
        "fps": args.fps,
        "total_frames": total_frames,
        "event_start_sec": args.event_at,
        "event_end_sec": args.event_at + args.event_len,
        "event_frames": [event_start, event_end - 1],
        "normal_sources": [str(p) for p in normal_paths],
        "event_source": str(event_path),
    }
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"已生成 {out_path}")
    print(f"  {args.duration:.0f}s / {args.fps}fps / {total_frames} 帧 / {size_mb:.1f} MB")
    print(f"  异常时段：{args.event_at:.0f}s ~ {args.event_at + args.event_len:.0f}s"
          f"（第 {event_start}~{event_end - 1} 帧）")


if __name__ == "__main__":
    main()
