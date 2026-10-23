"""用单帧画面拼接「巡视镜头」序列视频，用于验证跨帧时间融合。

为什么这样拼
------------
课程要求系统能处理短视频、并且结论要结合多个时间点的信息。真正要验证的是：

  * 危险在多数帧持续出现  → 应当确认
  * 危险只在个别帧一闪而过 → 应当降权，不能当成稳定存在的风险

实验室里不会为了做实验去制造违规现场并录像，所以这里用「同一巡视路线依次扫过
不同工位」的方式构造序列：每个镜头是一张独立画面的 3 秒定格，镜头间做淡入淡出。
它不是真实连续拍摄，但时间结构是真实的 —— 第几帧看到什么，与标注一一对应。

重要：这些视频的源图取自 **dev 集**，绝不动用 test 集，避免污染最终评测。
视频本身只用于机制演示，不计入 84 个独立图片样本的定量统计。

用法：
    python scripts/make_sequence_videos.py
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FPS = 10
SECONDS_PER_SHOT = 3
FADE_FRAMES = 6
OUT_SIZE = (1024, 768)  # (宽, 高)

#: 每段视频：文件名 -> (说明, 镜头列表[(图片相对路径, 该镜头真值)])
SEQUENCES = {
    "video_spill_persist": (
        "巡视开始于合规工位，随后连续三个镜头都拍到地面洒漏未处理 —— 危险持续存在",
        [
            ("data/synth/synth_confuse_01.jpg", "normal"),
            ("data/synth/synth_spill_or_obstruction_03.jpg", "spill_or_obstruction"),
            ("data/synth/synth_spill_or_obstruction_06.jpg", "spill_or_obstruction"),
            ("data/synth/synth_spill_or_obstruction_07.jpg", "spill_or_obstruction"),
        ],
    ),
    "video_food_transient": (
        "四个镜头里只有第二个出现实验区饮食，前后都是合规台面 —— 属瞬时出现，应降权",
        [
            ("data/synth/synth_confuse_02.jpg", "normal"),
            ("data/synth/synth_food_or_drink_01.jpg", "food_or_drink"),
            ("data/synth/synth_confuse_06.jpg", "normal"),
            ("data/synth/synth_confuse_15.jpg", "normal"),
        ],
    ),
    "video_ppe_persist": (
        "后三个镜头学生都没有正确佩戴护目镜 —— 危险持续存在，应确认",
        [
            ("data/synth/synth_confuse_01.jpg", "normal"),
            ("data/synth/synth_ppe_missing_01.jpg", "ppe_missing"),
            ("data/synth/synth_ppe_missing_04.jpg", "ppe_missing"),
            ("data/synth/synth_confuse_13.jpg", "ppe_missing"),
        ],
    ),
}


def _read_image(path: str):
    """读取图片。cv2.imread 在 Windows 上不认中文路径，改用 imdecode。"""
    import cv2
    import numpy as np

    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _fit(img, size):
    """等比缩放后居中裁切到目标画幅。"""
    import cv2

    target_w, target_h = size
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    nh, nw = resized.shape[:2]
    x0 = max(0, (nw - target_w) // 2)
    y0 = max(0, (nh - target_h) // 2)
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成演示用序列视频")
    parser.add_argument("--out", default="data/web_video", help="输出目录")
    parser.add_argument("--only", default="", help="只生成指定的一段")
    args = parser.parse_args()

    import cv2
    import numpy as np

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (desc, shots) in SEQUENCES.items():
        if args.only and args.only != name:
            continue

        frames = []
        missing = False
        for rel, _truth in shots:
            path = PROJECT_ROOT / rel
            if not path.exists():
                print(f"  ! 缺少 {rel}，跳过 {name}")
                missing = True
                break
            img = _read_image(str(path))
            if img is None:
                print(f"  ! 无法读取 {rel}，跳过 {name}")
                missing = True
                break
            frames.append(_fit(img, OUT_SIZE))
        if missing:
            continue

        video_path = out_dir / f"{name}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, OUT_SIZE)
        per_shot = FPS * SECONDS_PER_SHOT
        total = 0

        for i, shot in enumerate(frames):
            for f in range(per_shot):
                frame = shot.copy()
                # 镜头首尾做淡入淡出，模拟镜头切换；中间保持原样
                if f < FADE_FRAMES:
                    alpha = f / FADE_FRAMES
                    frame = (frame.astype(np.float32) * alpha).astype(np.uint8)
                elif f > per_shot - FADE_FRAMES:
                    alpha = max(0.0, (per_shot - f) / FADE_FRAMES)
                    frame = (frame.astype(np.float32) * alpha).astype(np.uint8)
                writer.write(frame)
                total += 1
        writer.release()

        seconds = total / FPS
        print(f"已生成 {video_path}（{len(frames)} 个镜头，{seconds:.1f}s，{total} 帧）")
        print(f"  说明：{desc}")
        print(f"  镜头真值：{', '.join(t for _, t in shots)}")

    print("\n提示：这些是拼接序列，不是真实连续拍摄，报告里要如实说明。")
    print("      源图取自 dev 集，未使用 test 样本。")


if __name__ == "__main__":
    main()
