"""把素材统一转成「真 JPEG」：修正格式、剥除元数据、限制尺寸。

为什么必须做
------------
Pexels 的图片 CDN 会根据请求头里的 `Accept` 返回 AVIF 格式，但文件仍然叫 `.jpg`。
本地看图没问题（Pillow 会按内容识别），可是多模态模型接口只认真正的 JPEG / PNG，
收到 AVIF 会直接返回 `400 code 1210 图片输入格式/解析错误`。

这个坑在第一次跑实验时才暴露：43 个测试样本里有 18 个（41.9%）调用失败，
而失败又被上层当成「没发现风险」静默吞掉，指标全部失真。

顺带解决两件事：
  1. 剥掉 EXIF —— 图库图不一定有 GPS，但统一剥掉是稳妥的隐私习惯；
  2. 限制最长边 —— 1280px 对视觉模型足够，能显著减少上传体积与响应时间。

用法：
    python scripts/normalize_images.py                 # 处理 data/web_raw 与 data/synth
    python scripts/normalize_images.py --dir data/web_raw --dry-run
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = ["data/web_raw", "data/synth"]
MAX_SIDE = 1280
QUALITY = 88


def probe(path: Path) -> dict:
    """读出一张图的真实格式信息。"""
    from PIL import Image

    with Image.open(path) as im:
        return {
            "format": im.format,
            "mode": im.mode,
            "size": im.size,
            "bytes": path.stat().st_size,
        }


def normalize(path: Path, max_side: int = MAX_SIDE, quality: int = QUALITY) -> str:
    """转成标准 sRGB JPEG，原地覆盖。返回处理说明；无需处理时返回空串。"""
    from PIL import Image

    with Image.open(path) as im:
        real_fmt = (im.format or "").upper()
        need = real_fmt != "JPEG" or im.mode not in ("RGB", "L")

        img = im.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
            need = True

        if not need:
            # 已经是标准 JPEG 且尺寸合规，只重写一遍以剥掉 EXIF
            pass

        # 不带 exif / icc 参数保存，等于剥除全部元数据
        img.save(path, "JPEG", quality=quality, optimize=True)

    return f"{real_fmt}->JPEG"


def main() -> None:
    parser = argparse.ArgumentParser(description="统一图片格式为真 JPEG")
    parser.add_argument("--dir", action="append", help="要处理的目录（可重复）")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--quality", type=int, default=QUALITY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dirs = args.dir or DEFAULT_DIRS
    total = 0
    converted = 0
    before_bytes = 0
    after_bytes = 0

    for d in dirs:
        folder = PROJECT_ROOT / d if not Path(d).is_absolute() else Path(d)
        if not folder.exists():
            print(f"跳过不存在的目录：{folder}")
            continue
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"\n{d}：{len(files)} 个文件")
        for path in files:
            total += 1
            before_bytes += path.stat().st_size
            info = probe(path)
            if args.dry_run:
                print(f"  {path.name}: {info['format']} {info['mode']} {info['size']}")
                after_bytes += info["bytes"]
                continue
            try:
                note = normalize(path, args.max_side, args.quality)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {path.name} 处理失败：{exc}")
                after_bytes += path.stat().st_size
                continue
            after_bytes += path.stat().st_size
            if note:
                converted += 1
                new = probe(path)
                print(f"  {path.name}: {note}  {info['size']}->{new['size']}"
                      f"  {info['bytes']//1024}KB->{new['bytes']//1024}KB")

    if args.dry_run:
        print(f"\n[试运行] 共 {total} 个文件，未修改任何内容")
        return

    print(f"\n共处理 {total} 个文件，其中格式转换 {converted} 个")
    print(f"总体积 {before_bytes/1024/1024:.1f} MB -> {after_bytes/1024/1024:.1f} MB")
    print("\n提醒：转换后请重跑数据集与实验，旧报告里的数字已不可用。")


if __name__ == "__main__":
    main()
