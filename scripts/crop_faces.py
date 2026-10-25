"""把图片的上部区域裁剪放大，用于核对「是否佩戴护目镜」这类细节。

人工标注的真值必须逐张看清眼睛部位，缩略图看不出来。
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="裁剪放大脸部区域")
    parser.add_argument("images", nargs="+", help="图片路径")
    parser.add_argument("--out-dir", default=".cache/crop_check")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--top-ratio", type=float, default=0.55, help="裁剪高度占原图比例")
    parser.add_argument("--width-ratio", type=float, default=0.6, help="裁剪宽度占比")
    args = parser.parse_args()

    from PIL import Image

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw in args.images:
        path = PROJECT_ROOT / raw if not Path(raw).is_absolute() else Path(raw)
        if not path.exists():
            print(f"  ! 不存在：{path}")
            continue
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            box = (0, 0, int(w * args.width_ratio), int(h * args.top_ratio))
            crop = rgb.crop(box)
            crop = crop.resize(
                (crop.width * args.scale, crop.height * args.scale), Image.LANCZOS
            )
            dest = out_dir / f"{path.stem}_faces.jpg"
            crop.save(dest, "JPEG", quality=95)
        print(f"  {path.name} -> {dest.relative_to(PROJECT_ROOT)}  {crop.size}")


if __name__ == "__main__":
    main()
