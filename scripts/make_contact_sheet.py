"""把一批图片拼成带编号的「接触印相表」（contact sheet），用于人工标注。

为什么需要它
------------
标注网络图库素材时，必须逐张看清画面。一张一张打开太慢，纯看缩略图列表又看不清
「护目镜戴没戴」「瓶子有没有标签」这类关键细节。折中做法：把 3x3 或 4x3 张图拼成
一张大图、每格左上角打上序号，一次看完 9~12 张。

配合 --index-out 输出的 JSON（序号 -> 文件名），可以把「看图得到的人工判断」
直接映射回具体文件，再生成标注 CSV。

用法：
    python scripts/make_contact_sheet.py --src data/web_raw --out reports/sheets/sheet_01.png
    python scripts/make_contact_sheet.py --src data/web_raw --page 2 --cols 3 --thumb 420
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(src: Path, skip_names=("manifest.json",), manifest: str = "",
            group_by_query: bool = False) -> list:
    files = [
        p for p in sorted(src.iterdir())
        if p.is_file() and p.suffix.lower() in EXTS and p.name not in skip_names
    ]
    if not manifest:
        return files, {}

    # 按 manifest 里的采集查询分组排序：同一查询词下载的图排在一起，
    # 看印相表时更容易判断「这批图是不是同一个拍摄题材」。
    mpath = PROJECT_ROOT / manifest if not Path(manifest).is_absolute() else Path(manifest)
    try:
        items = json.loads(mpath.read_text(encoding="utf-8")).get("items", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  ! 读取 manifest 失败（{exc}），按文件名排序")
        return files, {}

    order = {}
    query_of = {}
    for i, it in enumerate(items):
        pid = str(it.get("photo_id") or Path(it.get("file", "")).stem)
        query_of[pid] = it.get("query", "")
        order[pid] = i

    def key(p: Path):
        pid = p.stem
        # 不分组时按下载顺序，分组时按 (查询词, 下载顺序)
        q = query_of.get(pid, "~unknown")
        return (q, order.get(pid, 1 << 30)) if group_by_query else (order.get(pid, 1 << 30), q)

    files.sort(key=key)
    files = [p for p in files if p.stem in query_of] + [p for p in files if p.stem not in query_of]
    return files, query_of


def main() -> None:
    parser = argparse.ArgumentParser(description="生成带编号的接触印相表")
    parser.add_argument("--src", default="data/web_raw")
    parser.add_argument("--out", default="reports/sheets/sheet.png")
    parser.add_argument("--cols", type=int, default=3, help="每行几张")
    parser.add_argument("--thumb", type=int, default=430, help="单格宽度（像素）")
    parser.add_argument("--rows", type=int, default=3, help="每页行数")
    parser.add_argument("--page", type=int, default=0, help="第几页（从 1 开始，0 表示全部）")
    parser.add_argument("--start", type=int, default=1, help="起始序号（用于连续编号）")
    parser.add_argument("--index-out", default="", help="序号->文件名映射的 JSON 输出路径")
    parser.add_argument("--manifest", default="", help="采集清单 JSON，用于按查询词分组排序")
    parser.add_argument("--group-by-query", action="store_true", help="按采集查询词分组")
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    src = PROJECT_ROOT / args.src if not Path(args.src).is_absolute() else Path(args.src)
    files, query_of = collect(src, manifest=args.manifest,
                              group_by_query=args.group_by_query)
    if not files:
        print(f"{src} 下没有图片")
        return

    per_page = args.cols * args.rows
    if args.page > 0:
        begin = (args.page - 1) * per_page
        page_files = files[begin:begin + per_page]
    else:
        page_files = files

    if not page_files:
        print(f"第 {args.page} 页没有内容（共 {len(files)} 张，每页 {per_page} 张）")
        return

    thumb_w = args.thumb
    # 图片多为竖构图（网络图库常见 2:3），按 4:3 高度留白，保证标题条不遮挡画面
    cell_h = int(thumb_w * 0.78) + 34
    rows = (len(page_files) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (thumb_w * args.cols, cell_h * rows), (250, 250, 252))
    draw = ImageDraw.Draw(sheet)

    index = {}
    for i, path in enumerate(page_files):
        num = args.start + i
        r, c = divmod(i, args.cols)
        x0, y0 = c * thumb_w, r * cell_h
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                w, h = rgb.size
                # 等比缩放到格子内，居中留白
                box_w, box_h = thumb_w - 12, cell_h - 34 - 10
                scale = min(box_w / w, box_h / h)
                new = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                ox = x0 + (thumb_w - new.width) // 2
                oy = y0 + 30 + (box_h - new.height) // 2
                sheet.paste(new, (ox, oy))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 跳过 {path.name}：{exc}")
            continue

        label = f"#{num}  {path.stem}"
        q = query_of.get(path.stem, "")
        if q:
            label = f"#{num} [{q[:14]}] {path.stem}"
        draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + 26], fill=(28, 32, 40))
        draw.text((x0 + 6, y0 + 7), label[:44], fill=(255, 255, 255))
        draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + cell_h - 1],
                       outline=(200, 204, 212), width=1)
        index[str(num)] = path.name

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG")
    print(f"已生成 {out}（{len(page_files)} 张，序号 {args.start}~{args.start + len(page_files) - 1}）")

    if args.index_out:
        idx_path = PROJECT_ROOT / args.index_out
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if idx_path.exists():
            try:
                existing = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {}
        existing.update(index)
        idx_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"序号映射：{idx_path}")


if __name__ == "__main__":
    main()
