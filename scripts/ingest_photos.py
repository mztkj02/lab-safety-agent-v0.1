"""把手机拍的原始照片/视频整理成规范数据集。

三个人各自拍照汇总到一起，一定会遇到这些事：

1. 文件名重复（都是 IMG_0001.jpg）—— 直接拷贝会互相覆盖
2. 同一张照片被重复提交 —— 会同时进 dev 和 test，指标虚高
3. 照片带 GPS 定位 —— 直接把学校/宿舍坐标传到 GitHub 上
4. 视频抽出的相邻帧被分到不同集合 —— 等于让测试集偷看训练集（课程明令禁止）
5. 图片太大/太歪，模型读不出

本脚本按顺序解决：

    扫描 -> 去重 -> 剥除 EXIF 隐私信息 -> 归一化尺寸
         -> 统一编号 -> 以「视频」为单位分配 dev/test -> 产出标注草稿

用法：

    # 1) 把照片/视频放进 data/inbox/（子目录也可以）
    # 2) 生成数据集骨架与标注草稿
    python scripts/ingest_photos.py

    # 只看会做什么，不写文件
    python scripts/ingest_photos.py --dry-run

    # 视频抽帧密度（默认每 1 秒 1 帧）
    python scripts/ingest_photos.py --fps 0.5

产出：
    data/raw/lab_0001.jpg ...           归一化后的图片
    data/labels_draft.csv               标注草稿（risk_type 等待人工填写）
    data/ingest_report.json             去重、跳过、划分的详细记录

标注草稿的 risk_type 故意留空：**标注必须由人来做**，脚本猜出来的标注会让
整个实验失去意义。文件名里带 no_goggle / spill 之类的关键词只会写进 notes
作为提示，不会自动填进 risk_type。
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

#: 标注草稿的列，与 data/labels_template.csv 保持一致
LABEL_FIELDS = [
    "sample_id", "file_path", "split", "risk_type",
    "severity", "evidence", "location", "notes",
]

#: 文件名里出现这些词时，只在 notes 里提示「可能属于某类」，不写 risk_type
KEYWORD_HINTS = {
    "no_goggle": "可能为 ppe_missing（未戴护目镜）",
    "nogoggle": "可能为 ppe_missing（未戴护目镜）",
    "no_glove": "可能为 ppe_missing（未戴手套）",
    "no_coat": "可能为 ppe_missing（未穿实验服）",
    "reagent": "可能为 reagent_management（试剂管理）",
    "bottle": "可能为 reagent_management（试剂瓶）",
    "heater": "可能为 flammable_near_heat（热源）",
    "flame": "可能为 flammable_near_heat（明火）",
    "cup": "可能为 food_or_drink（水杯）",
    "drink": "可能为 food_or_drink（饮料）",
    "spill": "可能为 spill_or_obstruction（洒漏）",
    "cable": "可能为 spill_or_obstruction（电线）",
    "block": "可能为 spill_or_obstruction（阻挡）",
    "normal": "可能为正常场景",
}


def _rel(path: Path) -> str:
    """尽量返回相对项目根目录的路径，便于写进 CSV。

    素材目录允许放在项目外（比如直接指向手机备份盘），这时 relative_to 会抛
    ValueError，退回绝对路径即可，不能因此让整个导入崩掉。
    """
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan_media(inbox: Path) -> Tuple[List[Path], List[Path]]:
    """返回 (图片列表, 视频列表)，跳过隐藏文件和 macOS 的 ._ 垃圾文件。"""
    images, videos = [], []
    if not inbox.exists():
        return images, videos
    for path in sorted(inbox.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images.append(path)
        elif suffix in VIDEO_SUFFIXES:
            videos.append(path)
    return images, videos


def keyword_hint(name: str) -> str:
    lowered = name.lower()
    for key, hint in KEYWORD_HINTS.items():
        if key in lowered:
            return hint
    return ""


def normalize_image(src: Path, dst: Path, max_side: int = 1600, quality: int = 90) -> bool:
    """缩放并用**不含元数据**的方式重新保存，顺带剥掉 GPS 等信息。

    Pillow 保存时不会带原图的 EXIF（除非显式传入 exif=），所以这一步天然完成
    隐私清洗。这是课程「隐私合规」要求里最容易漏掉的一条。
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return False
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)      # 手机竖拍会带旋转标记，先摆正
            img = img.convert("RGB")                 # 统一去掉 alpha/CMYK
            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / longest
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def extract_video_frames(src: Path, out_dir: Path, fps: float, max_side: int = 1600) -> List[Path]:
    """按固定间隔抽帧。返回抽出的图片路径（按时间顺序）。"""
    try:
        import cv2
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # 每隔多少帧取一张
    step = max(1, int(round(native_fps / fps))) if fps > 0 else int(native_fps)

    frames: List[Path] = []
    index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            h, w = frame.shape[:2]
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / longest
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            saved += 1
            path = out_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frames.append(path)
        index += 1
    cap.release()
    return frames


def assign_splits(groups: Dict[str, int], ratio: float = 0.7, seed: int = 20261016) -> Dict[str, str]:
    """以「组」为单位分配 dev/test。

    组 = 一张照片 / 一段视频的全部帧。**同一视频的帧必须落在同一集合**，
    否则测试集里会出现训练时见过的画面，指标会虚高且违反课程要求。

    groups: {组名: 该组包含的样本数}
    返回: {组名: "dev" | "test"}
    """
    names = sorted(groups)
    random.Random(seed).shuffle(names)

    total = sum(groups.values()) or 1
    target_dev = total * ratio

    result: Dict[str, str] = {}
    dev_count = 0
    for name in names:
        # 只按样本数逼近比例；组是不可分割的最小单位
        if dev_count < target_dev - groups[name] / 2:
            result[name] = "dev"
            dev_count += groups[name]
        else:
            result[name] = "test"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="整理手机拍摄的原始素材为规范数据集")
    parser.add_argument("--inbox", default="data/inbox", help="原始素材目录（默认 data/inbox）")
    parser.add_argument("--out", default="data/raw", help="输出图片目录（默认 data/raw）")
    parser.add_argument("--fps", type=float, default=1.0, help="视频每秒抽帧数（默认 1）")
    parser.add_argument("--ratio", type=float, default=0.7, help="dev 集比例（默认 0.7）")
    parser.add_argument("--max-side", type=int, default=1600, help="图片最长边（默认 1600）")
    parser.add_argument("--seed", type=int, default=20261016, help="划分随机种子，固定才能复现")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写任何文件")
    args = parser.parse_args()

    inbox = PROJECT_ROOT / args.inbox
    out_dir = PROJECT_ROOT / args.out

    # 输出目录必须在项目内：标注 CSV 里的路径是相对项目根目录写的，
    # 放到项目外会变成绝对路径，换台电脑就跑不动了。
    try:
        out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        print(f"错误：输出目录必须位于项目内，当前为 {out_dir}")
        print("      建议使用默认的 data/raw。")
        sys.exit(1)

    images, videos = scan_media(inbox)
    if not images and not videos:
        print(f"未在 {args.inbox} 找到图片或视频。")
        print("请把手机拍的照片拷贝进这个目录（可以有子文件夹），然后重新运行。")
        print("支持格式：jpg / jpeg / png / bmp / webp / mp4 / avi / mov / mkv")
        sys.exit(1)

    print(f"扫描到 {len(images)} 张图片、{len(videos)} 段视频")

    # ---------- 1) 去重 ----------
    seen: Dict[str, str] = {}
    duplicates: List[Dict[str, str]] = []
    unique_images: List[Path] = []
    for path in images:
        digest = file_md5(path)
        if digest in seen:
            duplicates.append({"file": _rel(path), "same_as": seen[digest]})
            continue
        seen[digest] = _rel(path)
        unique_images.append(path)

    video_seen: Dict[str, str] = {}
    unique_videos: List[Path] = []
    for path in videos:
        digest = file_md5(path)
        if digest in video_seen:
            duplicates.append({"file": _rel(path), "same_as": video_seen[digest]})
            continue
        video_seen[digest] = _rel(path)
        unique_videos.append(path)

    print(f"去重后：{len(unique_images)} 张图片、{len(unique_videos)} 段视频"
          f"（去掉重复 {len(duplicates)} 个）")

    if args.dry_run:
        print("\n--- 干跑模式，未写任何文件 ---")
        for path in unique_images[:10]:
            print(f"  图片 {path.name}")
        for path in unique_videos[:5]:
            print(f"  视频 {path.name}（将按 {args.fps}/秒抽帧）")
        print(f"\n预计样本数：{len(unique_images)} 张图片"
              f" + {len(unique_videos)} 段视频的抽帧结果")
        return

    # ---------- 2) 归一化 + 编号 ----------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, str]] = []   # 每行一个样本，含 group 字段（稍后剔除）
    failures: List[str] = []
    seq = 0

    for path in unique_images:
        seq += 1
        sample_id = f"lab_{seq:04d}"
        dst = out_dir / f"{sample_id}.jpg"
        if not normalize_image(path, dst, args.max_side):
            failures.append(_rel(path))
            seq -= 1
            continue
        hint = keyword_hint(path.name)
        samples.append({
            "sample_id": sample_id,
            "file_path": _rel(dst),
            "group": sample_id,
            "notes": (f"来源 {path.name}；" + (f"文件名提示{hint}" if hint else "")).rstrip("；"),
        })

    for vidx, path in enumerate(unique_videos, 1):
        group = f"vid_{vidx:03d}"
        frames = extract_video_frames(path, out_dir / f"_tmp_{group}", args.fps, args.max_side)
        if not frames:
            failures.append(_rel(path))
            continue
        for frame in frames:
            seq += 1
            sample_id = f"{group}_{frame.stem.replace('frame_', 'f')}"
            dst = out_dir / f"{sample_id}.jpg"
            frame.replace(dst)
            samples.append({
                "sample_id": sample_id,
                "file_path": _rel(dst),
                "group": group,
                "notes": f"来自视频 {path.name}，同一视频的帧必须留在同一集合",
            })
        shutil.rmtree(out_dir / f"_tmp_{group}", ignore_errors=True)

    # ---------- 3) 以组为单位划分 ----------
    group_sizes: Dict[str, int] = defaultdict(int)
    for row in samples:
        group_sizes[row["group"]] += 1
    split_of = assign_splits(dict(group_sizes), args.ratio, args.seed)

    for row in samples:
        row["split"] = split_of[row["group"]]

    # ---------- 4) 写标注草稿 ----------
    draft_path = PROJECT_ROOT / "data" / "labels_draft.csv"
    with draft_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in samples:
            writer.writerow({
                **row,
                "risk_type": "",      # 必须人工填写，脚本不猜
                "severity": "",
                "evidence": "",
                "location": "",
            })

    # ---------- 5) 报告 ----------
    dev = sum(1 for r in samples if r["split"] == "dev")
    test = len(samples) - dev
    report = {
        "inbox": args.inbox,
        "total_samples": len(samples),
        "dev": dev,
        "test": test,
        "duplicates_removed": duplicates,
        "failures": failures,
        "groups": {g: {"size": n, "split": split_of[g]} for g, n in group_sizes.items()},
    }
    (PROJECT_ROOT / "data" / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n已生成 {len(samples)} 个样本：dev {dev} 个 / test {test} 个")
    if duplicates:
        print(f"去重 {len(duplicates)} 个（明细见 data/ingest_report.json）")
    if failures:
        print(f"处理失败 {len(failures)} 个：{', '.join(failures[:5])}")
    print(f"\n标注草稿：{draft_path.relative_to(PROJECT_ROOT)}")
    print("下一步：逐行填写 risk_type / severity / evidence / location。")
    print("  risk_type 取值：ppe_missing / reagent_management / flammable_near_heat /")
    print("                  food_or_drink / spill_or_obstruction / normal")
    print("  risk_type 留空的行不会被评测器计入，但也不会报错——记得填完。")


if __name__ == "__main__":
    main()
