"""修正人工标注中的错误（2026-10-25 复核）。

背景：用户质疑 `8532831.jpg` 是否真的戴了护目镜。逐张放大眼睛区域核对后发现
两张图的标注把「未佩戴护目镜」写成了「佩戴护目镜」，真值也因此从 normal 错标。

这个脚本按 sample_id 定点修正，其余行原样保留（不做全表重写，避免误伤）。
运行前后都会打印行数与改动项，便于核对。

    python scripts/fix_labels.py --dry-run   # 先看要改什么
    python scripts/fix_labels.py             # 真正写入
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

#: sample_id -> 新标注。只列需要改的字段，未列的保持原值。
FIXES = {
    "web_8532831": {
        "risk_type": "ppe_missing",
        "severity": "medium",
        "evidence": "两名人员在台前操作，均穿白色防护服但均未佩戴护目镜；"
                    "桌面上放着一副黄色镜框护目镜未被使用，一人低头观察显微镜、另一人在旁配合",
        "notes_note": "2026-10-25 复核修正：原标注 normal 且证据写「佩戴护目镜」，"
                      "放大核对后实际两人均未佩戴（护目镜搁在桌面），改判为 ppe_missing",
    },
    "web_8851786": {
        "risk_type": "ppe_missing",
        "severity": "medium",
        "evidence": "两名人员穿白大褂、戴深色手套做滴定操作，但均未佩戴护目镜"
                    "（左侧人员戴的是普通黑框眼镜，不构成实验眼部防护）",
        "notes_note": "2026-10-25 复核修正：原标注 normal 且证据写「佩戴护目镜」，"
                      "放大核对后确认为普通眼镜/未佩戴，改判为 ppe_missing",
    },
    # 这张原本就写明「眼部为普通眼镜而非护目镜」却判了 normal，
    # 与 synth_ppe_missing_01（只戴普通近视镜、未戴护目镜 → ppe_missing）
    # 的标准矛盾。统一到 ppe_missing 的定义（未佩戴 PPE 即风险）。
    "web_37951908": {
        "risk_type": "ppe_missing",
        "severity": "medium",
        "evidence": "实验员穿白大褂举锥形瓶观察紫色液体，戴的是普通黑框近视镜而非护目镜，"
                    "且未戴手套，不满足眼部与手部防护要求",
        "notes_note": "2026-10-25 复核修正：原标注为 normal（备注「边界样本，人工判为正常」），"
                      "与 synth_ppe_missing_01 同类情形判罚不一致，按 ppe_missing 定义统一改为风险样本",
    },
}

FILES = ["web_labels_dev.csv", "web_labels_test.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description="定点修正标注错误")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写入")
    args = parser.parse_args()

    total_changed = 0
    for name in FILES:
        path = DATA / name
        if not path.exists():
            print(f"! 不存在：{path}")
            continue

        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        before_count = len(rows)
        changed = []

        for row in rows:
            fix = FIXES.get(row.get("sample_id", ""))
            if not fix:
                continue
            old = (row["risk_type"], row["severity"])
            row["risk_type"] = fix["risk_type"]
            row["severity"] = fix["severity"]
            row["evidence"] = fix["evidence"]
            note = row.get("notes", "") or ""
            if fix["notes_note"] not in note:
                row["notes"] = (note + "；" if note else "") + fix["notes_note"]
            changed.append((row["sample_id"], old, (row["risk_type"], row["severity"])))

        if not changed:
            print(f"{name}: 无需修改（{before_count} 行）")
            continue

        print(f"{name}: {before_count} 行，修改 {len(changed)} 条")
        for sid, old, new in changed:
            print(f"   {sid}: {old[0]}/{old[1]} -> {new[0]}/{new[1]}")
        total_changed += len(changed)

        if args.dry_run:
            continue

        backup = path.with_suffix(".csv.bak")
        shutil.copyfile(path, backup)
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            after_count = len(list(csv.DictReader(fh)))
        print(f"   已写入，行数 {before_count} -> {after_count}，备份 {backup.name}")

    print(f"\n共修改 {total_changed} 条" + ("（dry-run，未写入）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
