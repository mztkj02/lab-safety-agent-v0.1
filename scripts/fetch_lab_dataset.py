"""从 Roboflow Universe 获取公开实验室 PPE 数据集，转换成本项目的标注格式。

为什么要这个脚本
----------------
课程要求「≥80 个独立样本 + 人工标注 + dev/test 划分」。公开数据集给的是
**目标检测框**（每张图有哪些 PPE），而本项目评测需要的是**图像级风险标签**
（这张图缺了哪一类防护）。本脚本负责这层转换，并把结果落成
`data/lab_labels_dev.csv` / `data/lab_labels_test.csv`，字段与
`data/labels_template.csv` 完全一致，可直接喂给 `src/main.py evaluate`。

推荐数据集：UAEU 的 Safety Lab（Roboflow Universe）
  https://universe.roboflow.com/uaeu-oc33o/safety-lab
  427 张高校实验室图像，CC BY 4.0，含 Mask / Gloves / Googles / Lab coat 四类。

⚠️ 转换是**规则推导**，不是人工标注。推导结果必须由组员逐张复核确认，
   报告里要如实写「规则预标注 + 人工复核」，不能直接说成纯人工标注。

用法
----
  # 1) 直接下载并转换（需要 Roboflow 免费账号的 API Key）
  python scripts/fetch_lab_dataset.py --roboflow-key YOUR_KEY --out data/lab

  # 2) 已经在网页上手动下载了 zip，用本地文件转换
  python scripts/fetch_lab_dataset.py --zip ~/Downloads/safety-lab.zip --out data/lab

  # 3) 只看转换结果分布，不落盘
  python scripts/fetch_lab_dataset.py --zip x.zip --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Roboflow 项目坐标（可被命令行覆盖，便于换用别的数据集）
DEFAULT_WORKSPACE = "uaeu-oc33o"
DEFAULT_PROJECT = "safety-lab"
DEFAULT_VERSION = "1"

# 数据集类别名 -> 本项目风险类型
# 键为 Roboflow data.yaml 里的类名（大小写不敏感匹配），值为 (风险类型, 默认等级)
CLASS_TO_RISK = {
    "googles": ("ppe_goggle_missing", "medium"),
    "goggles": ("ppe_goggle_missing", "medium"),
    "gloves": ("ppe_glove_missing", "medium"),
    "mask": ("ppe_mask_missing", "low"),
    "lab coat": ("ppe_labcoat_missing", "high"),
    "labcoat": ("ppe_labcoat_missing", "high"),
}

# 每类 PPE 的中文名，用于生成 evidence 文本
RISK_CN = {
    "ppe_goggle_missing": "未佩戴护目镜",
    "ppe_glove_missing": "未佩戴防护手套",
    "ppe_mask_missing": "未佩戴口罩",
    "ppe_labcoat_missing": "未穿实验服",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------
def download_zip(workspace: str, project: str, version: str, api_key: str,
                 dest: Path, fmt: str = "yolov8") -> Path:
    """从 Roboflow 拉取数据集 zip。返回本地 zip 路径。"""
    api = (f"https://api.roboflow.com/{workspace}/{project}/{version}"
           f"/download/{fmt}?api_key={urllib.parse.quote(api_key)}")
    print(f"请求 Roboflow 导出：{workspace}/{project} v{version} ({fmt})")

    req = urllib.request.Request(api)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            head = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "ignore")[:400]
        if err.code in (401, 403):
            raise SystemExit(f"Roboflow 鉴权/权限失败（HTTP {err.code}）：{detail}\n"
                             f"请确认 API Key 有效，且该数据集是公开可下载的。") from err
        raise SystemExit(f"下载失败（HTTP {err.code}）：{detail}") from err
    except Exception as err:  # noqa: BLE001
        raise SystemExit(f"下载失败：{err}") from err

    dest.parent.mkdir(parents=True, exist_ok=True)

    # 返回可能是 JSON（含签名 URL），也可能直接是 zip 字节
    if "json" in head or raw[:1] == b"{":
        try:
            info = json.loads(raw.decode("utf-8"))
            url = info.get("url") or info.get("download", {}).get("url")
        except Exception:  # noqa: BLE001
            url = None
        if not url:
            raise SystemExit(f"Roboflow 返回了无法识别的内容：{raw[:300]!r}")
        print("按签名链接下载 zip ...")
        with urllib.request.urlopen(url, timeout=300) as resp:
            dest.write_bytes(resp.read())
    else:
        dest.write_bytes(raw)

    print(f"已保存 {dest}（{dest.stat().st_size / 1e6:.1f} MB）")
    return dest


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------
def parse_yaml_names(path: Path) -> list:
    """从 Roboflow 导出的 data.yaml 里抠出类名列表。

    只解锁两种写法，避免为了一个字段引入 PyYAML：
      names: ['Mask', 'Gloves']
      names:
        0: Mask
        1: Gloves
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    # 行内列表：names: ['Mask', 'Gloves']
    m = re.search(r"^names:[ \t]*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    if m:
        items = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        if items:
            return items

    # 块状映射：逐行扫，比正则稳
    #   names:
    #     0: Mask
    #     1: Gloves
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.match(r"^names:[ \t]*$", line):
            continue
        pairs = []
        for follow in lines[i + 1:]:
            if not follow.strip():
                continue
            if not follow[:1].isspace():   # 缩进结束，块到此为止
                break
            mm = re.match(r"^[ \t]*(\d+)[ \t]*:[ \t]*(.+?)[ \t]*$", follow)
            if mm:
                pairs.append((int(mm.group(1)), mm.group(2).strip().strip("'\"")))
        if pairs:
            return [name for _, name in sorted(pairs)]
    return []


def find_dataset_root(extract_dir: Path) -> tuple:
    """在解压结果里定位 data.yaml 与图片目录。返回 (data_yaml, root)。"""
    candidates = list(extract_dir.rglob("data.yaml")) + list(extract_dir.rglob("data.yml"))
    if not candidates:
        raise SystemExit(f"解压后没找到 data.yaml，目录结构可能不是 YOLO 格式：{extract_dir}")
    data_yaml = candidates[0]
    return data_yaml, data_yaml.parent


def read_yolo_labels(txt_path: Path) -> Counter:
    """读一张图的 YOLO 标注，返回 {类别索引: 框数量}。"""
    counts: Counter = Counter()
    if not txt_path.exists():
        return counts
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            counts[int(float(parts[0]))] += 1
        except (ValueError, TypeError):
            continue
    return counts


def derive_risks(class_counts: Counter, class_names: list) -> tuple:
    """把「图里有哪些 PPE」推导成「缺哪几类 PPE」。

    返回 (负责的风险列表, 命中的 PPE 集合)。
    规则：图里出现至少 1 个 PPE 框 → 认为画面中有人在做实验；
    四类 PPE 里缺哪类，就报对应的缺失风险；四类齐全 → 无风险。
    """
    present = set()
    for idx, _ in class_counts.items():
        if 0 <= idx < len(class_names):
            name = class_names[idx].strip().lower()
            key = CLASS_TO_RISK.get(name)
            if key:
                present.add(key[0])

    if not present:
        # 一个 PPE 框都没有：可能是空场景、也可能标注缺失，无法判断，交给上层丢弃
        return [], present

    risks = []
    for risk_type, severity in {
        "ppe_goggle_missing": "medium",
        "ppe_glove_missing": "medium",
        "ppe_mask_missing": "low",
        "ppe_labcoat_missing": "high",
    }.items():
        if risk_type not in present:
            risks.append((risk_type, severity))
    return risks, present


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="下载并转换 Roboflow 实验室 PPE 数据集")
    ap.add_argument("--roboflow-key", default=None, help="Roboflow API Key（与 --zip 二选一）")
    ap.add_argument("--zip", default=None, help="本地已下载的数据集 zip 路径")
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--format", default="yolov8", help="Roboflow 导出格式，默认 yolov8")
    ap.add_argument("--out", default="data/lab", help="样本输出目录")
    ap.add_argument("--dev-ratio", type=float, default=0.7, help="dev 集比例，默认 0.7")
    ap.add_argument("--seed", type=int, default=20261023, help="划分随机种子，固定以便复现")
    ap.add_argument("--min-boxes", type=int, default=1,
                    help="一张图至少要有几个标注框才纳入（默认 1，用于排除无标注图）")
    ap.add_argument("--max-per-risk", type=int, default=0,
                    help="每个风险类型最多保留多少张（0 表示不限，用于类别平衡）")
    ap.add_argument("--dry-run", action="store_true", help="只统计分布，不写文件")
    args = ap.parse_args()

    # ---- 1. 取 zip ----
    tmp_dir = PROJECT_ROOT / ".cache" / "dataset"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.zip:
        zip_path = Path(args.zip).expanduser()
        if not zip_path.exists():
            raise SystemExit(f"找不到 zip：{zip_path}")
    elif args.roboflow_key:
        zip_path = download_zip(args.workspace, args.project, args.version,
                                args.roboflow_key, tmp_dir / f"{args.project}.zip",
                                fmt=args.format)
    else:
        raise SystemExit("请提供 --roboflow-key 或 --zip。\n"
                         "数据集页面：https://universe.roboflow.com/uaeu-oc33o/safety-lab")

    # ---- 2. 解压 ----
    extract_dir = tmp_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    data_yaml, root = find_dataset_root(extract_dir)
    class_names = parse_yaml_names(data_yaml)
    if not class_names:
        raise SystemExit(f"没能从 {data_yaml} 解析出类名，请人工检查该文件。")
    print(f"数据集根目录：{root}")
    print(f"类别列表：{class_names}")

    unknown = [n for n in class_names if n.strip().lower() not in CLASS_TO_RISK]
    if unknown:
        print(f"⚠️ 以下类别没有映射规则，将被忽略：{unknown}")

    # ---- 3. 逐图推导 ----
    rows = []          # (原图路径, [风险类型...], 命中 PPE 集合, 框数)
    skipped_no_box = 0
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label_txt = img.with_suffix(".txt")
        counts = read_yolo_labels(label_txt)
        total_boxes = sum(counts.values())
        if total_boxes < args.min_boxes:
            skipped_no_box += 1
            continue
        risks, present = derive_risks(counts, class_names)
        if not present:
            skipped_no_box += 1
            continue
        rows.append((img, risks, present, total_boxes))

    print(f"\n可用图像：{len(rows)} 张（因无标注框跳过 {skipped_no_box} 张）")

    # ---- 4. 类别平衡 ----
    if args.max_per_risk > 0:
        random.seed(args.seed)
        random.shuffle(rows)
        kept, per_risk = [], Counter()
        for item in rows:
            risks = item[1] or ["normal"]
            # 该图所有风险类别都还有额度才保留，避免把一张图拆散
            if all(per_risk[r] < args.max_per_risk for r in risks):
                kept.append(item)
                for r in risks:
                    per_risk[r] += 1
        rows = kept
        print(f"按 --max-per-risk={args.max_per_risk} 平衡后保留 {len(rows)} 张")

    if not rows:
        raise SystemExit("没有可用的样本。请检查数据集或放宽 --min-boxes。")

    # ---- 5. 统计 ----
    risk_counter = Counter()
    multi = 0
    for _, risks, _, _ in rows:
        if not risks:
            risk_counter["normal"] += 1
        else:
            risk_counter.update(r for r, _ in risks)
            if len(risks) > 1:
                multi += 1
    print("\n=== 风险类型分布 ===")
    for name, cnt in risk_counter.most_common():
        print(f"  {name:<24} {cnt}")
    print(f"  多标签样本（同时缺多类 PPE）：{multi} 张")

    if args.dry_run:
        print("\n--dry-run：未写入任何文件。")
        return

    # ---- 6. 落盘 ----
    out_dir = PROJECT_ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    random.seed(args.seed)
    random.shuffle(rows)
    n_dev = int(len(rows) * args.dev_ratio)

    fields = ["sample_id", "file_path", "split", "risk_type", "severity",
              "evidence", "location", "notes"]
    csv_rows = defaultdict(list)

    for i, (img, risks, present, boxes) in enumerate(rows, start=1):
        split = "dev" if i <= n_dev else "test"
        sample_id = f"lab_{i:04d}"
        target = out_dir / f"{sample_id}{img.suffix.lower()}"
        shutil.copyfile(img, target)
        rel = str(target.relative_to(PROJECT_ROOT)).replace("\\", "/")

        if not risks:
            csv_rows[split].append({
                "sample_id": sample_id, "file_path": rel, "split": split,
                "risk_type": "normal", "severity": "normal",
                "evidence": "画面中四类 PPE 均有标注，未见防护缺失",
                "location": "全画面",
                "notes": "规则推导；需人工复核确认",
            })
        else:
            for risk_type, severity in risks:
                csv_rows[split].append({
                    "sample_id": sample_id, "file_path": rel, "split": split,
                    "risk_type": risk_type, "severity": severity,
                    "evidence": f"{RISK_CN.get(risk_type, risk_type)}；"
                                f"该图标注到 {len(present)} 类 PPE",
                    "location": "画面中人员",
                    "notes": "规则推导；需人工复核确认",
                })

    for split in ("dev", "test"):
        path = PROJECT_ROOT / "data" / f"lab_labels_{split}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows[split])
        print(f"已写入 {path}（{len(csv_rows[split])} 行）")

    print(f"\n样本目录：{out_dir}（{len(rows)} 张图）")
    print("下一步：")
    print("  1) 组员逐张复核标注，修正推导错误（这是课程要求的「人工标注」环节）")
    print("  2) python scripts/run_experiment.py --labels data/lab_labels_test.csv "
          "--dev-labels data/lab_labels_dev.csv --groups base,multi --auto-evolve --rounds 2")


if __name__ == "__main__":
    sys.exit(main())
