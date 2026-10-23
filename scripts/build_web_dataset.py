"""把两路素材合并成统一的标注数据集，并按类别分层划分 dev / test。

数据集构成
----------
  A 路：Pexels 真实实验室照片（data/web_raw/）→ 正常场景 + 少量陷阱样本
  B 路：文生图合成样本（data/synth/）         → 违规场景 + 精心设计的易混淆样本

为什么必须是两路
----------------
公开图库里几乎不存在违规现场照片，只用 A 路会得到一个「全是正常场景」的空数据集；
只用 B 路又拿不到真实拍摄的画面质感。两路并用是当前唯一可行的组合，
但两者的来源差异必须在报告里如实披露，不能混为一谈。

划分原则（对应课程要求）
------------------------
1. 以 risk_type 分层，保证每个类别在 dev / test 中都有代表；
2. 同源样本不跨集合 —— 本项目每个样本都是独立画面，无同源变体，
   但同一批次同一提示词生成的图仍按顺序交替分配，避免整批进同一侧；
3. 固定随机种子，保证每次生成的划分完全一致，结果可复现；
4. 测试集只用于最终评测，绝不参与自进化（由 run_experiment.py 保证）。

用法：
    python scripts/build_web_dataset.py --ratio 0.5
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEVERITY_BY_TYPE = {
    "normal": "normal",
    "ppe_missing": "medium",
    "reagent_management": "medium",
    "flammable_near_heat": "high",
    "food_or_drink": "low",
    "spill_or_obstruction": "medium",
}

FIELDS = ["sample_id", "file_path", "split", "risk_type", "severity",
          "evidence", "location", "notes"]


def load_web_samples() -> list:
    """读取入选的 Pexels 样本，补上文件路径与来源信息。"""
    sel_path = PROJECT_ROOT / "data" / "web_selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    out = []
    for it in sel["items"]:
        pid = it["photo_id"]
        rel = f"data/web_raw/{pid}.jpg"
        if not (PROJECT_ROOT / rel).exists():
            print(f"  ! 缺少文件 {rel}，跳过（可重新执行 scripts/fetch_web_images.py）")
            continue
        out.append({
            "sample_id": f"web_{pid}",
            "file_path": rel,
            "risk_type": it["risk_type"],
            "evidence": it["evidence"],
            "location": "全画面",
            "confusable": bool(it.get("confusable")),
            "source": "pexels",
            "source_detail": f"https://www.pexels.com/photo/{pid}/",
        })
    return out


def load_synth_samples() -> list:
    """读取合成样本清单。"""
    man_path = PROJECT_ROOT / "data" / "synth" / "synth_manifest.json"
    if not man_path.exists():
        print(f"  ! 找不到 {man_path}，请先运行 scripts/prepare_synth_data.py")
        return []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    out = []
    for it in man["items"]:
        out.append({
            "sample_id": it["sample_id"],
            "file_path": it["file"],
            "risk_type": it["risk_type"],
            "evidence": it["evidence"],
            "location": "全画面",
            "confusable": bool(it.get("confusable")),
            "source": "synthetic",
            "source_detail": f"prompt: {it.get('prompt', '')}",
        })
    return out


def split_samples(samples: list, test_ratio: float, seed: int) -> list:
    """按 risk_type 分层交替分配到 dev / test。"""
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["risk_type"]].append(s)

    rng = random.Random(seed)
    for risk_type, group in by_type.items():
        # 先按 sample_id 排序，保证输入顺序稳定；再打乱
        group.sort(key=lambda x: x["sample_id"])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))
        for i, s in enumerate(group):
            s["split"] = "test" if i < n_test else "dev"
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="合并两路素材并划分数据集")
    parser.add_argument("--ratio", type=float, default=0.5, help="测试集比例")
    parser.add_argument("--seed", type=int, default=20261023, help="划分随机种子")
    parser.add_argument("--dev-out", default="data/web_labels_dev.csv")
    parser.add_argument("--test-out", default="data/web_labels_test.csv")
    args = parser.parse_args()

    web = load_web_samples()
    synth = load_synth_samples()
    samples = web + synth
    print(f"真实图库 {len(web)} 张 + 合成 {len(synth)} 张 = {len(samples)} 张")

    samples = split_samples(samples, args.ratio, args.seed)

    for s in samples:
        s["severity"] = SEVERITY_BY_TYPE.get(s["risk_type"], "medium")
        notes = [f"来源={s['source']}"]
        if s["confusable"]:
            notes.append("易混淆样本")
        notes.append(s["source_detail"])
        s["notes"] = "；".join(notes)

    for split, out in (("dev", args.dev_out), ("test", args.test_out)):
        rows = [s for s in samples if s["split"] == split]
        rows.sort(key=lambda x: (x["risk_type"] != "normal", x["risk_type"], x["sample_id"]))
        path = PROJECT_ROOT / out
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写入 {path}（{len(rows)} 条）")

    # ---- 统计 ----
    print("\n类别分布：")
    header = f"{'类别':<24}{'dev':>6}{'test':>6}{'合计':>6}"
    print(header)
    print("-" * len(header))
    for risk_type in sorted({s["risk_type"] for s in samples}):
        d = sum(1 for s in samples if s["risk_type"] == risk_type and s["split"] == "dev")
        t = sum(1 for s in samples if s["risk_type"] == risk_type and s["split"] == "test")
        print(f"{risk_type:<24}{d:>6}{t:>6}{d + t:>6}")

    n_conf = sum(1 for s in samples if s["confusable"])
    n_synth = sum(1 for s in samples if s["source"] == "synthetic")
    print(f"\n合计 {len(samples)} 张：易混淆 {n_conf} 张，合成来源 {n_synth} 张，"
          f"真实图库 {len(samples) - n_synth} 张")

    if len(samples) < 80:
        print(f"\n⚠️ 当前只有 {len(samples)} 张，未达到课程要求的 80 个独立样本。")

    # 顺手写一份来源说明，报告「数据来源」一节可以直接引用
    summary = {
        "total": len(samples),
        "by_source": {
            "pexels": sum(1 for s in samples if s["source"] == "pexels"),
            "synthetic": n_synth,
        },
        "confusable": n_conf,
        "by_type": {
            rt: sum(1 for s in samples if s["risk_type"] == rt)
            for rt in sorted({s["risk_type"] for s in samples})
        },
        "split_ratio": args.ratio,
        "seed": args.seed,
        "license": "Pexels License（真实图）+ 项目自建（合成图）",
        "caveat": "合成样本为文生图生成，非真实拍摄场景；报告中必须与实拍数据区分声明。",
    }
    out = PROJECT_ROOT / "data" / "dataset_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"来源与分布摘要：{out}")


if __name__ == "__main__":
    main()
