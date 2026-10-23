"""数据集导入的关键正确性测试。

重点不是「能不能跑」，而是两条会毁掉实验有效性的规则：

1. **同一视频的帧不能跨 dev/test** —— 否则测试集里出现训练时见过的画面，
   指标虚高，而且课程明确禁止；
2. **划分必须可复现** —— 换台电脑、重跑一次结果要一样，否则对比实验不可信。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ingest_photos import assign_splits, keyword_hint  # noqa: E402


def test_no_group_spans_both_splits():
    """核心约束：一个组（一张图 / 一段视频的全部帧）只能落在一个集合。"""
    groups = {f"vid_{i:03d}": 8 for i in range(10)}      # 10 段视频，每段 8 帧
    groups.update({f"lab_{i:04d}": 1 for i in range(30)})  # 30 张单张照片
    split_of = assign_splits(groups, ratio=0.7)

    assert set(split_of) == set(groups), "每个组都必须被分配"
    assert set(split_of.values()) == {"dev", "test"}

    # 同一组的成员天然共享同一个 split —— 这里验证的是比例合理且两种集合都出现
    dev = sum(n for g, n in groups.items() if split_of[g] == "dev")
    ratio = dev / sum(groups.values())
    # 不能精确到 0.70（组不可分割），但也不该离谱
    assert 0.45 <= ratio <= 0.95, f"dev 占比 {ratio:.2f} 偏离过大"


def test_deterministic_with_same_seed():
    groups = {f"g{i}": 1 for i in range(50)}
    a = assign_splits(groups, 0.7, seed=42)
    b = assign_splits(groups, 0.7, seed=42)
    assert a == b, "同一种子必须得到完全相同的划分"


def test_different_seed_changes_split():
    groups = {f"g{i}": 1 for i in range(50)}
    a = assign_splits(groups, 0.7, seed=1)
    b = assign_splits(groups, 0.7, seed=2)
    assert a != b, "不同种子应产生不同划分，否则说明种子没生效"


def test_large_group_does_not_break_ratio():
    """一段长视频（40 帧）不能被切成两半去凑比例。"""
    groups = {"long_video": 40}
    groups.update({f"lab_{i}": 1 for i in range(20)})
    split_of = assign_splits(groups, ratio=0.7)
    # 该视频整体归入某一个集合，两个集合都不为空
    assert split_of["long_video"] in ("dev", "test")
    assert set(split_of.values()) == {"dev", "test"}


def test_all_groups_assigned_even_when_tiny():
    split_of = assign_splits({"only_one": 1}, ratio=0.7)
    assert list(split_of.values()) == ["dev"]


def test_keyword_hint_only_suggests_never_asserts():
    """文件名提示只能写进 notes，绝不能自动变成标注。"""
    assert "ppe_missing" in keyword_hint("IMG_0001_no_goggle.jpg")
    assert keyword_hint("IMG_20261016_123456.jpg") == ""
    assert keyword_hint("random.jpg") == ""


def test_labels_draft_schema_matches_template():
    """草稿列必须与 labels_template.csv 一致，否则评测器读不到。"""
    import csv
    template = PROJECT_ROOT / "data" / "labels_template.csv"
    assert template.exists()
    with template.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    from ingest_photos import LABEL_FIELDS
    assert header == LABEL_FIELDS
