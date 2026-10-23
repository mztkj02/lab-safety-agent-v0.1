"""数据集导入脚本的解析逻辑测试。

重点覆盖 Roboflow YOLO 导出里 `data.yaml` 的两种类名写法，
以及「有哪些 PPE」到「缺哪类 PPE」的推导规则。
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fetch_lab_dataset import (  # noqa: E402
    derive_risks,
    find_dataset_root,
    parse_yaml_names,
    read_yolo_labels,
)

EXPECTED = ["Mask", "Gloves", "Googles", "Lab coat"]


@pytest.mark.parametrize(
    "label,text",
    [
        (
            "行内列表",
            "train: ../train/images\n"
            "names: ['Mask', 'Gloves', 'Googles', 'Lab coat']\n",
        ),
        (
            "块状映射",
            "names:\n"
            "  0: Mask\n"
            "  1: Gloves\n"
            "  2: Googles\n"
            "  3: Lab coat\n",
        ),
        (
            "块状映射带引号与后续配置",
            "names:\n"
            "  0: 'Mask'\n"
            "  1: 'Gloves'\n"
            "  2: 'Googles'\n"
            "  3: 'Lab coat'\n"
            "nc: 4\n",
        ),
        (
            "Roboflow 典型导出",
            "train: ../train/images\n"
            "val: ../valid/images\n"
            "nc: 4\n"
            "names: ['Mask', 'Gloves', 'Googles', 'Lab coat']\n"
            "roboflow:\n"
            "  workspace: uaeu-oc33o\n"
            "  project: safety-lab\n",
        ),
    ],
)
def test_parse_yaml_names_variants(tmp_path, label, text):
    path = tmp_path / "data.yaml"
    path.write_text(text, encoding="utf-8")
    assert parse_yaml_names(path) == EXPECTED, f"{label} 解析失败"


def test_parse_yaml_names_crlf(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_bytes(
        ("names:\r\n"
         "  0: Mask\r\n"
         "  1: Gloves\r\n"
         "  2: Googles\r\n"
         "  3: Lab coat\r\n").encode("utf-8")
    )
    assert parse_yaml_names(path) == EXPECTED


def test_parse_yaml_names_missing(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("train: ../train/images\nnc: 4\n", encoding="utf-8")
    assert parse_yaml_names(path) == []


def test_derive_risks_only_labcoat():
    """只标注到实验服 → 应推出缺护目镜、缺手套、缺口罩三项。"""
    risks, present = derive_risks(Counter({3: 2}), EXPECTED)
    types = [r for r, _ in risks]
    assert types == [
        "ppe_goggle_missing",
        "ppe_glove_missing",
        "ppe_mask_missing",
    ]
    assert present == {"ppe_labcoat_missing"}


def test_derive_risks_all_present_is_normal():
    risks, present = derive_risks(Counter({0: 1, 1: 1, 2: 1, 3: 1}), EXPECTED)
    assert risks == []
    assert len(present) == 4


def test_derive_risks_no_ppe_boxes_is_unjudgeable():
    """一个 PPE 框都没有 → 无法判断，不产生风险也不产生 normal。"""
    risks, present = derive_risks(Counter(), EXPECTED)
    assert risks == []
    assert present == set()


def test_derive_risks_unknown_class_ignored():
    """未映射的类别（比如 Person）不应被当成 PPE。"""
    risks, present = derive_risks(Counter({9: 3}), EXPECTED[:4])
    assert risks == []
    assert present == set()


def test_read_yolo_labels(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n3 0.6 0.6 0.2 0.2\n",
                   encoding="utf-8")
    counts = read_yolo_labels(txt)
    assert counts[0] == 2
    assert counts[3] == 1


def test_read_yolo_labels_missing_file(tmp_path):
    assert read_yolo_labels(tmp_path / "nope.txt") == Counter()


def test_find_dataset_root(tmp_path):
    nested = tmp_path / "Safety-Lab-1"
    nested.mkdir()
    (nested / "data.yaml").write_text("names: ['Mask']\n", encoding="utf-8")
    data_yaml, root = find_dataset_root(tmp_path)
    assert data_yaml.name == "data.yaml"
    assert root == nested
