"""仓库自检：确保「别人 clone 下来就能跑」所依赖的路径和文件真实存在。

这类测试的价值在于拦住一类很隐蔽的退化：
数据集重命名（labels_test.csv → web_labels_test.csv）之后，
代码能跑、测试能过，但 README 里的命令和 Web 界面的评测按钮全部失效。
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"


@pytest.fixture(scope="module")
def labels_path():
    sys.path.insert(0, str(APP_DIR))
    from server import resolve_labels_path  # noqa: E402

    return resolve_labels_path(PROJECT_ROOT)


def test_test_labels_exist(labels_path):
    """测试集标注必须存在，且能被 Web 服务自动找到。"""
    assert labels_path is not None, "data/ 下找不到测试集标注文件"
    assert labels_path.exists()
    assert labels_path.name in {"web_labels_test.csv", "labels_test.csv", "demo_labels_test.csv"}


def test_dev_labels_exist():
    """自进化用的开发集标注必须存在；测试集绝不能被当成开发集。"""
    assert (PROJECT_ROOT / "data" / "web_labels_dev.csv").exists()


def test_label_files_are_disjoint(labels_path):
    """dev / test 的 sample_id 不能有交集，否则自进化会污染测试集。"""
    import csv

    def ids(path: Path) -> set:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return {row["sample_id"].strip() for row in csv.DictReader(fh)}

    dev = ids(PROJECT_ROOT / "data" / "web_labels_dev.csv")
    test = ids(labels_path)
    assert dev and test
    assert not (dev & test), f"dev/test 有重叠样本：{sorted(dev & test)[:5]}"


def test_readme_data_paths_exist():
    """README 里以 data/ 开头、看起来是输入文件的路径必须真实存在。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    # 只看出现在命令/反引号里的 data/ 路径，跳过明显是产物的 reports/
    candidates = set(re.findall(r"data/[A-Za-z0-9_./-]+\.(?:csv|jpg|png|mp4)", readme))
    assert candidates, "README 里没有解析到任何 data/ 路径，正则可能失效"
    missing = [p for p in sorted(candidates) if not (PROJECT_ROOT / p).exists()]
    assert not missing, f"README 引用了不存在的文件：{missing}"


def test_synthetic_samples_present():
    """合成样本与真实照片都要随仓库提供，否则复现不了报告数字。"""
    synth = list((PROJECT_ROOT / "data" / "synth").glob("*.jpg"))
    web = list((PROJECT_ROOT / "data" / "web_raw").glob("*.jpg"))
    assert len(synth) >= 49, f"data/synth 只有 {len(synth)} 张"
    assert len(web) >= 35, f"data/web_raw 只有 {len(web)} 张"
