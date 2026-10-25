"""pytest 公共配置：把 src 目录加入导入路径。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def isolate_tool_log(tmp_path, monkeypatch):
    """把测试期间的工具调用日志重定向到临时目录。

    否则每跑一次 pytest，仓库里的 reports/tool_calls.jsonl 就会多出一堆
    mock 条目，`git status` 永远是脏的，提交上去的实验日志也失去意义。
    """
    monkeypatch.setenv("SAFETY_TOOL_LOG", str(tmp_path / "tool_calls.jsonl"))
