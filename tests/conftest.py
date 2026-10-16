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
