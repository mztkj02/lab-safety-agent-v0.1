"""`.env` 配置加载的测试。

重点验证三件事：
1. 常见写法都能解析（注释、空行、export、引号、等号在值里）
2. 已存在的环境变量不会被文件覆盖（临时覆盖能力）
3. `.env.example` 能真的被 git 跟踪（否则组员不知道怎么配）
"""

import os
import subprocess
from pathlib import Path

import pytest

from llm import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# 这是注释，应被忽略\n"
        "\n"
        "SAFETY_LLM_API_KEY=abc.def\n"
        "export SAFETY_LLM_MODEL=glm-4v-flash\n"
        'SAFETY_LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"\n'
        "SAFETY_LLM_IMAGE_FORMAT='base64'\n"
        "  SAFETY_LLM_TIMEOUT = 90  \n"
        "非法行没有等号\n"
        "=没有键\n"
        "1INVALID=x\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """每个测试前清空相关变量，避免相互污染。"""
    for key in ("SAFETY_LLM_API_KEY", "SAFETY_LLM_MODEL", "SAFETY_LLM_BASE_URL",
                "SAFETY_LLM_IMAGE_FORMAT", "SAFETY_LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)


def test_parses_common_styles(env_file):
    assert load_env_file(env_file) == 5
    assert os.environ["SAFETY_LLM_API_KEY"] == "abc.def"
    assert os.environ["SAFETY_LLM_MODEL"] == "glm-4v-flash"       # export 前缀
    assert os.environ["SAFETY_LLM_BASE_URL"] == "https://open.bigmodel.cn/api/paas/v4"  # 双引号
    assert os.environ["SAFETY_LLM_IMAGE_FORMAT"] == "base64"      # 单引号
    assert os.environ["SAFETY_LLM_TIMEOUT"] == "90"               # 两侧空格


def test_ignores_invalid_lines(env_file):
    load_env_file(env_file)
    assert "非法行没有等号" not in os.environ
    assert "" not in os.environ
    assert "1INVALID" not in os.environ


def test_existing_env_wins(env_file, monkeypatch):
    """已有环境变量优先，方便临时覆盖而不改文件。"""
    monkeypatch.setenv("SAFETY_LLM_MODEL", "qwen-vl-max")
    load_env_file(env_file)
    assert os.environ["SAFETY_LLM_MODEL"] == "qwen-vl-max"


def test_missing_file_is_noop(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == 0


def test_crlf_and_bom(tmp_path):
    """Windows 记事本保存的 .env 带 BOM 和 CRLF，必须能读。"""
    path = tmp_path / ".env"
    path.write_bytes(
        "\ufeffSAFETY_LLM_API_KEY=crlf.key\r\nSAFETY_LLM_MODEL=glm-4v-flash\r\n".encode("utf-8")
    )
    assert load_env_file(path) == 2
    assert os.environ["SAFETY_LLM_API_KEY"] == "crlf.key"
    assert os.environ["SAFETY_LLM_MODEL"] == "glm-4v-flash"


def test_example_file_is_trackable():
    """.gitignore 里的 `.env.*` 不能把模板一起忽略掉。"""
    example = PROJECT_ROOT / ".env.example"
    assert example.exists(), "缺少 .env.example，组员不知道要配哪些字段"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(example)],
        cwd=PROJECT_ROOT, capture_output=True,
    )
    # 返回码 0 表示被忽略 —— 那就是错的
    assert result.returncode != 0, ".env.example 被 .gitignore 忽略了，需要加 !.env.example"
