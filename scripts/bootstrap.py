"""组员首次拉取仓库后的环境初始化。

新成员 clone 完仓库，只要跑这一条命令：

    python scripts/bootstrap.py

它会依次完成：
  1. 检查 Python 版本
  2. 建 .venv 虚拟环境（已存在则复用）
  3. 装 requirements.txt 里的依赖
  4. 从 .env.example 复制出 .env（不覆盖已有文件）
  5. 跑单元测试，确认代码本身没问题
  6. 打印当前环境状态（会告诉你 API Key 填了没）

不联网、不写任何密钥——Key 由每个人自己填，见 docs/模型接入说明.md。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 9)


def _venv_python(venv_dir: Path) -> Path:
    """返回虚拟环境里解释器的路径（Windows 与 POSIX 不同）。"""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check, text=True)


def check_python() -> bool:
    step("1/6 检查 Python 版本")
    version = sys.version_info
    print(f"  当前：Python {version.major}.{version.minor}.{version.micro}")
    if (version.major, version.minor) < MIN_PYTHON:
        print(f"  ✗ 版本过低，需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 或更高。")
        return False
    print("  ✓ 版本满足要求")
    return True


def ensure_venv(force: bool = False) -> Path | None:
    step("2/6 准备虚拟环境 .venv")
    venv_dir = PROJECT_ROOT / ".venv"
    py = _venv_python(venv_dir)

    if py.exists() and not force:
        print(f"  ✓ 已存在，直接复用：{py}")
        return py

    if venv_dir.exists() and force:
        print("  删除旧的 .venv ……")
        shutil.rmtree(venv_dir, ignore_errors=True)

    print("  创建虚拟环境（约 10 秒）……")
    try:
        run([sys.executable, "-m", "venv", str(venv_dir)], PROJECT_ROOT)
    except subprocess.CalledProcessError as err:
        print(f"  ✗ 创建失败：{err}")
        print("    Linux 上可能需要先装 python3-venv：sudo apt install python3-venv")
        return None

    if not py.exists():
        print(f"  ✗ 没找到解释器：{py}")
        return None
    print(f"  ✓ 已创建：{py}")
    return py


def install_deps(py: Path) -> bool:
    step("3/6 安装依赖")
    try:
        run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], PROJECT_ROOT)
        run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"], PROJECT_ROOT)
    except subprocess.CalledProcessError as err:
        print(f"  ✗ 安装失败：{err}")
        print("    国内网络建议加镜像：")
        print(f"    {py} -m pip install -r requirements.txt "
              f"-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    print("  ✓ 依赖就绪")
    return True


def ensure_env_file() -> None:
    step("4/6 准备配置文件 .env")
    env_file = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print("  ✓ .env 已存在，未改动（避免覆盖你已填的 Key）")
        return
    if not example.exists():
        print("  ! 没有 .env.example，跳过")
        return

    shutil.copyfile(example, env_file)
    print("  ✓ 已从 .env.example 复制出 .env")
    print("  → 现在需要你自己填 API Key：")
    print(f"     {env_file}")
    print("     打开它，把 SAFETY_LLM_API_KEY= 后面填上你自己的 Key")
    print("     申请地址：https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys")


def run_tests(py: Path) -> bool:
    step("5/6 跑单元测试")
    try:
        result = subprocess.run(
            [str(py), "-m", "pytest", "tests", "-q"],
            cwd=str(PROJECT_ROOT), text=True, capture_output=True,
        )
    except FileNotFoundError:
        print("  ! 没有 pytest，跳过测试")
        return False

    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-6:])
    print(tail)
    if result.returncode == 0:
        print("  ✓ 测试通过，代码没问题")
        return True
    print("  ✗ 测试失败，先别往下做，把上面这段报错发给负责人")
    return False


def _has_api_key() -> bool:
    """.env 里是否已经填了非空的 Key（只判断有没有，不读值）。"""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False
    try:
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("SAFETY_LLM_API_KEY="):
                return bool(line.split("=", 1)[1].strip())
    except OSError:
        return False
    return False


def show_status(py: Path) -> None:
    step("6/6 环境状态")
    # 显式捕获再打印：直接继承 stdout 时，输出经管道会被缓冲，容易丢内容或串行
    result = subprocess.run(
        [str(py), "src/main.py", "status"],
        cwd=str(PROJECT_ROOT), text=True, capture_output=True,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        print(out)
    if result.returncode != 0:
        print(f"  ! status 执行失败（退出码 {result.returncode}）")
        if err:
            print(err[-800:])


def main() -> None:
    parser = argparse.ArgumentParser(description="组员首次初始化环境")
    parser.add_argument("--recreate-venv", action="store_true",
                        help="删掉已有 .venv 重建（依赖装坏时用）")
    parser.add_argument("--skip-tests", action="store_true", help="跳过单元测试")
    args = parser.parse_args()

    print("实验室危险场景识别 Agent —— 新成员环境初始化")
    print(f"项目目录：{PROJECT_ROOT}")

    if not check_python():
        sys.exit(1)

    py = ensure_venv(force=args.recreate_venv)
    if py is None:
        sys.exit(1)

    if not install_deps(py):
        sys.exit(1)

    ensure_env_file()

    if not args.skip_tests:
        run_tests(py)

    show_status(py)

    step("完成")
    print("  当前解释器（以后跑脚本都用它）：")
    print(f"    {py}")
    print("\n  下一步：")
    todos = []
    if not _has_api_key():
        todos.append("填 .env 里的 SAFETY_LLM_API_KEY（每人用自己的 Key，别共用）")
    todos.append("读 docs/执行清单.md 里「零、先做这两件事」")
    todos.append("跑 python src/main.py analyze --input <某张图> --mock 验证链路")
    for i, item in enumerate(todos, 1):
        print(f"    {i}. {item}")


if __name__ == "__main__":
    main()
