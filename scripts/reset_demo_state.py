"""把项目重置回干净的起点。

用途：跑完占位数据集的演示后，把自进化产生的策略版本、纠错案例、实验报告清掉，
      避免把这些模拟数据当成真实实验记录提交。

用法：
    python scripts/reset_demo_state.py            # 查看会删什么（不实际删除）
    python scripts/reset_demo_state.py --yes      # 确认执行
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 要清理的目录（相对项目根）
TARGETS = [
    "versions",
    "reports/sessions",
    "reports/experiments",
    "reports/smoke",
    "reports/video_test",
    "reports/uploads",
    "reports/frames",
    "data/demo",
    "data/demo_video",
]

#: 要清空的文件（保留文件本身，清空成初始内容）
RESET_FILES = {
    "memory/cases.json": "[]\n",
    "reports/tool_calls.jsonl": "",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="重置演示状态")
    parser.add_argument("--yes", action="store_true", help="确认执行删除")
    parser.add_argument("--keep-strategy", action="store_true",
                        help="保留 configs/strategy_current.json 不动（默认会把 v0.2 恢复为当前版本）")
    args = parser.parse_args()

    print("将清理以下内容：")
    for target in TARGETS:
        path = PROJECT_ROOT / target
        mark = "存在" if path.exists() else "不存在"
        print(f"  - {target}（{mark}）")
    for target in RESET_FILES:
        print(f"  - 清空 {target}")
    if not args.keep_strategy:
        print("  - 把 configs/strategy_v0.2.json 恢复为 configs/strategy_current.json")

    if not args.yes:
        print("\n这是预演模式，未做任何改动。确认后请加 --yes 重新执行。")
        return

    for target in TARGETS:
        path = PROJECT_ROOT / target
        if path.exists():
            shutil.rmtree(path)
            print(f"已删除 {target}")

    for target, content in RESET_FILES.items():
        path = PROJECT_ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"已重置 {target}")

    if not args.keep_strategy:
        base = PROJECT_ROOT / "configs" / "strategy_v0.2.json"
        if base.exists():
            shutil.copyfile(base, PROJECT_ROOT / "configs" / "strategy_current.json")
            print("已把 v0.2 恢复为当前生效策略")

    # 保留基线版本 v0.1 / v0.2，保证 `rollback --version v0.1` 这类演示仍然可用
    versions_dir = PROJECT_ROOT / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    for name in ("strategy_v0.1.json", "strategy_v0.2.json"):
        source = PROJECT_ROOT / "configs" / name
        if source.exists():
            shutil.copyfile(source, versions_dir / name)
            print(f"已保留基线版本 versions/{name}")

    print("\n重置完成。现在可以用真实数据重新开始。")
    sys.exit(0)


if __name__ == "__main__":
    main()
