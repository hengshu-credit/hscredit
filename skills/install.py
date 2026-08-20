"""从当前仓库安装已实现的 hscredit Skills。"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence


IMPLEMENTED_SKILLS = ("hsbin", "hsreport")


def _default_target() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
    return base / "skills"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安装 hscredit Agent Skills")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite", choices=["hscredit"], help="安装当前已实现的整套 Skill")
    selection.add_argument("--skills", nargs="+", choices=IMPLEMENTED_SKILLS, help="安装指定 Skill")
    parser.add_argument("--target-dir", default=str(_default_target()), help="Agent Skills 目标目录")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    selected = IMPLEMENTED_SKILLS if args.suite else tuple(dict.fromkeys(args.skills))
    source_root = Path(__file__).resolve().parent
    target_root = Path(args.target_dir).expanduser().resolve()
    sources = [(name, (source_root / name).resolve()) for name in selected]
    destinations = [(name, (target_root / name).resolve()) for name in selected]

    for name, source in sources:
        if source.parent != source_root or not (source / "SKILL.md").is_file():
            print(f"Skill 源目录无效：{name}", file=sys.stderr)
            return 1
    for name, destination in destinations:
        if destination.parent != target_root:
            print(f"Skill 目标目录越界：{name}", file=sys.stderr)
            return 1
        if destination.exists():
            print(f"Skill 目标目录已存在，未执行覆盖：{destination}", file=sys.stderr)
            return 1

    target_root.mkdir(parents=True, exist_ok=True)
    for (name, source), (_, destination) in zip(sources, destinations):
        shutil.copytree(source, destination)
        print(f"已安装 {name}：{destination}")
    print("Skill 将在 Agent 的下一轮对话中可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
