"""hscredit 命令行入口.

提供包初始化等显式系统配置命令。
"""

import argparse
from typing import Optional, Sequence

from .utils.fonts import FONT_NAME, install_bundled_font


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器."""
    parser = argparse.ArgumentParser(prog="python -m hscredit", description="hscredit 命令行工具")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="初始化 hscredit 运行环境")
    init_parser.add_argument("--force", action="store_true", help="强制覆盖已安装的内置字体")
    check_parser = subparsers.add_parser("check-environment", help="检查安装依赖和实际导入版本")
    check_parser.add_argument("--strict", action="store_true", help="环境异常时返回非零退出码")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 hscredit 命令行."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        try:
            destination, changed = install_bundled_font(force=args.force)
        except (OSError, RuntimeError) as exc:
            parser.exit(1, f"初始化失败：{exc}\n")

        action = "安装完成" if changed else "已经安装"
        print(f"字体“{FONT_NAME}”{action}：{destination}")
        print("如果 Excel 已经打开，请重启 Excel 后使用该字体。")
        return 0

    if args.command == "check-environment":
        import json
        from .utils.environment import check_environment

        result = check_environment()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(args.strict and not result["正常"])

    parser.error(f"未知命令：{args.command}")
    return 2
