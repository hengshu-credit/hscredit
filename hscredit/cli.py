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
    dashboard_parser = subparsers.add_parser("dashboard", help="启动使用 hscredit 配色和字体的 Optuna 看板")
    dashboard_parser.add_argument("storage", help="Optuna 存储地址，例如 sqlite:///hscredit.db")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    dashboard_parser.add_argument("--port", default=8080, type=int, help="监听端口，默认 8080")
    dashboard_parser.add_argument("--artifact-dir", help="可选的 Optuna 产物文件目录")
    dashboard_parser.add_argument("--quiet", action="store_true", help="隐藏 HTTP 请求日志")
    platform_parser = subparsers.add_parser("platform", help="启动 hscredit 风控平台")
    platform_parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    platform_parser.add_argument("--port", default=8765, type=int, help="端口，0 表示自动选择")
    platform_parser.add_argument("--data-dir", help="数据集与任务存储目录")
    platform_parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    platform_parser.add_argument("--demo", action="store_true", help="加载 hscredit_dataset 的统一样例字段（不抽样）")
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

    if args.command == "dashboard":
        from .dashboard import run_dashboard

        if not 1 <= args.port <= 65535:
            parser.error("端口必须在 1 到 65535 之间")
        try:
            run_dashboard(args.storage, host=args.host, port=args.port, artifact_dir=args.artifact_dir, quiet=args.quiet)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"启动看板失败：{exc}\n")
        return 0

    if args.command == "platform":
        import os
        from .platform import launch

        try:
            launch(host=args.host, port=args.port, data_dir=args.data_dir, open_browser=not args.no_browser, token=os.environ.get("HSCREDIT_PLATFORM_TOKEN"), demo=args.demo)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"启动风控平台失败：{exc}\n")
        return 0

    parser.error(f"未知命令：{args.command}")
    return 2
