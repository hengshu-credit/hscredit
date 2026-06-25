"""支持通过 ``python -m hscredit`` 运行命令行工具."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
