"""通过 JSON 文件调用 hscredit Agent Skills。"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import execute_skill
from .errors import SkillExecutionError


def build_parser() -> argparse.ArgumentParser:
    """创建 Skills 运行时命令行解析器。"""
    parser = argparse.ArgumentParser(prog="python -m hscredit.skills_runtime")
    parser.add_argument("--skill", required=True, choices=["hsbin", "hsreport"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """读取请求、执行操作并输出单个 JSON 信封。"""
    args = build_parser().parse_args(argv)
    try:
        request_path = Path(args.request).expanduser().resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = execute_skill(args.skill, request)
    except SkillExecutionError as exc:
        print(json.dumps(exc.to_dict(debug=args.debug), ensure_ascii=False))
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"无法读取 Skill 请求：{exc}",
            field="request",
            cause=exc,
        )
        print(json.dumps(error.to_dict(debug=args.debug), ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
