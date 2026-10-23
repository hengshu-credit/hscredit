"""统一 autodoc 参数值的行内代码与列表格式，仅作用于文档构建。"""

import re

_FIELD = re.compile(r"^(\s*):(\w+)(?:\s+[^:]+)?:\s*")
_BULLET = re.compile(r"^\s*[-+*]\s+")
# 优先识别完整 RST 标记，避免把角色、链接、数学公式中的内容再次包装。
_INLINE = re.compile(
    r"(?P<role>:[\w:]+:`[^`]+`)"
    r"|(?P<literal>``[^`]+``)"
    r"|(?P<link>`[^`]+`_+)"
    r"|(?P<markdown>(?<![\\`])`[^`\n]+`(?!`))"
    r"|(?P<quoted>(?<![A-Za-z0-9_\\])(?:'[^'\n]*'|\"[^\"\n]*\"))"
    r"|(?P<constant>(?<![A-Za-z0-9_])(?:True|False|None)(?![A-Za-z0-9_]))"
)
_CODE_DIRECTIVE = re.compile(r"\s*\.\.\s+(?:code(?:-block)?|sourcecode|parsed-literal|math|raw)::")


def _format_values(text):
    """将单个参数说明中的值统一为 RST literal，保留引用与复杂表达式。"""

    def replace(match):
        kind = match.lastgroup
        value = match.group()
        if kind in {"role", "link"}:
            return value
        if kind == "literal":
            value = value[2:-2].strip()
        elif kind == "markdown":
            value = value[1:-1].strip()
        # 单个字符串值统一使用单引号；含引号或转义的表达式保持原样。
        if re.fullmatch(r'"[^\'"\\]*"', value):
            value = "'" + value[1:-1] + "'"
        # 中文正文与 literal 之间也要留空白，否则 docutils 可能不识别边界。
        before = " " if match.start() and not text[match.start() - 1].isspace() else ""
        after = " " if match.end() < len(text) and not text[match.end()].isspace() else ""
        return before + "``" + value + "``" + after

    return _INLINE.sub(replace, text)


def normalize_parameter_docs(lines):
    """规范参数字段及其枚举列表，跳过示例代码、其他字段和显式 RST 指令。"""
    out = []
    parameter_indent = None
    literal_indent = None
    doctest = False
    table = False
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.match(r"(?:=+\s+)+=+$|\+(?:[-=]+\+)+$", stripped):
            table = True
        if table:
            out.append(line)
            if not stripped:
                table = False
            continue
        if literal_indent is not None:
            if not stripped or indent > literal_indent:
                out.append(line)
                continue
            literal_indent = None
        if doctest:
            if stripped:
                out.append(line)
                continue
            doctest = False
        if stripped.startswith(">>>"):
            doctest = True
            out.append(line)
            continue
        if _CODE_DIRECTIVE.match(line) or line.rstrip().endswith("::"):
            literal_indent = indent
            out.append(line)
            continue
        field = _FIELD.match(line)
        is_parameter = field and field[2] in {"param", "parameter", "arg", "argument", "keyword"}
        if (is_parameter or parameter_indent is not None) and line.count("``") % 2:
            # RST 行内代码允许跨行。先合并完整 literal，防止逐行处理时将
            # 字典/表达式内部的引号再包装成嵌套反引号。
            base_indent = indent if is_parameter else parameter_indent
            end = index
            joined = line
            while end < len(lines) and joined.count("``") % 2:
                following = lines[end]
                if not following.strip() or len(following) - len(following.lstrip()) <= base_indent:
                    break
                joined += " " + following.strip()
                end += 1
            if not joined.count("``") % 2:
                line, index = joined, end
        if field:
            parameter_indent = indent if is_parameter else None
            if parameter_indent is not None:
                line = line[: field.end()] + _format_values(line[field.end() :])
        elif stripped and parameter_indent is not None and indent <= parameter_indent:
            parameter_indent = None
        elif parameter_indent is not None and not stripped.startswith(".. "):
            line = _format_values(line)
        if parameter_indent is not None and _BULLET.match(line) and out and out[-1].strip():
            # RST 列表须与前文分隔；逐项分隔也兼容多段说明和嵌套列表。
            out.append("")
        out.append(line)
    return out
