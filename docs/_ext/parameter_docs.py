"""统一 autodoc 参数值的行内代码与列表格式，仅作用于文档构建。"""

import re

from docutils import nodes


def merge_api_field_lists(app, domain, objtype, contentnode):
    """合并同一 API 的参数字段，消除 autodoc 补入的只有类型的重复参数。

    部分 docstring 在示例之后补充参数。Sphinx 默认对每个顶层 field_list
    补入整个签名的类型，导致每个列表都包含其他列表中已说明的参数。
    只合并当前对象的直接子节点，不改变嵌套方法、说明框或示例中的字段。
    """
    if domain != "py":
        return
    field_lists = [child for child in contentnode if isinstance(child, nodes.field_list)]
    if not field_lists:
        return
    first = field_lists[0]
    for extra in field_lists[1:]:
        for field in list(extra):
            if field[0].astext().split()[0] in {"param", "parameter", "arg", "argument", "keyword", "type"}:
                extra.remove(field)
                first += field
    # class-doc-from: both 会同时读类和构造器的说明。同名字段仅展示一次，
    # 保留不同的补充段落，避免丢失构造器独有的约束或默认值说明。
    seen = {}
    for field_list in field_lists:
        for field in list(field_list):
            key = field[0].astext()
            if key not in seen:
                seen[key] = field
                continue
            if key.startswith("type ") or key == "rtype":
                # 显式类型与签名推导类型可能只是写法不同（Optional[T] / T | None）。
                # 类型字段只能有一个段落，否则 Sphinx 会丢掉类型展示。
                field_list.remove(field)
                continue
            previous = seen[key][1]
            existing = {child.astext() for child in previous}
            for child in list(field[1]):
                if child.astext() and child.astext() not in existing:
                    field[1].remove(child)
                    previous += child
                    existing.add(child.astext())
            field_list.remove(field)
        if len(field_list) == 0:
            contentnode.remove(field_list)


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
