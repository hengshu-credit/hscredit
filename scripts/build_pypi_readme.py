"""构建时在内存中转换 README 资源链接，不生成或维护第二份文档。"""

import ast
from pathlib import Path
import re
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "hengshu-credit/hscredit"


def project_version():
    """从版本字面量读取版本号，避免构建期导入运行依赖。"""
    module = ast.parse((ROOT / "hscredit" / "__init__.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("未找到项目版本号 __version__")


def render_readme(source, version):
    """只改资源链接，不改变正文、代码、结果表或图片内容。"""
    tag = "v" + version

    def absolute(target, image=False):
        target = target.strip()
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            return target
        path = parsed.path.removeprefix("./")
        encoded = quote(path, safe="/")
        if image or parsed.path.lower().endswith(".xlsx"):
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{tag}/{encoded}"
        else:
            kind = "tree" if (ROOT / path).is_dir() else "blob"
            url = f"https://github.com/{REPOSITORY}/{kind}/{tag}/{encoded}"
        if parsed.query:
            url += "?" + parsed.query
        if parsed.fragment:
            url += "#" + quote(parsed.fragment, safe="-_")
        return url

    def markdown_link(match):
        prefix, target = match.group(1), match.group(2)
        return prefix + absolute(target, image=prefix.startswith("!")) + ")"

    def html_link(match):
        attribute, delimiter, target = match.group(1), match.group(2), match.group(3)
        return attribute + "=" + delimiter + absolute(target, image=attribute == "src") + delimiter

    # 保留所有 fenced code block 原文，包括其中的文件路径。
    pieces = re.split(r"(```[^\n]*\n.*?```)", source, flags=re.S)
    for index in range(0, len(pieces), 2):
        piece = re.sub(r"(!?\[[^\]\n]*\]\()([^\)\n]+)\)", markdown_link, pieces[index])
        pieces[index] = re.sub(r"""\b(href|src)=(["'])(.*?)\2""", html_link, piece)
    return "".join(pieces)
