"""从仓库 README 生成带代码和原图的功能展示页。"""

import re
from html import escape
from pathlib import Path
from urllib.parse import urlsplit


# 保留 README 和已发布文档中指向各组图表的链接。
GALLERIES = {
    "为什么选 hscredit？": "gallery-reports",
    "分箱摘要、客群比较与二维分箱表": "gallery-bin-tables",
    "分箱图与跨期趋势": "gallery-bin-charts",
    "特征编码": "gallery-encoders",
    "特征筛选": "gallery-selectors",
    "风控建模": "gallery-scorecard",
    "手工拆分与多数据集策略报告": "gallery-trees",
    "常用图表": "gallery-diagnostics",
}


def _doc_link(match, root):
    attribute, target = match.groups()
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not parts.path:
        return match.group()
    if target.startswith("docs/assets/readme/"):
        target = target.replace("docs/assets/readme/", "_static/", 1)
    elif target.startswith("docs/"):
        target = target[5:].replace(".rst", ".html").replace(".md", ".html")
    else:
        kind = "tree" if (root / parts.path).is_dir() else "blob"
        target = f"https://github.com/hengshu-credit/hscredit/{kind}/main/{target}"
    return f'{attribute}="{target}"'


def _section(match):
    level, title = match.groups()
    anchor = GALLERIES.get(title)
    heading = f"{level} {title}"
    return f"({anchor})=\n{heading}" if anchor else heading


def _expand_readme(app, docname, source):
    if docname != "showcase":
        return
    path = Path(app.srcdir).parent / "README.md"
    app.env.note_dependency(str(path))
    readme = path.read_text(encoding="utf-8")
    body = readme[re.search(r"^## ", readme, re.M).start():]
    # 代码示例保留原文，路径和展示格式只在正文中转换。
    chunks = re.split(r"(^```[^\n]*\n[\s\S]*?^```[ \t]*$)", body, flags=re.M)
    for i in range(0, len(chunks), 2):
        text = re.sub(r"^<a href=\"#[^\n]+$|^<a href=\"https://hscredit\.hengshucredit\.com/showcase\.html#gallery-[^\n]+$", "", chunks[i], flags=re.M)
        text = re.sub(r"</?details>\s*", "", text)
        text = re.sub(r"<summary>(.*?)</summary>", lambda m: f"#### {m[1]}" if m[1] in GALLERIES else f"**{m[1]}**", text)
        text = re.sub(r"^(#{2,4}) (.+)$", _section, text, flags=re.M)
        text = re.sub(
            r"!\[([^\]]*)\]\((docs/assets/readme/[^)]+)\)",
            lambda m: f'<p><a href="{m[2]}"><img src="{m[2]}" alt="{escape(m[1], quote=True)}"></a></p>',
            text,
        )
        chunks[i] = re.sub(r'\b(href|src)="([^"]+)"', lambda m: _doc_link(m, path.parent), text)
    source[0] += "\n" + "".join(chunks)


def setup(app):
    app.connect("source-read", _expand_readme)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
