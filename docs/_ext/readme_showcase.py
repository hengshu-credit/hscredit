"""从仓库 README 生成带代码和原图的功能展示页。"""

import re
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

# 正文菜单、侧栏和 README 图组链接共用这些章节锚点。
SECTION_ANCHORS = {
    "为什么选 hscredit？": "gallery-reports",
    "数据集": "showcase-dataset",
    "特征分箱": "showcase-binning",
    "分箱摘要、客群比较与二维分箱表": "gallery-bin-tables",
    "分箱图与跨期趋势": "gallery-bin-charts",
    "特征编码": "gallery-encoders",
    "特征筛选": "gallery-selectors",
    "风控建模": "gallery-scorecard",
    "规则分析": "showcase-rules",
    "数据挖掘": "showcase-eda",
    "Excel 写入": "showcase-excel",
    "数据库读写": "showcase-database",
    "Agent 能力": "showcase-skills",
    "手工拆分与多数据集策略报告": "gallery-trees",
    "常用图表": "gallery-diagnostics",
    "公共方法": "public-utils",
    "品牌色与语义配色": "brand-colors",
}


def _doc_link(match, root):
    attribute, target = match.groups()
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not parts.path:
        return match.group()
    if target.startswith("docs/assets/readme/"):
        target = target.replace("docs/assets/readme/", "_static/", 1)
    elif target.startswith("docs/assets/brand/"):
        target = target.replace("docs/assets/brand/", "_static/", 1)
    elif target.startswith("docs/"):
        target = target[5:].replace(".rst", ".html").replace(".md", ".html")
    else:
        kind = "tree" if (root / parts.path).is_dir() else "blob"
        target = f"https://github.com/hengshu-credit/hscredit/{kind}/main/{target}"
    return f'{attribute}="{target}"'


def _section(level, title):
    anchor = SECTION_ANCHORS.get(title)
    heading = f"{'#' * level} {title}"
    return f"({anchor})=\n{heading}" if anchor else heading


def _expand_readme(app, docname, source):
    if docname != "showcase":
        return
    path = Path(app.srcdir).parent / "README.md"
    app.env.note_dependency(str(path))
    readme = path.read_text(encoding="utf-8")
    body = readme[re.search(r"^## ", readme, re.M).start() :]
    # 代码示例保留原文，路径和展示格式只在正文中转换。
    chunks = re.split(r"(^```[^\n]*\n[\s\S]*?^```[ \t]*$)", body, flags=re.M)
    intro = []
    in_examples = False
    module = None
    for i in range(0, len(chunks), 2):
        lines = []
        for line in chunks[i].split("\n"):
            if line.startswith("## "):
                title = line[3:]
                in_examples = title == "功能展示"
                module = None
                if in_examples:
                    continue
                line = _section(2, title)
            elif in_examples and line.startswith("### "):
                module = line[4:]
                line = _section(2, module)
            elif in_examples and module is None:
                # 把 README 的模块快捷入口及配套 Notebook 放在文档页首。
                # 用 Sphinx 引用生成链接，避免中文版标题在两种渲染器中产生不同的 ID。
                if line.startswith('<a href="#'):
                    line = re.sub(r'<a href="#[^"]+"[^>]*>([^<]+)</a>', lambda m: f"{{ref}}`{m[1]} <{SECTION_ANCHORS[m[1]]}>`", line)
                intro.append(re.sub(r'\b(href|src)="([^"]+)"', lambda m: _doc_link(m, path.parent), line))
                continue
            elif line in ("<details>", "</details>") or line.startswith('<a href="https://hscredit.hengshucredit.com/showcase.html#gallery-'):
                continue
            elif line.startswith("<summary>"):
                title = line.removeprefix("<summary>").removesuffix("</summary>")
                line = _section(3, title) if module and module != "风控建模" else f"**{title}**"
            elif line.startswith("**") and module:
                title, _, description = line[2:].partition("**")
                title = title.rstrip("。")
                if module == "风控建模" or title.startswith(("单规则：", "规则集：", "多叉树组合策略汇总", "auto_feature_analysis：")):
                    level = 4 if title in ("分类模型预测逾期", "回归模型预测金额") else 3
                    line = _section(level, title)
                    if description.strip():
                        line += "\n\n" + description.lstrip(" ·")
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(
            r"!\[([^\]]*)\]\((docs/assets/readme/[^)]+)\)",
            lambda m: f'<p><a href="{m[2]}"><img src="{m[2]}" alt="{escape(m[1], quote=True)}"></a></p>',
            text,
        )
        chunks[i] = re.sub(r'\b(href|src)="([^"]+)"', lambda m: _doc_link(m, path.parent), text)
    source[0] += "\n" + "\n".join(intro).strip() + "\n\n" + "".join(chunks)


def setup(app):
    app.connect("source-read", _expand_readme)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
