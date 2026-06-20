# -*- coding: utf-8 -*-
"""hscredit 文档 Sphinx 配置.

构建命令::

    cd docs && make html        # 生成 docs/_build/html/index.html

依赖（已在 pyproject.toml 的 [docs] 可选依赖中声明）::

    pip install hscredit[docs]
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# -- 路径设置 ---------------------------------------------------------------
# 将项目根目录加入 sys.path，使 autodoc 能够 import hscredit。
sys.path.insert(0, os.path.abspath(".."))

import hscredit  # noqa: E402  (置于 sys.path 设置之后)

# -- 项目信息 ---------------------------------------------------------------
project = "hscredit"
author = "hscredit team"
copyright = f"{datetime.now():%Y}, {author}"
version = hscredit.__version__
release = hscredit.__version__

# -- 通用配置 ---------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # 从 docstring 自动生成 API 文档
    "sphinx.ext.autosummary",   # 生成 API 汇总表
    "sphinx.ext.napoleon",      # 支持 Google / NumPy 风格 docstring
    "sphinx.ext.viewcode",      # 在文档中链接到源码
    "sphinx.ext.intersphinx",   # 跨项目引用（numpy/pandas/sklearn）
    "sphinx.ext.mathjax",       # 数学公式
    "sphinx.ext.todo",
    "myst_parser",              # 支持 Markdown（.md）文档
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "PACKAGING.md"]

# 中文界面
language = "zh_CN"

# 同时支持 .rst 与 .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
root_doc = "index"

# -- autodoc 配置 -----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "undoc-members": False,
}
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
autoclass_content = "class"
# docstring 已用 RST field（:param:）书写，无需 napoleon 的 param 转换，
# 但保留 napoleon 以兼容个别 Google/NumPy 段落。
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# 可选依赖在当前环境可能缺失，mock 掉以保证文档构建不依赖其安装。
autodoc_mock_imports = [
    "ngboost",
    "shap",
    "sklearn2pmml",
    "sklearn_pandas",
    "pypmml",
    "optunahub",
    "optuna_dashboard",
]

# 抑制重复 import 造成的告警，避免 -W 模式下因聚合 __all__ 重复定义而失败。
suppress_warnings = ["autosummary", "duplicate_object"]

# -- intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
# 离线构建时无法下载 inventory，禁用超时阻塞。
intersphinx_timeout = 1
intersphinx_disabled_reftypes = ["*"]

# -- HTML 输出 --------------------------------------------------------------
# RTD 的信息密度和分层导航适合 hscredit 的大型 API 文档；品牌识别由模板与
# custom.css 完成。该依赖已在 pyproject.toml 的 docs 可选依赖中声明。
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
    # 衡枢真信品牌：深靛蓝导航头，使霓虹 Logo 发光（具体渐变与发光在 custom.css 中）
    "style_nav_header_background": "#140e35",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
}
html_static_path = ["_static"]
html_title = f"hscredit {release} 文档"
html_short_title = "hscredit"
# 品牌 Logo（hengshucredit 动态霓虹 SVG，蓝紫粉同步渐变）与同款 favicon
html_logo = "_static/hengshucredit_logo.svg"
html_favicon = "_static/hengshucredit_logo.svg"
html_css_files = ["custom.css"]
html_js_files = ["sidebar-nav.js", "copy-code.js"]

# 文档只引用包内的官方字体源，构建时同步到 Sphinx 静态目录，避免维护两份字体。
_BRAND_FONT_SOURCE = Path(__file__).resolve().parents[1] / "hscredit" / "resources" / "fonts" / "font.ttf"


def _sync_brand_font(app, exception):
    """将 hscredit 官方字体直接同步到 HTML 输出目录。"""
    if exception is not None or app.builder.format != "html":
        return
    if not _BRAND_FONT_SOURCE.is_file():
        raise FileNotFoundError(f"衡枢品牌字体不存在: {_BRAND_FONT_SOURCE}")
    target = Path(app.outdir) / "_static" / "fonts" / "font.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_BRAND_FONT_SOURCE, target)

# -- todo -------------------------------------------------------------------
todo_include_todos = False


# -- docstring 预处理 -------------------------------------------------------
# 项目的 docstring 采用「RST 字段（:param:）+ Markdown 表格 + sklearn 风格属性名」
# 混排风格。其中：
#   1. Markdown 表格（| a | b |\n|---|---|）不是合法 RST，会渲染为带竖线的乱码；
#   2. 形如 ``splits_`` / ``n_bins_`` 的 fitted 属性名以下划线结尾，RST 会将其误判为
#      超链接引用，产生大量 “Unknown target name” 报错。
# 这里在 autodoc 读取 docstring 后、交给 RST 解析前做一次预处理：把 Markdown 表格转成
# RST list-table，把结尾下划线的标识符包裹为行内代码。预处理只作用于文档构建，不修改
# 源码，且跳过 doctest（>>>）与指令行，避免破坏代码示例。
import re  # noqa: E402

_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_TRAILING_UNDERSCORE = re.compile(r"(?<![\w`\\])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_)(?![\w`])")


def _md_table_cells(line, ncols=None):
    """切分 Markdown 表格行。

    部分单元格内含绝对值记号 ``|x|``，朴素按 ``|`` 切分会过度切分。当已知列数
    ``ncols`` 时，将超出的片段合并回最后一列，保证与表头列数一致。
    """
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if ncols is not None and len(cells) > ncols:
        cells = cells[: ncols - 1] + ["|".join(cells[ncols - 1:]).strip()]
    if ncols is not None and len(cells) < ncols:
        cells = cells + [""] * (ncols - len(cells))
    # list-table 单元格内残留的 ``|`` 仍会被当作替换引用，转义为字面竖线。
    return [c.replace("|", r"\|") for c in cells]


def _convert_md_tables(lines):
    """将 docstring 中的 GitHub 风格 Markdown 表格转换为 RST list-table。"""
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        if _MD_TABLE_ROW.match(line) and i + 1 < n and _MD_TABLE_SEP.match(lines[i + 1]):
            indent = line[: len(line) - len(line.lstrip())]
            header = _md_table_cells(line)
            ncols = len(header)
            j = i + 2
            rows = []
            while j < n and _MD_TABLE_ROW.match(lines[j]) and not _MD_TABLE_SEP.match(lines[j]):
                rows.append(_md_table_cells(lines[j], ncols=ncols))
                j += 1
            out.append("")
            out.append(f"{indent}.. list-table::")
            out.append(f"{indent}   :header-rows: 1")
            out.append("")
            for row in [header] + rows:
                out.append(f"{indent}   * - {row[0]}")
                for cell in row[1:]:
                    out.append(f"{indent}     - {cell}")
            out.append("")
            i = j
        else:
            out.append(line)
            i += 1
    return out


def _wrap_trailing_underscore(lines):
    """把结尾下划线的标识符（sklearn fitted 属性）包裹为行内代码，避免被当作引用。"""
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith((">>>", "...", ".. ")) or _MD_TABLE_ROW.match(line):
            out.append(line)
        else:
            out.append(_TRAILING_UNDERSCORE.sub(r"``\1``", line))
    return out


def _process_docstring(app, what, name, obj, options, lines):
    new_lines = _wrap_trailing_underscore(_convert_md_tables(lines))
    lines[:] = new_lines


# -- autodoc 钩子 -----------------------------------------------------------
def _skip_sklearn_plumbing(app, what, name, obj, skip, options):
    """跳过 sklearn 注入的元数据路由方法。

    新版 sklearn 会在 ``BaseEstimator`` 子类上动态生成 ``set_*_request`` /
    ``get_metadata_routing`` 等方法，其 docstring 引用了 sklearn 文档中的
    ``metadata_routing`` 标签。这些方法对 hscredit 用户无实际意义，且会在离线
    构建时产生大量 “undefined label” 告警，统一跳过。
    """
    if name == "get_metadata_routing" or (name.startswith("set_") and name.endswith("_request")):
        return True
    return skip


def setup(app):
    app.connect("build-finished", _sync_brand_font)
    app.connect("autodoc-skip-member", _skip_sklearn_plumbing)
    app.connect("autodoc-process-docstring", _process_docstring)
