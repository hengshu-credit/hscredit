# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HSCredit'
copyright = '2026, HSCredit Team'
author = 'HSCredit Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

extensions = [
    # Sphinx核心扩展
    'sphinx.ext.autodoc',          # 自动从docstring生成文档
    'sphinx.ext.autosummary',      # 自动生成API摘要
    'sphinx.ext.viewcode',         # 添加源代码链接
    'sphinx.ext.napoleon',         # 支持Google和NumPy风格的docstring
    'sphinx.ext.intersphinx',      # 链接到其他项目的文档
    'sphinx.ext.coverage',         # 文档覆盖率检查
    'sphinx.ext.githubpages',      # GitHub Pages支持

    # Markdown支持
    'myst_parser',                 # Markdown解析器

    # 其他扩展
    'sphinx_copybutton',           # 代码复制按钮
]

# 模板路径
templates_path = ['_templates']

# 排除的文件模式
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# 源文件后缀
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# 主语言
language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 主题配置
html_theme = 'pydata_sphinx_theme'  # 现代化的主题，适合数据科学项目

# 强制使用暗黑模式
html_context = {
    "theme_toggle_state": "dark",
}

html_theme_options = {
    # 导航栏配置
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],

    # 侧边栏配置
    "sidebar_exclude_pages": ["search"],
    "show_prev_next": True,

    # 主题颜色 - 科幻霓虹风格
    "pygment_light_style": "monokai",
    "pygment_dark_style": "monokai",

    # 强制暗黑模式
    "default_mode": "dark",
    "dark_theme": "pydata_sphinx_theme",

    # 页脚配置
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],

    # Logo和图标 - 使用横竖科技logo
    "logo": {
        "text": "HSCredit",
        "image_light": "https://hengshucredit.com/images/hengshucredit_animated.svg",
        "image_dark": "https://hengshucredit.com/images/hengshucredit_animated.svg",
    },

    # 导航链接
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hscredit/hscredit",
            "icon": "fab fa-github",
        },
    ],

    # 搜索配置
    "search_bar_text": "搜索文档...",

    # 其他选项
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "announcement": "HSCredit 信用评分卡建模工具 | 科幻主题文档",

    # 配色方案 - 科幻霓虹风格
    "coloraccent": "#00ffff",
    "gray_600": "#6c757d",
    "graydark": "#212529",
}

# 上下文配置（用于编辑页面按钮）
html_context = {
    "github_user": "hscredit",
    "github_repo": "hscredit",
    "github_version": "main",
    "doc_path": "docs/sphinx",
}

# 静态文件路径
html_static_path = ['_static']

# CSS文件
html_css_files = [
    'custom.css',
]

# JavaScript文件
html_js_files = [
    'custom.js',
]

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# 自动生成API文档
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Napoleon配置（支持Google和NumPy风格的docstring）
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary配置
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# -- Options for MyST parser -------------------------------------------------
# https://myst-parser.readthedocs.io/

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_title_to_header = True

# -- 自定义配置 --------------------------------------------------------------

# 添加自定义的CSS和JS
def setup(app):
    """自定义Sphinx设置"""
    app.add_css_file('custom.css')
    app.add_js_file('custom.js')
