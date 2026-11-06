#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容旧版安装工具的 setuptools 入口。

项目元数据、依赖和可选依赖统一维护在 pyproject.toml 中。
"""

from pathlib import Path
import runpy

from setuptools import setup


root = Path(__file__).resolve().parent
readme_tools = runpy.run_path(str(root / "scripts" / "build_pypi_readme.py"))
setup(
    long_description=readme_tools["render_readme"](
        (root / "README.md").read_text(encoding="utf-8"),
        readme_tools["project_version"](),
    ),
    long_description_content_type="text/markdown",
)
