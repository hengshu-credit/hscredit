#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hscredit 安装脚本

支持开发模式安装:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# 读取requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="hscredit",
    version="0.1.0",
    description="金融风控建模工具包 - 完整的评分卡建模解决方案",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="hscredit team",
    author_email="hscredit@hengshucredit.com",
    url="https://github.com/hscredit/hscredit",
    license="MIT",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.4.0"],
        "lightgbm": ["lightgbm>=3.2.0"],
        "catboost": ["catboost>=1.0.0"],
        "deep-learning": ["torch>=1.8.0", "pytorch-tabnet>=3.1"],
        "pmml": ["sklearn2pmml>=0.82.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "": ["*.xlsx", "*.json", "*.yaml"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="credit risk scorecard modeling finance machine-learning woe binning",
    zip_safe=False,
)
