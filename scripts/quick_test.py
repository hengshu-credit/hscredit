#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证模块导入
使用sys.path.insert，无需安装包
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("hscredit 模块导入测试")
print("=" * 60)
print(f"\n项目路径: {project_root}")
print(f"路径存在: {project_root.exists()}")
print()

# 测试模块导入
modules = [
    ("hscredit", "主模块"),
    ("hscredit.report", "报告模块"),
    ("hscredit.report.excel", "Excel报告"),
    ("hscredit.report.excel.writer", "Excel写入器"),
    ("hscredit.core.models", "模型模块"),
    ("hscredit.core.models.losses", "损失函数"),
    ("hscredit.core.models.losses.base", "损失基类"),
    ("hscredit.core.models.losses.focal_loss", "Focal Loss"),
    ("hscredit.core.models.losses.weighted_loss", "加权损失"),
    ("hscredit.core.models.losses.risk_loss", "风控损失"),
    ("hscredit.core.models.losses.custom_metrics", "自定义指标"),
    ("hscredit.core.models.losses.adapters", "框架适配器"),
]

success_count = 0
for module_path, desc in modules:
    try:
        module = __import__(module_path, fromlist=[''])
        print(f"✅ {module_path:40s} - {desc}")
        success_count += 1
    except Exception as e:
        print(f"❌ {module_path:40s} - {desc}")
        print(f"   错误: {e}")

print()
print("-" * 60)
print(f"导入成功: {success_count}/{len(modules)}")
print("-" * 60)

if success_count == len(modules):
    print()
    print("🎉 所有模块导入成功！")
    print()
    print("已实现的功能:")
    print("  1. Excel报告生成 (hscredit.report.excel)")
    print("  2. 自定义损失函数 (hscredit.core.models)")
    print()
    print("可以开始使用Jupyter Notebook验证:")
    print("  jupyter notebook examples/00_project_overview.ipynb")
else:
    print()
    print("⚠️ 部分模块导入失败")
    print("请检查:")
    print("  1. 项目路径是否正确")
    print("  2. __init__.py文件是否正确配置")
    print("  3. 是否修改了代码后需要重启Jupyter kernel")

print("=" * 60)
