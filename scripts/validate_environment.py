#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hscredit 环境验证脚本

使用方法:
    python scripts/validate_environment.py
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version >= (3, 8):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python版本过低: {version.major}.{version.minor}.{version.micro}"


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """检查依赖包"""
    dependencies = [
        ("numpy", "1.19.0"),
        ("pandas", "1.2.0"),
        ("scipy", "1.5.0"),
        ("sklearn", "0.24.0"),
        ("openpyxl", "3.0.0"),
        ("matplotlib", "3.3.0"),
        ("seaborn", "0.11.0"),
    ]
    
    results = []
    for pkg_name, min_version in dependencies:
        try:
            pkg = __import__(pkg_name)
            version = getattr(pkg, '__version__', 'unknown')
            status = "✅"
            
            # 简单版本检查
            if version != 'unknown':
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        status = "⚠️"
                except ImportError:
                    pass
            
            results.append((pkg_name, True, f"{version} (需要>={min_version})"))
        except ImportError:
            results.append((pkg_name, False, f"未安装 (需要>={min_version})"))
    
    return results


def check_module_imports() -> List[Tuple[str, bool, str]]:
    """检查模块导入"""
    # 添加项目路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    modules = [
        ("hscredit", "主模块"),
        ("hscredit.report", "报告模块"),
        ("hscredit.report.excel", "Excel报告"),
        ("hscredit.core.models", "模型模块"),
        ("hscredit.core.models.losses", "损失函数"),
    ]
    
    results = []
    for module_path, desc in modules:
        try:
            __import__(module_path)
            results.append((module_path, True, desc))
        except Exception as e:
            results.append((module_path, False, f"{desc}: {e}"))
    
    return results


def run_quick_test() -> Tuple[bool, str]:
    """运行快速功能测试"""
    try:
        import numpy as np
        import pandas as pd
        from hscredit.report.excel import ExcelWriter
        from hscredit.core.models import FocalLoss
        
        # 测试Excel写入
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        writer = ExcelWriter()
        ws = writer.get_sheet_by_name("Sheet")
        writer.insert_df2sheet(ws, df, 1, 1)
        
        # 测试损失函数
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        focal_loss = FocalLoss()
        loss = focal_loss(y_true, y_pred)
        
        return True, "所有功能测试通过"
    except Exception as e:
        return False, f"功能测试失败: {e}"


def main():
    """主函数"""
    print("=" * 60)
    print("hscredit 环境验证")
    print("=" * 60)
    print()
    
    # 1. 检查Python版本
    print("1. Python版本检查")
    print("-" * 60)
    success, message = check_python_version()
    symbol = "✅" if success else "❌"
    print(f"  {symbol} {message}")
    if not success:
        print("\n❌ Python版本不满足要求，需要Python 3.8+")
        sys.exit(1)
    print()
    
    # 2. 检查依赖
    print("2. 依赖包检查")
    print("-" * 60)
    dep_results = check_dependencies()
    all_deps_ok = True
    for pkg_name, installed, version_info in dep_results:
        symbol = "✅" if installed else "❌"
        print(f"  {symbol} {pkg_name:15s} {version_info}")
        if not installed:
            all_deps_ok = False
    print()
    
    if not all_deps_ok:
        print("⚠️  部分依赖未安装，请运行: pip install -r requirements.txt")
        print()
    
    # 3. 检查模块导入
    print("3. 模块导入检查")
    print("-" * 60)
    import_results = check_module_imports()
    all_imports_ok = True
    for module_path, imported, info in import_results:
        symbol = "✅" if imported else "❌"
        print(f"  {symbol} {module_path:40s} {info}")
        if not imported:
            all_imports_ok = False
    print()
    
    if not all_imports_ok:
        print("❌ 部分模块导入失败，请检查代码")
        sys.exit(1)
    
    # 4. 快速功能测试
    print("4. 快速功能测试")
    print("-" * 60)
    success, message = run_quick_test()
    symbol = "✅" if success else "❌"
    print(f"  {symbol} {message}")
    print()
    
    # 5. 总结
    print("=" * 60)
    if all_deps_ok and all_imports_ok and success:
        print("✅ 环境验证通过")
        print()
        print("下一步:")
        print("  1. 运行 jupyter notebook")
        print("  2. 打开 examples/00_project_overview.ipynb")
        print("  3. 执行notebook进行详细验证")
    else:
        print("❌ 环境验证失败")
        print()
        print("请检查以上错误信息并修复")
    print("=" * 60)


if __name__ == "__main__":
    main()
