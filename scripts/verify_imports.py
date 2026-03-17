#!/usr/bin/env python
"""
验证所有示例文件的导入方式是否正确

检查所有Python脚本和notebook是否使用了sys.path.insert方式导入hscredit。
"""

import sys
from pathlib import Path
import json
import re

def check_python_file(filepath):
    """检查Python脚本的导入方式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否使用了sys.path.insert
    has_sys_path = 'sys.path.insert' in content
    
    # 检查是否导入了hscredit
    imports_hscredit = bool(re.search(r'^(?:from|import)\s+hscredit', content, re.MULTILINE))
    
    # 如果导入了hscredit，必须有sys.path.insert
    if imports_hscredit:
        return has_sys_path, "✅ 使用sys.path.insert" if has_sys_path else "❌ 缺少sys.path.insert"
    else:
        return True, "ℹ️  未导入hscredit"

def check_notebook(filepath):
    """检查Jupyter Notebook的导入方式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 检查所有代码单元格
    has_sys_path = False
    imports_hscredit = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'sys.path.insert' in source:
                has_sys_path = True
            if re.search(r'^(?:from|import)\s+hscredit', source, re.MULTILINE):
                imports_hscredit = True
    
    # 如果导入了hscredit，必须有sys.path.insert
    if imports_hscredit:
        return has_sys_path, "✅ 使用sys.path.insert" if has_sys_path else "❌ 缺少sys.path.insert"
    else:
        return True, "ℹ️  未导入hscredit"

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"
    
    print("=" * 70)
    print("hscredit 导入方式验证")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 检查Python脚本
    print("Python脚本检查:")
    print("-" * 70)
    for py_file in sorted(examples_dir.glob("*.py")):
        passed, message = check_python_file(py_file)
        print(f"  {py_file.name:30s} {message}")
        if not passed:
            all_passed = False
    
    print()
    
    # 检查Jupyter Notebook
    print("Jupyter Notebook检查:")
    print("-" * 70)
    for nb_file in sorted(examples_dir.glob("*.ipynb")):
        passed, message = check_notebook(nb_file)
        print(f"  {nb_file.name:30s} {message}")
        if not passed:
            all_passed = False
    
    print()
    
    # 检查测试文件（应该不使用sys.path.insert）
    print("测试文件检查:")
    print("-" * 70)
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        for test_file in sorted(tests_dir.glob("test_*.py")):
            passed, message = check_python_file(test_file)
            # 测试文件不应该使用sys.path.insert
            if 'sys.path.insert' in test_file.read_text():
                print(f"  {test_file.name:30s} ⚠️  测试文件不需要sys.path.insert")
            else:
                print(f"  {test_file.name:30s} ✅ 正确（pytest自动处理）")
    
    print()
    print("=" * 70)
    if all_passed:
        print("✅ 所有文件导入方式正确")
    else:
        print("❌ 部分文件需要修改")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
