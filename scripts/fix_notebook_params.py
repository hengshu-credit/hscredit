#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复notebook中的insert_df2sheet参数错误

错误用法: insert_df2sheet(ws, df, start_row=1, start_col=1)
正确用法: insert_df2sheet(ws, df, (1, 1))
"""

import re
from pathlib import Path

def fix_insert_df2sheet(content: str) -> str:
    """修复insert_df2sheet的参数"""
    
    # 模式1: start_row=X, start_col=Y
    # 替换为: (X, Y)
    pattern1 = r'insert_df2sheet\(\s*(\w+),\s*(\w+),\s*start_row\s*=\s*(\d+),\s*start_col\s*=\s*(\d+)([^)]*)\)'
    
    def replace_func1(match):
        worksheet = match.group(1)
        df = match.group(2)
        row = match.group(3)
        col = match.group(4)
        other_params = match.group(5)
        
        return f'insert_df2sheet({worksheet}, {df}, ({row}, {col}){other_params})'
    
    content = re.sub(pattern1, replace_func1, content)
    
    # 模式2: merge_header=True 需要改为 merge=True
    content = content.replace('merge_header=True', 'merge=True')
    content = content.replace('merge_header=False', 'merge=False')
    
    return content

# 修复00_project_overview.ipynb
file_path = Path('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/00_project_overview.ipynb')
if file_path.exists():
    content = file_path.read_text(encoding='utf-8')
    fixed_content = fix_insert_df2sheet(content)
    file_path.write_text(fixed_content, encoding='utf-8')
    print(f"✅ 已修复: {file_path.name}")

# 修复01_excel_writer_validation.ipynb
file_path = Path('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/01_excel_writer_validation.ipynb')
if file_path.exists():
    content = file_path.read_text(encoding='utf-8')
    fixed_content = fix_insert_df2sheet(content)
    file_path.write_text(fixed_content, encoding='utf-8')
    print(f"✅ 已修复: {file_path.name}")

print("\n修复完成！")
