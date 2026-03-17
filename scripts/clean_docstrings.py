#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清理和修正docstring格式."""

import re
from pathlib import Path


def clean_docstring(content: str) -> str:
    """清理docstring中的格式问题.
    
    Args:
        content: 文件内容
        
    Returns:
        清理后的内容
    """
    # 修复 np.: ndarray -> np.ndarray
    content = content.replace('np.: ndarray', 'np.ndarray')
    content = content.replace('pd.: DataFrame', 'pd.DataFrame')
    
    # 修复返回类型后面多余的冒号
    content = re.sub(r'(Callable):\s*$', r'\1', content, flags=re.MULTILINE)
    
    # 修复函数参数缺少 Args section 的问题
    # 如果函数有参数但没有 Args: section，添加一个
    lines = content.split('\n')
    result = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测函数定义
        if stripped.startswith('def ') and '(' in stripped:
            # 提取函数签名
            func_def = stripped
            j = i + 1
            
            # 收集多行函数定义
            while j < len(lines) and not lines[j].strip().startswith('"""') and not lines[j].strip().startswith("'''"):
                j += 1
            
            # 检查是否有docstring
            if j < len(lines) and (lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''")):
                quote = '"""' if '"""' in lines[j] else "'''"
                
                # 找到docstring结束
                k = j + 1
                while k < len(lines):
                    if quote in lines[k] and k > j:
                        break
                    k += 1
                
                # 检查docstring中是否有参数说明（旧格式 :param）
                has_param = False
                has_args_section = False
                
                for m in range(j, min(k + 1, len(lines))):
                    if ':param ' in lines[m]:
                        has_param = True
                    if lines[m].strip() == 'Args:':
                        has_args_section = True
                
                # 如果有 :param 但没有 Args: section，需要转换
                if has_param and not has_args_section:
                    # 这里可以添加转换逻辑
                    pass
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def process_file(file_path: Path) -> bool:
    """处理文件.
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否修改
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = clean_docstring(content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """主函数."""
    project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit")
    python_files = list(project_root.rglob("*.py"))
    python_files = [f for f in python_files if "__pycache__" not in str(f)]
    
    print(f"处理 {len(python_files)} 个文件...")
    modified = sum(1 for f in python_files if process_file(f))
    print(f"完成！修改 {modified} 个文件")


if __name__ == "__main__":
    main()
