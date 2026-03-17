#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将代码注释风格从NumPy/旧式风格转换为Google风格

转换规则：
1. :param 3.: param -> Args
:param 4.: return -> Returns
:param 5.: type -> (type)

"""

import re
from pathlib import Path
from typing import List, Tuple


def convert_numpy_to_google(docstring: str) -> str:
    """将NumPy风格的docstring转换为Google风格.
    
    Args:
        docstring (str): 原始docstring
        
    Returns:
        str: 转换后的Google风格docstring
    """
    if not docstring or not docstring.strip():
        return docstring
    
    lines = docstring.split('\n')
    result_lines = []
    current_section = None
    section_indent = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测section标题
        if stripped in ['Parameters', 'Returns', 'Raises', 'Examples', 'References', 'Notes', 'Attributes', 'Yields']:
            current_section = stripped
            
            # 转换section名称
            section_mapping = {
                'Parameters': 'Args',
            }
            
            new_section = section_mapping.get(stripped, stripped)
            result_lines.append(line.replace(stripped, new_section))
            
            # 记录section缩进
            section_indent = len(line) - len(line.lstrip())
            
            # 跳过分隔线（----）
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('-'):
                i += 1
            
            i += 1
            continue
        
        # 处理参数描述
        if current_section in ['Args', 'Parameters'] and stripped:
            # NumPy风格: param_name : type
            # 转换为: param_name (type):
            param_match = re.match(r'^(\w+)\s*:\s*(.+)$', stripped)
            if param_match:
                param_name = param_match.group(1)
                param_type = param_match.group(2).strip()
                
                # 找到缩进
                line_indent = len(line) - len(line.lstrip())
                
                # 生成Google风格
                indent_str = ' ' * line_indent
                result_lines.append(f"{indent_str}{param_name} ({param_type}):")
                
                i += 1
                continue
        
        # 处理返回值描述
        if current_section in ['Returns'] and stripped:
            # NumPy风格: type
            # 或: return_name : type
            # 保持不变，Google风格兼容
            
            # 检查是否是类型行（不以字母开头或全大写）
            type_match = re.match(r'^([A-Z]\w*|Optional|Union|List|Dict|Tuple|Callable|np\.|pd\.)', stripped)
            if type_match and not ':' in stripped:
                # 纯类型行，保持不变
                pass
            
            result_lines.append(line)
            i += 1
            continue
        
        # 处理旧式:param风格
        if stripped.startswith(':param '):
            # :param param_name: description
            # :type param_name: type
            param_match = re.match(r'^:param\s+(\w+):\s*(.*)$', stripped)
            if param_match:
                param_name = param_match.group(1)
                description = param_match.group(2)
                
                # 查找类型
                param_type = 'Any'
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    type_match = re.match(r'^:type\s+\w+:\s*(.+)$', next_line)
                    if type_match:
                        param_type = type_match.group(1).strip()
                        i += 1  # 跳过:type行
                
                # 生成Google风格
                indent_str = ' ' * (len(line) - len(line.lstrip()))
                result_lines.append(f"{indent_str}{param_name} ({param_type}): {description}")
                
                i += 1
                continue
        
        # 处理:return风格
        if stripped.startswith(':return:') or stripped.startswith(':returns:'):
            return_match = re.match(r'^:returns?:\s*(.*)$', stripped)
            if return_match:
                description = return_match.group(1)
                
                # 查找返回类型
                return_type = 'Any'
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    type_match = re.match(r'^:rtype:\s*(.+)$', next_line)
                    if type_match:
                        return_type = type_match.group(1).strip()
                        i += 1  # 跳过:rtype行
                
                # 生成Google风格
                indent_str = ' ' * (len(line) - len(line.lstrip()))
                if description:
                    result_lines.append(f"{indent_str}Returns:")
                    result_lines.append(f"{indent_str}    {return_type}: {description}")
                else:
                    result_lines.append(f"{indent_str}Returns:")
                    result_lines.append(f"{indent_str}    {return_type}:")
                
                i += 1
                continue
        
        # 其他行保持不变
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def convert_file_to_google(file_path: Path) -> Tuple[bool, str]:
    """转换单个文件的docstring风格.
    
    Args:
        file_path (Path): 文件路径
        
    Returns:
        Tuple[bool, str]: (是否修改, 错误信息)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 查找所有docstring
        # 匹配三引号包裹的内容
        docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
        
        def replace_docstring(match):
            docstring = match.group(1)
            quote_char = docstring[0]  # " 或 '
            
            # 提取内容
            inner = docstring[3:-3]
            
            # 转换
            converted = convert_numpy_to_google(inner)
            
            # 重新包裹
            return f'{quote_char * 3}{converted}{quote_char * 3}'
        
        new_content = re.sub(docstring_pattern, replace_docstring, content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            return True, ""
        else:
            return False, ""
    
    except Exception as e:
        return False, str(e)


def main():
    """主函数."""
    # 查找所有Python文件
    project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit")
    python_files = list(project_root.rglob("*.py"))
    
    # 过滤掉__pycache__
    python_files = [f for f in python_files if "__pycache__" not in str(f)]
    
    print(f"找到 {len(python_files)} 个Python文件")
    print("=" * 60)
    
    modified_count = 0
    error_count = 0
    
    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(project_root.parent)
        modified, error = convert_file_to_google(file_path)
        
        if error:
            print(f"❌ {rel_path}: {error}")
            error_count += 1
        elif modified:
            print(f"✅ {rel_path}")
            modified_count += 1
        else:
            print(f"⏭️  {rel_path} (无需修改)")
    
    print("=" * 60)
    print(f"完成！修改 {modified_count} 个文件，错误 {error_count} 个")


if __name__ == "__main__":
    main()
