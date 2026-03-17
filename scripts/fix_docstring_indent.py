#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复docstring缩进问题."""

import re
from pathlib import Path


def fix_docstring_indent(content: str) -> str:
    """修复docstring中的缩进问题.
    
    Args:
        content: 文件内容
        
    Returns:
        修复后的内容
    """
    lines = content.split('\n')
    result = []
    
    in_args = False
    in_returns = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 检测Args: section
        if stripped == 'Args:':
            result.append(line)
            in_args = True
            in_returns = False
            continue
        
        # 检测Returns: section
        if stripped == 'Returns:':
            result.append(line)
            in_args = False
            in_returns = True
            continue
        
        # 检测其他section（结束Args/Returns）
        if stripped in ['Raises:', 'Examples:', 'References:', 'Notes:', 'Attributes:', 'Yields:']:
            in_args = False
            in_returns = False
            result.append(line)
            continue
        
        # 检测空行（可能结束section）
        if not stripped:
            # 检查下一行是否是新的section或参数
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                # 如果下一行不是参数定义，结束section
                if not re.match(r'^\w+\s*\([^)]+\):', next_stripped) and not next_stripped.startswith('np.') and not next_stripped.startswith('pd.'):
                    in_args = False
                    in_returns = False
            
            result.append(line)
            continue
        
        # 修复Args缩进
        if in_args:
            # 检查是否是参数定义: param_name (type):
            param_match = re.match(r'^(\w+)\s*\(([^)]+)\):\s*(.*)$', stripped)
            if param_match:
                # 计算正确缩进（Args:的缩进 + 4空格）
                args_indent = 0
                for j in range(i - 1, -1, -1):
                    if lines[j].strip() == 'Args:':
                        args_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                
                correct_indent = ' ' * (args_indent + 4)
                param_name = param_match.group(1)
                param_type = param_match.group(2)
                description = param_match.group(3)
                
                result.append(f"{correct_indent}{param_name} ({param_type}): {description}")
                continue
        
        # 修复Returns缩进
        if in_returns:
            # 检查是否是返回类型定义
            type_match = re.match(r'^([A-Z]\w*|Optional|Union|List|Dict|Tuple|Callable|np\.|pd\.)(\[[^\]]+\])?:?\s*(.*)$', stripped)
            if type_match:
                # 计算正确缩进
                returns_indent = 0
                for j in range(i - 1, -1, -1):
                    if lines[j].strip() == 'Returns:':
                        returns_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                
                correct_indent = ' ' * (returns_indent + 4)
                ret_type = type_match.group(1)
                if type_match.group(2):
                    ret_type += type_match.group(2)
                description = type_match.group(3)
                
                if description:
                    result.append(f"{correct_indent}{ret_type}: {description}")
                else:
                    result.append(f"{correct_indent}{ret_type}:")
                continue
        
        # 其他行保持不变
        result.append(line)
    
    return '\n'.join(result)


def process_file(file_path: Path) -> bool:
    """处理单个文件.
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否修改
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = fix_docstring_indent(content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """主函数."""
    project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit")
    python_files = list(project_root.rglob("*.py"))
    python_files = [f for f in python_files if "__pycache__" not in str(f)]
    
    print(f"找到 {len(python_files)} 个Python文件")
    print("=" * 60)
    
    modified_count = 0
    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(project_root.parent)
        if process_file(file_path):
            print(f"✅ {rel_path}")
            modified_count += 1
        else:
            print(f"⏭️  {rel_path}")
    
    print("=" * 60)
    print(f"完成！修改 {modified_count} 个文件")


if __name__ == "__main__":
    main()
