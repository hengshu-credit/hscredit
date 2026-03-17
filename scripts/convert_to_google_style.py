#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将代码注释风格统一为Google风格.

Google风格示例:
    一句话描述。
    
    详细描述。
    
    Args:
        param1 (type): 参数描述
        param2 (type, optional): 可选参数描述. Defaults to default_value.
    
    Returns:
        type: 返回值描述
        
    Raises:
        ExceptionType: 异常描述
        
    Examples:
        示例代码
        
    References:
        参考资料
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import copy


class DocstringConverter:
    """Docstring风格转换器."""
    
    def __init__(self):
        """初始化转换器."""
        self.indent = "    "  # 默认4空格缩进
    
    def convert(self, docstring: str) -> str:
        """转换docstring为Google风格.
        
        Args:
            docstring: 原始docstring
            
        Returns:
            转换后的Google风格docstring
        """
        if not docstring or not docstring.strip():
            return docstring
        
        lines = docstring.split('\n')
        result = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 检测并转换Parameters section
            if stripped == 'Parameters':
                # 转换为Args:
                result.append(line.replace('Parameters', 'Args:'))
                i += 1
                
                # 跳过分隔线
                if i < len(lines) and lines[i].strip().startswith('-'):
                    i += 1
                
                # 处理参数列表
                while i < len(lines):
                    param_line = lines[i]
                    param_stripped = param_line.strip()
                    
                    if not param_stripped or param_stripped in ['Returns', 'Raises', 'Examples', 'References', 'Notes', 'Attributes', 'Yields']:
                        break
                    
                    # 解析参数: param_name : type
                    match = re.match(r'^(\w+)\s*:\s*(.+)$', param_stripped)
                    if match:
                        param_name = match.group(1)
                        param_type = match.group(2).strip()
                        
                        # 获取缩进
                        indent = len(param_line) - len(param_line.lstrip())
                        indent_str = ' ' * indent
                        
                        # 读取参数描述
                        description_lines = []
                        i += 1
                        while i < len(lines):
                            desc_line = lines[i]
                            desc_stripped = desc_line.strip()
                            
                            # 检查是否是下一个参数或section
                            if not desc_stripped:
                                break
                            if re.match(r'^\w+\s*:\s*.+$', desc_stripped):
                                break
                            if desc_stripped in ['Returns', 'Raises', 'Examples', 'References', 'Notes', 'Attributes', 'Yields']:
                                break
                            
                            # 描述行
                            desc_indent = len(desc_line) - len(desc_line.lstrip())
                            if desc_indent > indent:
                                description_lines.append(desc_stripped)
                                i += 1
                            else:
                                break
                        
                        # 构建Google风格参数
                        description = ' '.join(description_lines)
                        if description:
                            result.append(f"{indent_str}{param_name} ({param_type}): {description}")
                        else:
                            result.append(f"{indent_str}{param_name} ({param_type}):")
                        
                        continue
                    else:
                        i += 1
                
                continue
            
            # 检测并转换Returns section
            elif stripped == 'Returns':
                result.append(line + ':')
                i += 1
                
                # 跳过分隔线
                if i < len(lines) and lines[i].strip().startswith('-'):
                    i += 1
                
                # 处理返回值
                while i < len(lines):
                    ret_line = lines[i]
                    ret_stripped = ret_line.strip()
                    
                    if not ret_stripped or ret_stripped in ['Raises', 'Examples', 'References', 'Notes', 'Attributes', 'Args', 'Yields']:
                        break
                    
                    # 返回类型和描述
                    indent = len(ret_line) - len(ret_line.lstrip())
                    indent_str = ' ' * indent
                    
                    # 检查是否是类型行
                    if re.match(r'^[A-Z]', ret_stripped) or re.match(r'^(Optional|Union|List|Dict|Tuple|Callable|np\.|pd\.)', ret_stripped):
                        ret_type = ret_stripped
                        i += 1
                        
                        # 读取描述
                        description_lines = []
                        while i < len(lines):
                            desc_line = lines[i]
                            desc_stripped = desc_line.strip()
                            
                            if not desc_stripped:
                                break
                            if desc_stripped in ['Raises', 'Examples', 'References', 'Notes', 'Attributes', 'Args', 'Yields']:
                                break
                            
                            desc_indent = len(desc_line) - len(desc_line.lstrip())
                            if desc_indent > indent:
                                description_lines.append(desc_stripped)
                                i += 1
                            else:
                                break
                        
                        description = ' '.join(description_lines)
                        if description:
                            result.append(f"{indent_str}{ret_type}: {description}")
                        else:
                            result.append(f"{indent_str}{ret_type}:")
                        
                        continue
                    else:
                        result.append(ret_line)
                        i += 1
                
                continue
            
            # 转换旧式:param风格
            elif stripped.startswith(':param '):
                match = re.match(r'^:param\s+(\w+):\s*(.*)$', stripped)
                if match:
                    param_name = match.group(1)
                    description = match.group(2)
                    
                    # 查找类型
                    param_type = 'Any'
                    if i + 1 < len(lines):
                        next_stripped = lines[i + 1].strip()
                        type_match = re.match(r'^:type\s+\w+:\s*(.+)$', next_stripped)
                        if type_match:
                            param_type = type_match.group(1).strip()
                            i += 1
                    
                    # 确保Args section存在
                    if not any(l.strip() == 'Args:' for l in result):
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        result.append(f"{indent_str}Args:")
                    
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    result.append(f"{indent_str}    {param_name} ({param_type}): {description}")
                    i += 1
                    continue
            
            # 转换:return风格
            elif stripped.startswith(':return:') or stripped.startswith(':returns:'):
                match = re.match(r'^:returns?:\s*(.*)$', stripped)
                if match:
                    description = match.group(1)
                    
                    # 查找返回类型
                    ret_type = 'Any'
                    if i + 1 < len(lines):
                        next_stripped = lines[i + 1].strip()
                        type_match = re.match(r'^:rtype:\s*(.+)$', next_stripped)
                        if type_match:
                            ret_type = type_match.group(1).strip()
                            i += 1
                    
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    result.append(f"{indent_str}Returns:")
                    result.append(f"{indent_str}    {ret_type}: {description}")
                    i += 1
                    continue
            
            # 其他行保持不变
            result.append(line)
            i += 1
        
        return '\n'.join(result)


def convert_file(file_path: Path) -> Tuple[bool, str]:
    """转换单个文件.
    
    Args:
        file_path: 文件路径
        
    Returns:
        (是否修改, 错误信息)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 先恢复到原始状态（移除之前脚本的影响）
        # 修复 Args 后面没有冒号的问题
        content = re.sub(r'^(\s*)Args\s*$', r'\1Args:', content, flags=re.MULTILINE)
        content = re.sub(r'^(\s*)Returns\s*$', r'\1Returns:', content, flags=re.MULTILINE)
        
        # 修复参数格式: param_name (type):\n        description -> param_name (type): description
        def fix_param_format(match):
            indent = match.group(1)
            param_name = match.group(2)
            param_type = match.group(3)
            description = match.group(4).strip() if match.group(4) else ''
            
            if description:
                return f"{indent}{param_name} ({param_type}): {description}"
            else:
                return f"{indent}{param_name} ({param_type}):"
        
        # 修复多行描述的情况
        lines = content.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # 检测参数行: param_name (type):
            param_match = re.match(r'^(\s+)(\w+)\s*\(([^)]+)\):\s*$', line)
            
            if param_match and i + 1 < len(lines):
                # 检查下一行是否是描述
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                
                # 如果下一行不是新参数或section，则合并到当前行
                if next_stripped and not re.match(r'^\w+\s*\([^)]+\):', next_stripped) and not next_stripped.endswith(':'):
                    indent = param_match.group(1)
                    param_name = param_match.group(2)
                    param_type = param_match.group(3)
                    
                    result_lines.append(f"{indent}{param_name} ({param_type}): {next_stripped}")
                    i += 2
                    continue
            
            result_lines.append(line)
            i += 1
        
        new_content = '\n'.join(result_lines)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            return True, ""
        else:
            return False, ""
    
    except Exception as e:
        return False, str(e)


def main():
    """主函数."""
    project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit")
    python_files = list(project_root.rglob("*.py"))
    python_files = [f for f in python_files if "__pycache__" not in str(f)]
    
    print(f"找到 {len(python_files)} 个Python文件")
    print("=" * 60)
    
    modified_count = 0
    error_count = 0
    
    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(project_root.parent)
        modified, error = convert_file(file_path)
        
        if error:
            print(f"❌ {rel_path}: {error}")
            error_count += 1
        elif modified:
            print(f"✅ {rel_path}")
            modified_count += 1
        else:
            print(f"⏭️  {rel_path}")
    
    print("=" * 60)
    print(f"完成！修改 {modified_count} 个文件，错误 {error_count} 个")


if __name__ == "__main__":
    main()
