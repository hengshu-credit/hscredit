#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量转换hscredit代码注释为Google-style格式.

转换规则:
1. Parameters -> :param
2. Returns -> :return
3. 移除分隔线 (------)
4. 简化格式
"""

import re
from pathlib import Path


def convert_numpy_to_google_style(content: str) -> str:
    """将NumPy风格docstring转换为Google风格.
    
    :param content: 文件内容
    :return: 转换后的内容
    """
    # 替换类/函数文档字符串中的 Parameters 部分
    # 模式: Parameters\n    ----------\n    param : type\n        description
    
    # 先处理 Parameters
    def replace_params(match):
        params_text = match.group(1)
        # 移除分隔线
        params_text = re.sub(r'\n\s*-+\s*\n', '\n', params_text)
        
        # 转换每个参数
        lines = params_text.strip().split('\n')
        result = []
        current_param = None
        
        for line in lines:
            stripped = line.strip()
            # 匹配参数定义: param : type
            param_match = re.match(r'^(\w+)\s*:\s*(.+)$', stripped)
            if param_match:
                param_name = param_match.group(1)
                param_type = param_match.group(2)
                # 移除 by default XXX
                param_type = re.sub(r',?\s*by default\s+[^,\n]+', '', param_type)
                current_param = f":param {param_name}: {param_type}"
                result.append(current_param)
            elif current_param and stripped and not stripped.startswith(':') and not stripped.startswith('**'):
                # 这是参数描述的续行
                result[-1] += " " + stripped
            elif stripped:
                result.append(line)
        
        return '\n'.join(result)
    
    # 处理 Parameters 块
    content = re.sub(
        r'Parameters\n\s*-+\n((?:\s+\w+\s*:\s*[^\n]+\n(?:\s+[^\n\S]+[^\n]*\n)*)+)',
        lambda m: replace_params(m),
        content
    )
    
    # 处理 Returns
    def replace_returns(match):
        returns_text = match.group(1)
        # 移除分隔线
        returns_text = re.sub(r'\n\s*-+\s*\n', '\n', returns_text)
        
        lines = returns_text.strip().split('\n')
        result = []
        return_type = None
        
        for line in lines:
            stripped = line.strip()
            if not return_type and stripped and not stripped.startswith('**'):
                # 第一行是返回类型
                return_type = stripped
                result.append(f":return: {return_type}")
            elif return_type and stripped and not stripped.startswith('**'):
                # 返回描述
                result[-1] += " " + stripped
            elif stripped:
                result.append(line)
        
        return '\n'.join(result) if result else ""
    
    content = re.sub(
        r'Returns\n\s*-+\n((?:\s+[^\n]+\n)+)',
        lambda m: replace_returns(m),
        content
    )
    
    # 简化 Examples 和 References
    content = re.sub(r'Examples\n\s*-+\n', '**参考样例**\n\n', content)
    content = re.sub(r'References\n\s*-+\n', '**参考**\n\n', content)
    content = re.sub(r'Notes\n\s*-+\n', '**注意**\n\n', content)
    
    return content


def process_file(file_path: Path) -> bool:
    """处理单个文件.
    
    :param file_path: 文件路径
    :return: 是否修改
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = convert_numpy_to_google_style(content)
        
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
    python_files = [
        project_root / "core/metrics/classification.py",
        project_root / "core/metrics/importance.py",
        project_root / "core/metrics/regression.py",
        project_root / "core/metrics/stability.py",
        project_root / "core/binning/base.py",
        project_root / "core/encoding/base.py",
        project_root / "model/losses/focal_loss.py",
        project_root / "model/losses/risk_loss.py",
        project_root / "model/losses/weighted_loss.py",
        project_root / "model/losses/adapters.py",
        project_root / "model/losses/custom_metrics.py",
    ]
    
    print(f"处理 {len(python_files)} 个文件...")
    print("=" * 60)
    
    modified_count = 0
    for file_path in python_files:
        if file_path.exists():
            rel_path = file_path.relative_to(project_root.parent)
            if process_file(file_path):
                print(f"✅ {rel_path}")
                modified_count += 1
            else:
                print(f"⏭️  {rel_path}")
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    print("=" * 60)
    print(f"完成！修改 {modified_count} 个文件")


if __name__ == "__main__":
    main()
