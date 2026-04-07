#!/usr/bin/env python3
"""
批量测试所有examples下的notebook
"""
import os
import sys
import subprocess
from pathlib import Path

def run_notebook(notebook_path):
    """运行单个notebook并返回结果"""
    print(f"\n{'='*60}")
    print(f"运行: {notebook_path.name}")
    print('='*60)
    
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--allow-errors",
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=notebook_path.parent
        )
        
        if result.returncode == 0:
            print(f"✅ {notebook_path.name} - 成功")
            return True, None
        else:
            print(f"❌ {notebook_path.name} - 失败")
            error_msg = result.stderr if result.stderr else "未知错误"
            print(f"错误: {error_msg[:500]}")
            return False, error_msg
    except subprocess.TimeoutExpired:
        print(f"⏱️ {notebook_path.name} - 超时")
        return False, "执行超时(超过5分钟)"
    except Exception as e:
        print(f"💥 {notebook_path.name} - 异常: {e}")
        return False, str(e)

def main():
    examples_dir = Path(__file__).parent / "examples"
    
    # 获取所有需要测试的notebook
    notebooks = sorted([
        f for f in examples_dir.glob("*.ipynb")
        if f.name.startswith(tuple(f"{i:02d}" for i in range(1, 13)))
    ])
    
    print(f"发现 {len(notebooks)} 个待测试notebook")
    
    results = {}
    for nb in notebooks:
        success, error = run_notebook(nb)
        results[nb.name] = {"success": success, "error": error}
    
    # 汇总报告
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r["success"])
    fail_count = len(results) - success_count
    
    for name, result in results.items():
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {len(results)} | 成功: {success_count} | 失败: {fail_count}")
    
    if fail_count > 0:
        print("\n失败详情:")
        for name, result in results.items():
            if not result["success"]:
                print(f"\n{name}:")
                print(f"  错误: {result['error'][:300] if result['error'] else 'Unknown'}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
