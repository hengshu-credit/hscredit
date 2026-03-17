#!/usr/bin/env python
"""
Sphinx文档构建脚本

提供多种文档构建方式：
- HTML文档
- PDF文档
- 实时预览
- 部署到GitHub Pages
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def check_sphinx_installed():
    """检查Sphinx是否已安装"""
    try:
        import sphinx
        print(f"✅ Sphinx {sphinx.__version__} 已安装")
        return True
    except ImportError:
        print("❌ Sphinx未安装")
        print("\n安装命令:")
        print("  pip install -r docs/sphinx/requirements.txt")
        return False


def install_dependencies():
    """安装文档构建依赖"""
    print("安装文档构建依赖...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ 依赖安装成功")
        return True
    else:
        print(f"❌ 依赖安装失败:\n{result.stderr}")
        return False


def build_html():
    """构建HTML文档"""
    print("\n" + "="*70)
    print("构建HTML文档")
    print("="*70 + "\n")
    
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "html"
    
    # 清理旧的构建文件
    if build_dir.exists():
        print("清理旧的构建文件...")
        shutil.rmtree(build_dir)
    
    # 构建文档
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-W",  # 将警告视为错误
            "--keep-going",  # 继续处理其他文件
            str(docs_dir),
            str(build_dir)
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ HTML文档构建成功")
        print(f"📂 文档路径: {build_dir}")
        print(f"🌐 打开文档: file://{build_dir}/index.html")
        return True
    else:
        print("\n❌ HTML文档构建失败")
        print("\n错误信息:")
        print(result.stderr)
        print("\n输出信息:")
        print(result.stdout)
        return False


def build_pdf():
    """构建PDF文档"""
    print("\n" + "="*70)
    print("构建PDF文档")
    print("="*70 + "\n")
    
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "latex"
    
    # 构建LaTeX
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "latex",
            str(docs_dir),
            str(build_dir)
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ LaTeX文档构建失败")
        print(result.stderr)
        return False
    
    # 编译PDF
    print("编译PDF...")
    result = subprocess.run(
        ["make", "all-pdf"],
        cwd=str(build_dir),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        pdf_file = list(build_dir.glob("*.pdf"))[0]
        print(f"\n✅ PDF文档构建成功")
        print(f"📄 PDF路径: {pdf_file}")
        return True
    else:
        print("❌ PDF编译失败")
        print(result.stderr)
        return False


def serve_live():
    """启动实时预览服务器"""
    print("\n" + "="*70)
    print("启动实时预览服务器")
    print("="*70 + "\n")
    
    try:
        import sphinx_autobuild
    except ImportError:
        print("❌ sphinx-autobuild未安装")
        print("安装命令: pip install sphinx-autobuild")
        return False
    
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "html"
    
    subprocess.run([
        sys.executable, "-m", "sphinx_autobuild",
        "--open-browser",
        "--host", "0.0.0.0",
        "--port", "8000",
        str(docs_dir),
        str(build_dir)
    ])


def deploy_to_github_pages():
    """部署到GitHub Pages"""
    print("\n" + "="*70)
    print("部署到GitHub Pages")
    print("="*70 + "\n")
    
    # 先构建HTML文档
    if not build_html():
        print("❌ 部署失败：HTML文档构建失败")
        return False
    
    # 检查ghp-import是否安装
    try:
        import ghp_import
    except ImportError:
        print("❌ ghp-import未安装")
        print("安装命令: pip install ghp-import")
        return False
    
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "html"
    
    # 部署
    result = subprocess.run(
        ["ghp-import", "-n", "-p", "-f", str(build_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ 部署成功")
        print("🌐 访问: https://hscredit.github.io/hscredit/")
        return True
    else:
        print("❌ 部署失败")
        print(result.stderr)
        return False


def check_links():
    """检查外部链接"""
    print("\n" + "="*70)
    print("检查外部链接")
    print("="*70 + "\n")
    
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "linkcheck"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "linkcheck",
            str(docs_dir),
            str(build_dir)
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ 所有链接检查通过")
    else:
        print("⚠️  部分链接检查失败")
        print(result.stdout)
    
    return result.returncode == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="hscredit文档构建工具")
    parser.add_argument(
        "command",
        choices=["install", "html", "pdf", "live", "deploy", "check"],
        help="执行的命令"
    )
    
    args = parser.parse_args()
    
    # 检查Sphinx是否安装（除了install命令）
    if args.command != "install" and not check_sphinx_installed():
        print("\n请先安装依赖:")
        print("  python build_docs.py install")
        return 1
    
    # 执行命令
    commands = {
        "install": install_dependencies,
        "html": build_html,
        "pdf": build_pdf,
        "live": serve_live,
        "deploy": deploy_to_github_pages,
        "check": check_links
    }
    
    success = commands[args.command]()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
