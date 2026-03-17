#!/bin/bash
# 快速测试文档构建

set -e

echo "=========================================="
echo "hscredit文档构建测试"
echo "=========================================="
echo

# 进入文档目录
cd "$(dirname "$0")"

# 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python --version 2>&1)
echo "   $python_version"
echo

# 检查Sphinx是否安装
echo "2. 检查Sphinx..."
if python -c "import sphinx" 2>/dev/null; then
    sphinx_version=$(python -c "import sphinx; print(sphinx.__version__)")
    echo "   ✅ Sphinx $sphinx_version 已安装"
else
    echo "   ❌ Sphinx未安装"
    echo "   安装命令: pip install -r requirements.txt"
    exit 1
fi
echo

# 清理旧构建
echo "3. 清理旧构建文件..."
rm -rf _build/*
echo "   ✅ 清理完成"
echo

# 构建HTML文档
echo "4. 构建HTML文档..."
if python -m sphinx -b html . _build/html 2>&1 | tee build.log; then
    echo
    echo "   ✅ HTML文档构建成功"
else
    echo
    echo "   ❌ HTML文档构建失败"
    echo "   查看详细日志: build.log"
    exit 1
fi
echo

# 检查生成的文件
echo "5. 检查生成的文件..."
if [ -f "_build/html/index.html" ]; then
    html_files=$(find _build/html -name "*.html" | wc -l)
    echo "   ✅ 生成了 $html_files 个HTML文件"
    echo "   📂 文档路径: $(pwd)/_build/html/index.html"
else
    echo "   ❌ 未找到index.html"
    exit 1
fi
echo

echo "=========================================="
echo "✅ 文档构建测试通过"
echo "=========================================="
echo
echo "下一步:"
echo "  1. 预览文档: open _build/html/index.html"
echo "  2. 实时预览: make livehtml"
echo "  3. 部署文档: make deploy"
echo
