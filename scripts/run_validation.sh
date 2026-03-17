#!/bin/bash
# hscredit 快速启动脚本

set -e

echo "========================================="
echo "hscredit 项目验证脚本"
echo "========================================="
echo ""

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  未检测到虚拟环境"
    echo "建议创建虚拟环境:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 虚拟环境: $VIRTUAL_ENV"
fi

echo ""
echo "步骤1: 安装依赖"
echo "========================================="
pip install -e . --quiet
echo "✅ 依赖安装完成"

echo ""
echo "步骤2: 检查模块导入"
echo "========================================="
python3 -c "
import sys
sys.path.insert(0, '.')

print('检查模块导入...')
modules = [
    'hscredit',
    'hscredit.report.excel',
    'hscredit.core.models'
]

for mod in modules:
    try:
        __import__(mod)
        print(f'  ✅ {mod}')
    except Exception as e:
        print(f'  ❌ {mod}: {e}')
        sys.exit(1)

print('所有模块导入成功')
"

echo ""
echo "步骤3: 运行单元测试"
echo "========================================="
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short -x
    echo "✅ 单元测试通过"
else
    echo "⚠️  pytest未安装，跳过单元测试"
fi

echo ""
echo "步骤4: 启动Jupyter验证"
echo "========================================="
echo "即将启动Jupyter Notebook进行交互式验证"
echo ""
echo "验证清单:"
echo "  1. 打开 00_project_overview.ipynb"
echo "  2. 运行所有单元格验证环境"
echo "  3. 打开 01_excel_writer_validation.ipynb"
echo "  4. 验证Excel写入功能"
echo ""
read -p "是否启动Jupyter? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd examples
    jupyter notebook
fi

echo ""
echo "========================================="
echo "✅ 验证完成"
echo "========================================="
