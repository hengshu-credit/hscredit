# 安装指南

## 系统要求

- Python >= 3.8
- 操作系统: Windows / macOS / Linux

## 安装方式

### 方式1: 从PyPI安装（推荐）

```bash
pip install hscredit
```

### 方式2: 从源码安装

```bash
# 克隆仓库
git clone https://github.com/hscredit/hscredit.git
cd hscredit

# 安装依赖
pip install -r requirements.txt

# 安装包（开发模式）
pip install -e .
```

### 方式3: 开发模式（无需安装）

在代码中直接使用`sys.path.insert`：

```python
import sys
from pathlib import Path

# 添加项目路径
project_root = Path("/path/to/hscredit")
sys.path.insert(0, str(project_root))

# 导入模块
from hscredit.report.excel import ExcelWriter
```

## 依赖项

### 核心依赖

```
numpy>=1.20.0
pandas>=1.3.0
openpyxl>=3.0.0
scikit-learn>=0.24.0
```

### 可选依赖

```bash
# XGBoost支持
pip install hscredit[xgboost]

# LightGBM支持
pip install hscredit[lightgbm]

# CatBoost支持
pip install hscredit[catboost]

# 所有框架支持
pip install hscredit[all]
```

## 验证安装

```python
import hscredit

# 打印版本信息
hscredit.info()
```

预期输出：

```
hscredit - 信用评分卡建模工具包
版本: 0.1.0
路径: /path/to/hscredit/__init__.py
Python: 3.10.x

已实现模块:
  ✅ hscredit.report.excel - Excel报告生成
  ✅ hscredit.model.losses - 自定义损失函数
```

## 常见问题

### Q1: 安装失败，提示缺少编译器

某些依赖（如lightgbm）需要C++编译器。

**解决方案**:

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# Windows
# 下载并安装 Visual Studio Build Tools
```

### Q2: 导入失败，提示找不到模块

**解决方案**:

```python
# 检查安装路径
import sys
print(sys.path)

# 重新安装
pip install --upgrade --force-reinstall hscredit
```

### Q3: 版本冲突

**解决方案**:

```bash
# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

pip install hscredit
```

## 下一步

安装完成后，请查看：

- {doc}`quickstart` - 快速开始教程
- {doc}`examples/index` - 示例代码
- {doc}`api/report` - API参考文档
