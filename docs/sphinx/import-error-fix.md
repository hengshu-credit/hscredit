# 模块导入错误解决方案

## 问题描述

运行 `00_project_overview.ipynb` 时出现以下错误：

```
❌ hscredit.report                          - 报告模块
   错误: No module named 'hscredit.report'
```

## 原因分析

主 `__init__.py` 文件尝试导入了很多还未实现的模块，导致整个包无法加载：

```python
# 错误的导入（模块不存在）
from .core.binning import OptimalBinning  # 模块未实现
from .model.linear import LogisticRegression  # 模块未实现
```

## 解决方案

### 步骤1: 修改__init__.py文件

已修改以下文件，只导入已实现的模块：

- ✅ `hscredit/__init__.py` - 主模块
- ✅ `hscredit/model/__init__.py` - 只导入losses
- ✅ `hscredit/core/__init__.py` - 设为占位符
- ✅ `hscredit/analysis/__init__.py` - 设为占位符
- ✅ `hscredit/utils/__init__.py` - 设为占位符

### 步骤2: 添加项目路径到sys.path

**重要**: 使用 `sys.path.insert` 而不是安装包，更适合开发调试。

**在Jupyter Notebook中**

打开 `00_project_overview.ipynb`，运行第一个单元格：

```python
import sys
from pathlib import Path

# 获取项目根目录
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")

# 添加到sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 验证导入
import hscredit
print(f"✅ hscredit导入成功")
print(f"版本: {hscredit.__version__}")
print(f"路径: {hscredit.__file__}")
```

**在Python脚本中**

```python
import sys
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
sys.path.insert(0, str(project_root))

# 导入模块
from hscredit.report.excel import ExcelWriter
from hscredit.core.models import FocalLoss
```

### 步骤3: 验证导入

**快速验证脚本**

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit
python scripts/quick_test.py
```

**预期输出**

```
✅ hscredit                                 - 主模块
✅ hscredit.report                          - 报告模块
✅ hscredit.report.excel                    - Excel报告
✅ hscredit.model                           - 模型模块
✅ hscredit.model.losses                    - 损失函数
...
导入成功: 12/12
```

### 步骤4: 重启Jupyter Kernel（如果需要）

**如果之前导入失败，需要重启kernel**：

1. 在Jupyter中，点击 **Kernel → Restart**
2. 或点击 **Kernel → Restart & Clear Output**
3. 然后重新运行所有单元格

## 完整操作流程

### 在Jupyter Notebook中

```bash
# 1. 启动Jupyter
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples
jupyter notebook

# 2. 打开 00_project_overview.ipynb

# 3. 运行第一个单元格（添加sys.path）

# 4. 运行所有单元格（Cell → Run All）
```

### 在命令行中

```bash
# 1. 进入项目目录
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit

# 2. 验证导入
python scripts/quick_test.py

# 3. 启动Jupyter
cd examples
jupyter notebook
```

## 当前已实现的模块

| 模块 | 状态 | 功能 |
|------|------|------|
| `hscredit.report.excel` | ✅ 已实现 | Excel报告生成 |
| `hscredit.model.losses` | ✅ 已实现 | 自定义损失函数 |

## 待实现的模块

| 模块 | 状态 | 功能 |
|------|------|------|
| `hscredit.core.binning` | 📋 占位符 | 分箱算法 |
| `hscredit.core.encoding` | 📋 占位符 | 编码转换 |
| `hscredit.core.selection` | 📋 占位符 | 特征筛选 |
| `hscredit.core.metrics` | 📋 占位符 | 指标计算 |
| `hscredit.model.linear` | 📋 占位符 | 线性模型 |
| `hscredit.model.scorecard` | 📋 占位符 | 评分卡 |
| `hscredit.analysis.strategy` | 📋 占位符 | 策略分析 |

## 常见问题

### Q1: 为什么使用sys.path.insert而不是pip install？

**优点**:
- ✅ 修改代码立即生效，无需重新安装
- ✅ 适合开发调试
- ✅ 不影响系统Python环境

**对比pip install -e .**:
- ⚠️ 修改`__init__.py`后需要重新安装
- ⚠️ 有时缓存问题导致导入失败

### Q2: 还是报错怎么办？

**检查Python路径**

```python
import sys
import hscredit

print(f"Python路径: {sys.executable}")
print(f"hscredit路径: {hscredit.__file__}")
```

确保hscredit路径指向项目目录。

**清理缓存**

```bash
# 删除__pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 删除.egg-info（如果存在）
rm -rf hscredit.egg-info
```

**重启Jupyter Kernel**

Kernel → Restart & Clear Output

### Q3: 如何确认导入成功？

```python
import hscredit

# 打印版本信息
hscredit.info()

# 检查已实现的功能
print(hscredit.__all__)
```

### Q4: 为什么不能导入其他模块？

因为这些模块还在开发中，`__init__.py` 中暂时注释了它们的导入。等待模块实现后会逐步开放。

## 下一步

1. ✅ 问题已解决
2. 📝 继续运行notebook验证
3. 🚀 开始迁移下一个模块（core.binning）

---

**更新时间**: 2024-01
**文档版本**: 2.0
