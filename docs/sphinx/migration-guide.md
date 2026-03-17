# hscredit 代码迁移指南

## 概述

本文档提供hscredit项目代码迁移的详细指南，确保在Python 3.8+环境下正常工作。

## 迁移原则

### 1. 边迁移边验证
- ✅ 使用Jupyter Notebook进行交互式验证
- ✅ 每个模块迁移后立即测试
- ✅ 确保所有功能在目标Python版本下工作正常

### 2. 版本兼容性
- 最低支持: Python 3.8
- 最高支持: Python 3.12+
- 使用类型注解提高代码质量
- 避免使用版本特定特性（或提供兼容方案）

### 3. 代码质量
- 完整的类型注解
- 详细的文档字符串
- 单元测试覆盖
- 性能基准测试

## 迁移流程

### 阶段1: 环境准备

#### 1.1 创建虚拟环境

```bash
# 使用Python 3.8创建虚拟环境
python3.8 -m venv venv38
source venv38/bin/activate  # Linux/Mac
# 或 venv38\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .  # 开发模式安装
```

#### 1.2 安装Jupyter

```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=hscredit
```

#### 1.3 验证环境

运行 `00_project_overview.ipynb` 验证环境配置。

### 阶段2: 模块迁移顺序

建议按以下顺序迁移模块：

```
1. ✅ 报告模块 (report)
   └── excel/ - Excel写入（已完成）

2. 🚧 核心模块 (core)
   ├── binning/ - 分箱算法
   ├── encoding/ - 编码转换
   ├── selection/ - 特征筛选
   └── metrics/ - 指标计算

3. 📋 模型模块 (model)
   ├── linear/ - 线性模型
   ├── scorecard/ - 评分卡
   └── losses/ - 自定义损失（已完成）

4. 📊 分析模块 (analysis)
   ├── strategy/ - 策略分析
   └── rules/ - 规则挖掘

5. 🔧 工具模块 (utils)
   └── 通用工具函数
```

### 阶段3: 单模块迁移步骤

#### 3.1 创建迁移Notebook

每个模块迁移时，创建对应的验证notebook：

```
examples/
├── 00_project_overview.ipynb        # 项目总览
├── 01_excel_writer_validation.ipynb  # Excel写入验证
├── 02_binning_validation.ipynb       # 分箱模块验证
├── 03_encoding_validation.ipynb      # 编码模块验证
└── ...
```

#### 3.2 迁移步骤模板

以分箱模块为例：

**Step 1: 分析原代码**

```python
# 在notebook中分析原模块结构
from scorecardpipeline import binning

# 查看API
print(dir(binning))
print(binning.__all__ if hasattr(binning, '__all__') else "无__all__")
```

**Step 2: 创建目标目录**

```python
from pathlib import Path
Path("hscredit/core/binning").mkdir(parents=True, exist_ok=True)
```

**Step 3: 迁移代码**

```python
# 复制核心代码文件
import shutil
shutil.copy("scorecardpipeline/binning.py", "hscredit/core/binning/binning.py")
```

**Step 4: 添加类型注解**

```python
# 改进代码示例
def decision_tree_binning(
    X: np.ndarray,
    y: np.ndarray,
    max_bins: int = 10,
    min_samples_leaf: int = 50
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    决策树分箱
    
    Parameters
    ----------
    X : np.ndarray
        特征数组
    y : np.ndarray
        标签数组
    max_bins : int, default=10
        最大分箱数
    min_samples_leaf : int, default=50
        叶子节点最小样本数
    
    Returns
    -------
    bins : np.ndarray
        分箱边界
    info : Dict[str, Any]
        分箱信息
    """
    # 实现...
```

**Step 5: 创建模块入口**

```python
# hscredit/core/binning/__init__.py
from .binning import (
    decision_tree_binning,
    chi_merge_binning,
    monotonic_binning
)

__all__ = [
    "decision_tree_binning",
    "chi_merge_binning",
    "monotonic_binning"
]
```

**Step 6: 编写单元测试**

```python
# tests/test_binning.py
import pytest
import numpy as np
from hscredit.core.binning import decision_tree_binning

def test_decision_tree_binning():
    """测试决策树分箱"""
    X = np.random.randn(1000)
    y = np.random.randint(0, 2, 1000)
    
    bins, info = decision_tree_binning(X, y, max_bins=10)
    
    assert len(bins) <= 11  # max_bins + 1
    assert "iv" in info
```

**Step 7: 在Notebook中验证**

```python
# 在validation notebook中运行
from hscredit.core.binning import decision_tree_binning
import numpy as np

# 测试基本功能
X = np.random.randn(1000)
y = np.random.randint(0, 2, 1000)

bins, info = decision_tree_binning(X, y, max_bins=10)

print(f"分箱数: {len(bins) - 1}")
print(f"IV值: {info['iv']:.4f}")
print("✅ 分箱功能验证通过")
```

### 阶段4: 版本兼容性处理

#### 4.1 Python 3.8 兼容性

```python
# 使用typing模块的类型
from typing import List, Dict, Optional, Union, Tuple

# 字典合并（兼容Python 3.8）
def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """兼容Python 3.8的字典合并"""
    result = dict1.copy()
    result.update(dict2)
    return result

# 或使用版本检查
import sys

if sys.version_info >= (3, 9):
    merged = dict1 | dict2
else:
    merged = {**dict1, **dict2}
```

#### 4.2 避免Python 3.10+特性

```python
# ❌ 不要使用（仅Python 3.10+）
match value:
    case 1:
        ...
    case _:
        ...

# ✅ 使用兼容方案
if value == 1:
    ...
else:
    ...
```

#### 4.3 使用条件导入

```python
# 可选依赖处理
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def train_xgboost(*args, **kwargs):
    if not HAS_XGBOOST:
        raise ImportError("xgboost未安装，请使用: pip install xgboost")
    # 训练代码...
```

## 验证清单

每个模块迁移完成后，需要验证以下内容：

### 功能验证

- [ ] 模块可以正常导入
- [ ] 所有公共API可访问
- [ ] 核心功能正常工作
- [ ] 边界情况处理正确
- [ ] 错误提示清晰

### 质量验证

- [ ] 完整的类型注解
- [ ] 详细的文档字符串
- [ ] 单元测试覆盖
- [ ] 代码风格一致
- [ ] 无linter警告

### 性能验证

- [ ] 基本性能可接受
- [ ] 大数据量测试通过
- [ ] 内存使用合理
- [ ] 无内存泄漏

### 兼容性验证

- [ ] Python 3.8测试通过
- [ ] Python 3.9测试通过
- [ ] Python 3.10测试通过
- [ ] Python 3.11测试通过
- [ ] Python 3.12测试通过

## 测试方法

### 单元测试

```bash
# 运行单个模块测试
pytest tests/test_binning.py -v

# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=hscredit --cov-report=html
```

### Notebook测试

```bash
# 执行notebook验证
jupyter nbconvert --to notebook --execute examples/01_excel_writer_validation.ipynb

# 或使用papermill
pip install papermill
papermill examples/01_excel_writer_validation.ipynb outputs/result.ipynb
```

### 多版本测试

```bash
# 使用tox测试多版本
pip install tox

# 创建tox.ini
cat > tox.ini << EOF
[tox]
envlist = py38,py39,py310,py311,py312

[testenv]
deps = 
    -r requirements.txt
    pytest
commands = 
    pytest tests/ -v
EOF

# 运行测试
tox
```

## 常见问题

### Q1: 如何处理可选依赖？

```python
# 在pyproject.toml中定义可选依赖
[project.optional-dependencies]
xgboost = ["xgboost>=1.4.0"]

# 在代码中使用
try:
    import xgboost
except ImportError:
    raise ImportError("请安装: pip install hscredit[xgboost]")
```

### Q2: 如何处理不同版本的API差异？

```python
# 使用版本检查
import sys

if sys.version_info >= (3, 9):
    # 新版本API
    result = new_api()
else:
    # 旧版本兼容
    result = old_api()
```

### Q3: 如何确保类型注解正确？

```bash
# 使用mypy检查类型
pip install mypy
mypy hscredit --ignore-missing-imports
```

### Q4: 如何处理大型数据测试？

```python
# 在测试中使用fixture生成数据
@pytest.fixture
def large_dataset():
    """生成大型测试数据"""
    np.random.seed(42)
    return {
        'X': np.random.randn(100000, 50),
        'y': np.random.randint(0, 2, 100000)
    }

def test_large_data(large_dataset):
    """测试大数据处理"""
    # 测试代码...
```

## 迁移进度追踪

使用以下表格追踪迁移进度：

| 模块 | 状态 | 迁移日期 | 验证日期 | 备注 |
|------|------|----------|----------|------|
| report.excel | ✅ 已完成 | 2024-01-15 | 2024-01-15 | 通过所有测试 |
| model.losses | ✅ 已完成 | 2024-01-16 | 2024-01-16 | 通过所有测试 |
| core.binning | 🚧 进行中 | - | - | 待迁移 |
| core.encoding | 📋 计划中 | - | - | 待迁移 |
| core.selection | 📋 计划中 | - | - | 待迁移 |
| core.metrics | 📋 计划中 | - | - | 待迁移 |
| model.linear | 📋 计划中 | - | - | 待迁移 |
| model.scorecard | 📋 计划中 | - | - | 待迁移 |

## 参考资料

- [Python 3.8 新特性](https://docs.python.org/3.8/whatsnew/3.8.html)
- [Python 3.9 新特性](https://docs.python.org/3.9/whatsnew/3.9.html)
- [Python 3.10 新特性](https://docs.python.org/3.10/whatsnew/3.10.html)
- [Python 3.11 新特性](https://docs.python.org/3.11/whatsnew/3.11.html)
- [Python 3.12 新特性](https://docs.python.org/3.12/whatsnew/3.12.html)
- [PEP 484 类型提示](https://www.python.org/dev/peps/pep-0484/)
- [NumPy 类型注解指南](https://numpy.org/devdocs/reference/typing.html)

## 下一步

1. ✅ 完成Excel写入模块迁移（已完成）
2. ✅ 完成自定义损失函数迁移（已完成）
3. 🚧 开始核心分箱模块迁移
4. 📋 创建分箱模块验证notebook
5. 📋 迁移特征筛选模块
6. 📋 迁移编码模块
7. 📋 迁移指标计算模块

---

**注意**: 迁移过程中如遇到问题，请在notebook中记录并更新本文档。
