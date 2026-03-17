# hscredit 导入方式说明

## 开发模式导入（推荐）

所有示例代码和notebook都使用`sys.path.insert`方式导入hscredit，无需安装包。

### Python脚本

```python
import sys
from pathlib import Path

# 添加项目路径到sys.path（使用绝对路径）
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以导入hscredit
from hscredit.report.excel import ExcelWriter
from hscredit.core.models import FocalLoss
```

### Jupyter Notebook

```python
import sys
from pathlib import Path

# 使用绝对路径
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入模块
from hscredit.report.excel import ExcelWriter
from hscredit.core.models import FocalLoss
```

---

## 为什么使用sys.path.insert？

### 优点 ✅

1. **修改立即生效** - 无需重新安装包
2. **适合开发调试** - 修改代码后直接运行即可
3. **不影响系统环境** - 不安装到系统Python环境
4. **避免缓存问题** - 不会因为缓存导致旧代码生效
5. **多版本共存** - 可以同时开发多个版本

### 对比pip install -e .

| 方式 | 优点 | 缺点 |
|------|------|------|
| **sys.path.insert** | ✅ 修改立即生效<br>✅ 无需重启kernel<br>✅ 不影响系统环境 | 需在每个脚本中添加 |
| **pip install -e .** | 一次安装到处使用 | ⚠️ 修改`__init__.py`需重新安装<br>⚠️ 可能有缓存问题 |

---

## 不同场景的路径设置

### 场景1: 在examples目录下的notebook

```python
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent  # 上两级到项目根目录
sys.path.insert(0, str(project_root))
```

### 场景2: 在项目根目录下的脚本

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent  # 当前目录就是项目根目录
sys.path.insert(0, str(project_root))
```

### 场景3: 在tests目录下的测试脚本

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # 上两级到项目根目录
sys.path.insert(0, str(project_root))
```

### 场景4: 任意位置的脚本

```python
import sys
from pathlib import Path

# 使用绝对路径
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
sys.path.insert(0, str(project_root))
```

---

## 验证导入是否成功

```python
import hscredit

# 打印信息
print(f"✅ hscredit导入成功")
print(f"版本: {hscredit.__version__}")
print(f"路径: {hscredit.__file__}")
print(f"Python: {sys.executable}")

# 验证模块
hscredit.info()
```

---

## 常见问题

### Q1: ImportError: No module named 'hscredit'

**原因**: sys.path设置错误

**解决**:
```python
# 检查project_root是否正确
print(f"项目根目录: {project_root}")
print(f"路径是否存在: {project_root.exists()}")
print(f"hscredit目录是否存在: {(project_root / 'hscredit').exists()}")

# 确保添加到sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print("✅ 已添加到sys.path")
```

### Q2: 导入的是旧代码

**原因**: 缓存问题或导入了错误路径

**解决**:
```python
# 清理Python缓存
import importlib
import hscredit
importlib.reload(hscredit)

# 检查导入路径
import hscredit
print(f"导入路径: {hscredit.__file__}")
# 应该指向项目目录，而不是site-packages
```

### Q3: ModuleNotFoundError: No module named 'hscredit.xxx'

**原因**: 该模块还未实现

**解决**:
```python
# 检查模块是否存在
import os
module_path = Path(__file__).parent.parent / "hscredit" / "xxx"
print(f"模块路径: {module_path}")
print(f"是否存在: {module_path.exists()}")

# 检查__init__.py
init_file = module_path / "__init__.py"
print(f"__init__.py是否存在: {init_file.exists()}")
```

### Q4: 在Jupyter中修改代码后不生效

**解决**:
```python
# 方法1: 重启kernel
# Kernel → Restart

# 方法2: 使用autoreload（推荐）
%load_ext autoreload
%autoreload 2

# 方法3: 手动reload
import importlib
import hscredit
importlib.reload(hscredit)
```

---

## 最佳实践

### 1. 在notebook开头统一添加

```python
# 第一个单元格
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

# 验证
import hscredit
print(f"✅ hscredit {hscredit.__version__}")
```

### 2. 使用autoreload（可选）

```python
# 自动重载修改的模块
%load_ext autoreload
%autoreload 2
```

### 3. 检查导入路径

```python
# 确保导入的是开发版本
import hscredit
assert "CodeBuddy/hscredit/hscredit" in hscredit.__file__, \
    f"导入了错误的路径: {hscredit.__file__}"
```

---

## 项目结构

```
hscredit/
├── hscredit/              # 源代码（需要添加到sys.path）
│   ├── __init__.py
│   ├── report/
│   ├── model/
│   └── core/
├── examples/              # 示例代码
│   ├── 00_project_overview.ipynb
│   ├── 01_excel_writer_validation.ipynb
│   ├── custom_loss_usage.py
│   └── excel_report_examples.py
├── tests/                 # 测试代码
├── docs/                  # 文档
└── scripts/               # 脚本
```

---

## 快速参考

### 标准导入模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本说明
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入hscredit模块
from hscredit.xxx import YYY

# 你的代码...
```

---

**更新时间**: 2024-01
**文档版本**: 1.0
