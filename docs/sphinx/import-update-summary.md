# 导入方式统一修改总结

## 修改原则

所有示例代码和notebook使用`sys.path.insert`方式导入hscredit，无需安装包。

---

## 已修改文件列表

### ✅ Python脚本（examples/）

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `custom_loss_usage.py` | 添加sys.path.insert | ✅ 完成 |
| `excel_report_examples.py` | 添加sys.path.insert | ✅ 完成 |
| `basic_usage.py` | 无需修改（未导入hscredit） | ✅ 无需修改 |

### ✅ Jupyter Notebook（examples/）

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `00_project_overview.ipynb` | 已有sys.path.insert | ✅ 已正确 |
| `01_excel_writer_validation.ipynb` | 已有sys.path.insert | ✅ 已正确 |

### ✅ 文档文件（docs/）

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `examples/README.md` | 更新说明，移除安装步骤 | ✅ 完成 |
| `docs/IMPORT_GUIDE.md` | 新增完整导入指南 | ✅ 新增 |

### ✅ 测试文件（tests/）

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `test_excel_writer.py` | 无需修改（pytest自动处理） | ✅ 无需修改 |
| `test_losses.py` | 无需修改（pytest自动处理） | ✅ 无需修改 |
| `test_binning.py` | 无需修改（pytest自动处理） | ✅ 无需修改 |

---

## 修改详情

### 1. custom_loss_usage.py

**修改前**:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from hscredit.core.models import (
    FocalLoss,
    ...
)
```

**修改后**:
```python
import sys
from pathlib import Path

# 添加项目路径到sys.path（开发模式）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from hscredit.core.models import (
    FocalLoss,
    ...
)
```

### 2. excel_report_examples.py

**修改前**:
```python
import pandas as pd
import numpy as np
from hscredit.report.excel import ExcelWriter, dataframe2excel
```

**修改后**:
```python
import sys
from pathlib import Path

# 添加项目路径到sys.path（开发模式）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from hscredit.report.excel import ExcelWriter, dataframe2excel
```

### 3. examples/README.md

**修改前**:
```bash
# 安装项目（开发模式）
pip install -e ..
```

**修改后**:
```bash
# 安装依赖（不安装hscredit包）
pip install numpy pandas openpyxl scikit-learn
```

---

## 新增文档

### docs/IMPORT_GUIDE.md

完整的导入方式指南，包含：

- ✅ Python脚本的导入方式
- ✅ Jupyter Notebook的导入方式
- ✅ sys.path.insert的优点
- ✅ 不同场景的路径设置
- ✅ 常见问题解决方案
- ✅ 最佳实践

---

## 验证方式

### 验证Python脚本

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples

# 运行示例脚本
python custom_loss_usage.py
python excel_report_examples.py
```

### 验证Jupyter Notebook

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples
jupyter notebook

# 打开并运行
# - 00_project_overview.ipynb
# - 01_excel_writer_validation.ipynb
```

### 验证测试文件

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit
pytest tests/
```

---

## 统一模板

### Python脚本模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本说明
"""

import sys
from pathlib import Path

# 添加项目路径到sys.path（开发模式）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入hscredit模块
from hscredit.xxx import YYY

# 你的代码...
```

### Jupyter Notebook模板

```python
# 第一个单元格
import sys
from pathlib import Path

# 添加项目路径到sys.path（开发模式）
project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入hscredit模块
from hscredit.xxx import YYY
```

---

## 注意事项

### ⚠️ 不要同时使用多种导入方式

避免在同一环境中混用`pip install -e .`和`sys.path.insert`。

### ⚠️ 路径设置要正确

确保`project_root`指向正确的项目根目录（包含hscredit包的目录）。

### ⚠️ 测试文件无需修改

pytest会自动处理Python路径，测试文件直接使用`from hscredit.xxx import xxx`即可。

---

## 相关文档

- [导入指南](./IMPORT_GUIDE.md) - 完整的导入方式说明
- [API快速参考](./API_QUICK_REFERENCE.md) - API使用指南
- [版本兼容性](./VERSION_COMPATIBILITY.md) - Python版本兼容性说明

---

**更新时间**: 2024-01
**文档版本**: 1.0
