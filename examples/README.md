# hscredit 示例代码

本目录包含hscredit项目的所有示例代码和验证notebook。

## 📚 Notebook列表

### 验证Notebook

| 编号 | 名称 | 描述 | 状态 |
|------|------|------|------|
| 00 | `project_overview.ipynb` | 项目总览和环境验证 | ✅ 可用 |
| 01 | `excel_writer_validation.ipynb` | Excel写入模块验证 | ✅ 可用 |

### 使用示例

| 编号 | 名称 | 描述 | 状态 |
|------|------|------|------|
| - | `custom_loss_usage.py` | 自定义损失函数使用示例 | ✅ 可用 |
| - | `excel_report_examples.py` | Excel报告生成示例 | ✅ 可用 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖（不安装hscredit包）
pip install numpy pandas openpyxl scikit-learn

# 安装Jupyter（如果未安装）
pip install jupyter notebook ipykernel

# 注册kernel
python -m ipykernel install --user --name=hscredit
```

### 2. 启动Jupyter

```bash
# 在examples目录下启动
cd examples
jupyter notebook
```

### 3. 运行验证

1. 打开 `00_project_overview.ipynb`
2. 第一个单元格会自动添加项目路径到sys.path
3. 执行所有单元格（Cell -> Run All）
4. 检查输出是否正常

**注意**: 所有示例都使用`sys.path.insert`方式导入hscredit，无需安装包即可运行。

## 📖 Notebook详细说明

### 00_project_overview.ipynb

**目的**: 验证项目整体环境和模块导入

**内容**:
- Python环境检查
- 依赖版本检查
- 模块导入验证
- API导出验证
- 基本功能测试
- Python版本兼容性测试
- 性能基准测试

**预期结果**: 所有测试通过，输出✅标记

### 01_excel_writer_validation.ipynb

**目的**: 验证Excel写入模块功能

**内容**:
- ExcelWriter基本使用
- DataFrame写入测试
- 值插入和合并单元格
- 数字格式化
- 条件格式
- 多层索引
- 超链接
- 追加模式
- 样式系统
- 错误处理

**预期结果**: 生成多个测试Excel文件

## 🔧 Python脚本说明

### custom_loss_usage.py

展示自定义损失函数的完整使用流程：

```python
# 运行示例
python custom_loss_usage.py
```

包含示例：
- Focal Loss处理不平衡数据
- 成本敏感学习
- 坏账率优化
- 利润最大化
- 自定义评估指标

### excel_report_examples.py

展示Excel报告生成的各种用法：

```python
# 运行示例
python excel_report_examples.py
```

包含示例：
- 基本使用
- 便捷函数
- 高级功能
- 多工作表
- 样式定制
- 追加模式

## 📊 输出文件

运行notebook后，会在`outputs/`目录生成以下文件：

```
outputs/
├── excel_writer_test.xlsx      # Excel写入测试
├── quick_export.xlsx           # 快速导出测试
├── multi_index.xlsx            # 多层索引测试
├── hyperlinks.xlsx             # 超链接测试
├── append_test.xlsx            # 追加模式测试
├── styles_showcase.xlsx        # 样式展示
└── api_test_basic.xlsx         # API测试
```

## 🐛 常见问题

### Q1: ModuleNotFoundError

**问题**: 导入模块失败

**解决**:
```python
# 在notebook第一个单元格添加
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
```

### Q2: 找不到模板文件

**问题**: template.xlsx不存在

**解决**:
```bash
# 检查模板文件是否存在
ls -la ../hscredit/report/excel/template.xlsx

# 如果不存在，从原项目复制
cp ../../scorecardpipeline/scorecardpipeline/template.xlsx ../hscredit/report/excel/
```

### Q3: 版本不兼容

**问题**: 某些功能在特定Python版本不可用

**解决**: 参考notebook中的版本检查代码，使用条件判断

```python
import sys

if sys.version_info >= (3, 9):
    # Python 3.9+ 特性
    result = dict1 | dict2
else:
    # Python 3.8 兼容方案
    result = {**dict1, **dict2}
```

### Q4: Jupyter无法启动

**问题**: 端口被占用或权限问题

**解决**:
```bash
# 使用其他端口
jupyter notebook --port 8889

# 或清理缓存
jupyter notebook --clear-output
```

## 📝 编写新的验证Notebook

创建新模块验证notebook的模板：

```python
# 新模块验证模板

## 1. 环境检查
import sys
from pathlib import Path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

## 2. 导入模块
from hscredit.xxx import YYY

## 3. 基本功能测试
# ...

## 4. 高级功能测试
# ...

## 5. 性能测试
# ...

## 6. 错误处理测试
# ...

## 7. 总结
print("✅ 验证完成")
```

## 🔗 相关链接

- [迁移指南](../docs/MIGRATION_GUIDE.md)
- [API文档](../docs/)
- [项目计划](../PROJECT_PLAN.md)

---

**提示**: 建议按顺序运行notebook，确保环境正确配置。
