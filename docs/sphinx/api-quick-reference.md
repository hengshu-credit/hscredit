# hscredit API 快速参考卡

## ExcelWriter 常见错误与正确用法

### ❌ 错误用法1: 参数未用元组包裹

```python
# 错误 ❌
writer.insert_df2sheet(ws, df, 1, 1)
# TypeError: cannot unpack non-iterable int object

# 错误 ❌
writer.insert_df2sheet(ws, df, start_row=1, start_col=1)
# TypeError: insert_df2sheet() got an unexpected keyword argument 'start_row'
```

### ✅ 正确用法

```python
# 方式1: 使用元组（推荐）
writer.insert_df2sheet(ws, df, (1, 1))  # 第1行，第1列
writer.insert_df2sheet(ws, df, (3, 2))  # 第3行，第2列（B列）

# 方式2: 使用Excel单元格坐标字符串
writer.insert_df2sheet(ws, df, "A1")  # A1单元格
writer.insert_df2sheet(ws, df, "B3")  # B3单元格
```

---

## insert_df2sheet 完整签名

```python
def insert_df2sheet(
    worksheet,      # 工作表对象
    data,           # DataFrame数据
    insert_space,   # 位置：(行, 列)元组 或 "A1"字符串
    merge_column=None,
    header=True,
    index=False,
    auto_width=True,
    fill=False,
    merge=False,
    merge_index=False
)
```

### 参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `worksheet` | Worksheet | 工作表对象 | `ws` |
| `data` | DataFrame | 要写入的数据 | `df` |
| `insert_space` | tuple或str | 起始位置 | `(1, 1)` 或 `"A1"` |
| `header` | bool | 是否写入列名 | `True` (默认) |
| `index` | bool | 是否写入索引 | `False` (默认) |
| `fill` | bool | 是否填充颜色 | `False` (默认) |
| `auto_width` | bool | 是否自动列宽 | `True` (默认) |
| `merge` | bool | 是否合并单元格 | `False` (默认) |

---

## 完整示例

### 示例1: 基础写入

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd

# 创建数据
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '广州']
})

# 创建写入器
writer = ExcelWriter()
ws = writer.get_sheet_by_name("Sheet")

# 写入数据（从A1开始）
writer.insert_df2sheet(ws, df, (1, 1))

# 保存文件
writer.save("output.xlsx")
```

### 示例2: 带样式写入

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd

# 创建数据
df = pd.DataFrame({
    '指标': ['准确率', '召回率', 'F1分数'],
    '值': [0.95, 0.88, 0.91]
})

# 创建写入器
writer = ExcelWriter()
ws = writer.get_sheet_by_name("Sheet")

# 写入标题
writer.insert_value2sheet(ws, (1, 1), "模型评估报告")
writer.merge_cells(ws, (1, 1), (1, 2))

# 写入数据（从第3行开始，带颜色填充）
writer.insert_df2sheet(ws, df, (3, 1), fill=True)

# 保存
writer.save("model_report.xlsx")
```

### 示例3: 多表写入

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd

# 创建多个数据表
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

# 创建写入器
writer = ExcelWriter()
ws = writer.get_sheet_by_name("Sheet")

# 写入第一个表（从A1开始）
writer.insert_df2sheet(ws, df1, (1, 1), fill=True)

# 写入第二个表（从A5开始，留一行空行）
writer.insert_df2sheet(ws, df2, (5, 1), fill=True)

# 保存
writer.save("multi_table.xlsx")
```

---

## 常见问题

### Q1: TypeError: cannot unpack non-iterable int object

**原因**: 参数未用元组包裹

```python
# 错误 ❌
writer.insert_df2sheet(ws, df, 1, 1)

# 正确 ✅
writer.insert_df2sheet(ws, df, (1, 1))
```

### Q2: TypeError: got an unexpected keyword argument 'start_row'

**原因**: 参数名错误

```python
# 错误 ❌
writer.insert_df2sheet(ws, df, start_row=1, start_col=1)

# 正确 ✅
writer.insert_df2sheet(ws, df, (1, 1))
```

### Q3: 如何跳过表头？

```python
writer.insert_df2sheet(ws, df, (1, 1), header=False)
```

### Q4: 如何写入索引？

```python
writer.insert_df2sheet(ws, df, (1, 1), index=True)
```

### Q5: 如何自动调整列宽？

```python
# 默认就是自动调整
writer.insert_df2sheet(ws, df, (1, 1), auto_width=True)

# 关闭自动调整
writer.insert_df2sheet(ws, df, (1, 1), auto_width=False)
```

---

## 参数位置转换表

| Excel坐标 | 元组坐标 | 说明 |
|-----------|----------|------|
| A1 | (1, 1) | 第1行，第1列 |
| B1 | (1, 2) | 第1行，第2列 |
| A2 | (2, 1) | 第2行，第1列 |
| C3 | (3, 3) | 第3行，第3列 |
| Z1 | (1, 26) | 第1行，第26列 |
| AA1 | (1, 27) | 第1行，第27列 |

---

## 性能提示

### 大数据量写入

```python
import pandas as pd
from hscredit.report.excel import ExcelWriter

# 创建大数据集
df = pd.DataFrame(np.random.randn(10000, 10))

# 性能测试
import time
start = time.time()

writer = ExcelWriter()
ws = writer.get_sheet_by_name("Sheet")
writer.insert_df2sheet(ws, df, (1, 1))
writer.save("large_data.xlsx")

elapsed = time.time() - start
print(f"写入10000行数据耗时: {elapsed:.2f}秒")
```

### 批量写入多个工作表

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd

# 创建多个数据集
datasets = {
    'Sheet1': pd.DataFrame({'A': [1, 2]}),
    'Sheet2': pd.DataFrame({'B': [3, 4]}),
    'Sheet3': pd.DataFrame({'C': [5, 6]})
}

# 批量写入
writer = ExcelWriter()
for sheet_name, df in datasets.items():
    ws = writer.get_sheet_by_name(sheet_name)
    writer.insert_df2sheet(ws, df, (1, 1))

writer.save("multi_sheet.xlsx")
```

---

## 记忆口诀

```
insert_df2sheet参数三要素：
1. worksheet - 工作表对象
2. data - DataFrame数据
3. insert_space - 位置用元组包裹

元组格式：(行号, 列号)
字符串格式："A1", "B2", "C3"

常见错误要避免：
❌ 单独传递两个数字
❌ 使用start_row参数名
✅ 用元组或字符串包裹
```

---

**更新时间**: 2024-01
**文档版本**: 1.0
