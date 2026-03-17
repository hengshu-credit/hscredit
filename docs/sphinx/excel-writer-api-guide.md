# ExcelWriter API使用指南

## insert_df2sheet 方法

### 方法签名

```python
def insert_df2sheet(
    self,
    worksheet: Worksheet,
    data: pd.DataFrame,
    insert_space: Union[str, Tuple[int, int]],
    merge_column: Optional[Union[str, List[str]]] = None,
    header: bool = True,
    index: bool = False,
    auto_width: bool = False,
    fill: bool = False,
    merge: bool = False,
    merge_index: bool = True
) -> Tuple[int, int]:
```

### 参数说明

#### 必需参数

- **worksheet**: Worksheet对象
  - Excel工作表对象

- **data**: pd.DataFrame
  - 要插入的DataFrame数据

- **insert_space**: Union[str, Tuple[int, int]]
  - 插入位置，支持两种格式：
  - 字符串格式：如 `'B2'`, `'A1'`
  - 元组格式：如 `(2, 2)`, `(1, 1)` - (行号, 列号)

#### 可选参数

- **merge_column**: Union[str, List[str]], optional
  - 需要分组显示的列，默认None
  - 可以是单个列名或列名列表

- **header**: bool, optional
  - 是否写入DataFrame的列名，默认True

- **index**: bool, optional
  - 是否写入DataFrame的索引，默认False

- **auto_width**: bool, optional
  - 是否自动调整列宽，默认False

- **fill**: bool, optional
  - 是否填充样式，默认False

- **merge**: bool, optional
  - 是否合并单元格，默认False

- **merge_index**: bool, optional
  - 是否合并索引单元格，默认True

### 返回值

返回 `Tuple[int, int]`，表示插入内容后最后一行和最后一列的位置。

### 使用示例

#### 1. 基本用法

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd

# 创建DataFrame
df = pd.DataFrame({
    '特征': ['年龄', '收入', '学历'],
    'IV': [0.15, 0.23, 0.08],
    'KS': [0.32, 0.41, 0.25]
})

# 创建写入器
writer = ExcelWriter(theme_color='3f1dba')
ws = writer.get_sheet_by_name("Sheet")

# 使用字符串指定位置
writer.insert_df2sheet(ws, df, "B2")

# 或使用元组指定位置
writer.insert_df2sheet(ws, df, (2, 2))

# 保存
writer.save("output.xlsx")
```

#### 2. 带样式的写入

```python
# 填充样式
writer.insert_df2sheet(ws, df, "B2", fill=True)

# 自动调整列宽
writer.insert_df2sheet(ws, df, "B2", auto_width=True)

# 组合使用
writer.insert_df2sheet(ws, df, "B2", fill=True, auto_width=True)
```

#### 3. 带索引的写入

```python
# 设置索引的DataFrame
df_indexed = df.set_index('特征')

# 写入索引
writer.insert_df2sheet(ws, df_indexed, "B2", index=True)
```

#### 4. 合并单元格

```python
# 创建带重复值的数据
df_group = pd.DataFrame({
    '类别': ['A', 'A', 'B', 'B'],
    '特征': ['年龄', '收入', '年龄', '收入'],
    'IV': [0.15, 0.23, 0.18, 0.21]
})

# 合并指定列
writer.insert_df2sheet(ws, df_group, "B2", merge_column='类别')

# 合并多列
writer.insert_df2sheet(ws, df_group, "B2", merge_column=['类别', '特征'])
```

#### 5. 多层索引

```python
# 创建多层索引DataFrame
import numpy as np

multi_df = pd.DataFrame(
    np.random.randn(4, 6),
    columns=pd.MultiIndex.from_product([['组1', '组2'], ['A', 'B', 'C']]),
    index=pd.MultiIndex.from_product([['训练集', '测试集'], ['样本1', '样本2']])
)

# 写入多层索引
writer.insert_df2sheet(ws, multi_df, "B2", index=True, merge_index=True)
```

#### 6. 获取结束位置继续写入

```python
# 第一次写入
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
end_row, end_col = writer.insert_df2sheet(ws, df1, "B2")

# 在上次结束位置后继续写入
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
writer.insert_df2sheet(ws, df2, (end_row + 2, 2))  # 空一行
```

### 常见错误

#### ❌ 错误用法

```python
# 错误：使用start_row和start_col参数
writer.insert_df2sheet(ws, df, start_row=1, start_col=1)
# TypeError: insert_df2sheet() got an unexpected keyword argument 'start_row'
```

#### ✅ 正确用法

```python
# 正确：使用insert_space参数
writer.insert_df2sheet(ws, df, (1, 1))  # 元组格式
writer.insert_df2sheet(ws, df, "A1")    # 字符串格式
```

## insert_value2sheet 方法

### 方法签名

```python
def insert_value2sheet(
    self,
    worksheet: Worksheet,
    insert_space: Union[str, Tuple[int, int]],
    value: Any,
    style: Optional[str] = None,
    number_format: Optional[str] = None,
    font: Optional[Dict] = None,
    border: Optional[Dict] = None,
    fill: Optional[Dict] = None,
    alignment: Optional[Dict] = None
) -> None:
```

### 参数说明

- **worksheet**: Worksheet对象
- **insert_space**: 插入位置，支持字符串或元组
- **value**: 要插入的值
- **style**: 命名样式名称
- **number_format**: 数字格式
- **font**: 字体设置
- **border**: 边框设置
- **fill**: 填充设置
- **alignment**: 对齐设置

### 使用示例

```python
# 插入文本
writer.insert_value2sheet(ws, "A1", "标题")

# 插入带样式的文本
writer.insert_value2sheet(ws, "A1", "标题", style="header")

# 插入数字并格式化
writer.insert_value2sheet(ws, "B2", 0.8567, number_format="0.00%")

# 插入带千分位的数字
writer.insert_value2sheet(ws, "C3", 1234567, number_format="#,##0")
```

## merge_cells 方法

### 使用示例

```python
# 合并单元格
writer.merge_cells(ws, 1, 1, 1, 5)  # 合并A1:E1
```

## add_link 方法

### 使用示例

```python
# 添加超链接
writer.insert_value2sheet(ws, "A1", "跳转到Sheet2")
writer.add_link(ws, 1, 1, "Sheet2")
```

## 命名样式列表

ExcelWriter提供以下预定义样式：

- `header`, `header_1`, `header_2` - 标题样式
- `table_header`, `table_content`, `table_content_1` - 表格样式
- `text_left`, `text_center`, `text_right` - 文本对齐
- `number_percentage`, `number_thousands` - 数字格式
- 等26种样式

### 使用示例

```python
# 查看所有可用样式
print(writer.named_styles.keys())

# 使用命名样式
writer.insert_value2sheet(ws, "A1", "标题", style="header")
```

## 完整示例

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd
import numpy as np

# 创建数据
df = pd.DataFrame({
    '特征名称': ['年龄', '收入', '学历', '婚姻状况', '工作年限'],
    'IV值': [0.1523, 0.2341, 0.0892, 0.1234, 0.1876],
    'KS值': [0.3245, 0.4123, 0.2567, 0.3124, 0.3567],
    '缺失率': [0.0234, 0.0567, 0.0123, 0.0000, 0.0345],
})

# 创建写入器
writer = ExcelWriter(theme_color='3f1dba')
ws = writer.get_sheet_by_name("报告")

# 插入标题
writer.insert_value2sheet(ws, "A1", "特征筛选报告", style="header")
writer.merge_cells(ws, 1, 1, 1, 4)

# 插入数据
writer.insert_df2sheet(ws, df, "A3", fill=True, auto_width=True)

# 插入汇总信息
end_row, _ = writer.insert_df2sheet(ws, df, "A3")
writer.insert_value2sheet(ws, (end_row + 2, 1), f"特征总数: {len(df)}")

# 保存
writer.save("feature_report.xlsx")
```

---

**注意**: 在使用时请确保参数名称正确，避免使用`start_row`、`start_col`等错误参数。
