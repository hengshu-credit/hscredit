# 分箱表美化展示功能使用指南

## 功能概述

在 Jupyter Notebook 中，特征分箱表通常会以原始的 DataFrame 形式展示，不够美观且难以快速获取关键信息。此功能通过 pandas Styler 对分箱表进行美化和高亮，使其更易读、更专业。

## 快速开始

### 1. 启用功能

```python
from hscredit.utils import enable_dataframe_show

# 启用 DataFrame 的 show 方法
enable_dataframe_show()
```

### 2. 基本使用

```python
from hscredit.report import feature_bin_stats

# 生成分箱表
table = feature_bin_stats(data, 'score', target='target', method='mdlp')

# 美化展示
table.show()
```

### 3. 紧凑模式

```python
# 只显示核心列
table.show(compact=True)
```

## 功能特性

### 🎨 自动高亮

- **坏样本率热力图**：红色表示高风险，绿色表示低风险
- **IV 值颜色编码**：
  - 红色：无预测力（IV < 0.02）
  - 黄色：弱预测力（0.02 ≤ IV < 0.1）
  - 绿色：中等预测力（0.1 ≤ IV < 0.3）
  - 深绿：强预测力（0.3 ≤ IV < 0.5）
  - 最深绿：超强预测力（IV ≥ 0.5）
- **LIFT 值热力图**：快速识别高价值分箱
- **KS 值热力图**：评估分箱区分能力

### 📊 支持多级表头

多目标分析时自动适配多级表头：

```python
table = feature_bin_stats(
    data, 'score',
    overdue=['MOB1', 'MOB3'],
    dpds=[7, 30]
)
table.show()
```

### 🔧 灵活配置

```python
# 自定义高亮
table.show(
    highlight_iv=True,        # 高亮 IV 值
    highlight_bad_rate=True,  # 高亮坏样本率
    highlight_lift=True,      # 高亮 LIFT 值
    highlight_ks=True,        # 高亮 KS 值
    compact=False             # 紧凑模式
)

# 自定义小数位数
table.show(precision={'坏样本率': 2, '分档WOE值': 2})

# 限制显示行数
table.show(max_rows=10)
```

### 💾 导出功能

```python
# 导出为 HTML（保留样式）
from hscredit.utils import BinTableDisplay

display_obj = BinTableDisplay(table)
display_obj.show().export_html('bin_table.html')

# 导出为 Excel（保留样式）
display_obj.to_excel('bin_table.xlsx')
```

## 高级用法

### 高亮特定分箱

```python
display_obj = BinTableDisplay(table)
display_obj.show().highlight_bins([0, 1], color='#FFEB9C')
```

### 使用函数式 API

```python
from hscredit.utils import style_bin_table

# 直接使用函数
styler = style_bin_table(table, compact=True)
# 在 Jupyter 中显示
styler
```

## 完整示例

```python
import pandas as pd
from hscredit.report import feature_bin_stats
from hscredit.utils import enable_dataframe_show

# 1. 启用功能
enable_dataframe_show()

# 2. 准备数据
data = pd.read_csv('your_data.csv')

# 3. 单特征分析
table = feature_bin_stats(
    data, 'credit_score',
    target='default',
    method='mdlp',
    max_n_bins=5,
    desc='信用评分'
)

# 4. 美化展示
table.show(compact=True)

# 5. 多特征对比
multi_table = feature_bin_stats(
    data,
    ['credit_score', 'age', 'income'],
    target='default',
    method='mdlp'
)
multi_table.show()

# 6. 多目标对比
multi_target_table = feature_bin_stats(
    data, 'credit_score',
    overdue=['MOB1', 'MOB3', 'MOB6'],
    dpds=[7, 30, 60]
)
multi_target_table.show()
```

## 效果对比

### 原始展示

```
   指标名称   指标含义  分箱  分箱标签  样本总数  好样本数  坏样本数  样本占比  ...  分档KS值
0  score  信用评分   0  (-inf, -1.23]  500  450  50  0.1  ...  0.123456
1  score  信用评分   1  (-1.23, 0.45]  1000  800  200  0.2  ...  0.234567
...
```

### 美化展示

- ✅ 表头深蓝色背景，白色字体
- ✅ 交替行背景色
- ✅ 坏样本率列带有热力图渐变
- ✅ IV 值根据预测力强度着色
- ✅ LIFT 和 KS 值带有渐变色
- ✅ 数值自动格式化，避免科学计数法
- ✅ 鼠标悬停高亮行

## 注意事项

1. **Jupyter 环境**：此功能专为 Jupyter Notebook 设计，在其他环境中可能无法正常显示
2. **自动注入**：调用 `enable_dataframe_show()` 后，所有 DataFrame 都会有 `.show()` 方法
3. **不影响原数据**：所有操作都在副本上进行，不会修改原始 DataFrame
4. **导出限制**：导出为 Excel 需要安装 `openpyxl` 库

## 依赖

- pandas >= 1.0
- IPython (Jupyter 环境)
- openpyxl (可选，用于导出 Excel)

## 相关文档

- [特征分箱统计表文档](./feature_bin_stats.md)
- [Jupyter Notebook 示例](../examples/bin_table_display_demo.ipynb)
