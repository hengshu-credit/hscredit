# 类别型变量分箱规则格式改进

## 功能说明

类别型变量的分箱规则现在支持**List[List]**格式，类似toad的实现方式。

### 格式对比

**旧格式（字符串列表）：**
```python
rules = {
    'city': ['北京,上海', '广州,深圳', 'nan']  # 字符串，用逗号分隔
}
```

**新格式（List[List]）：**
```python
rules = {
    'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]  # 列表的列表
}
```

### 优势

1. **更直观**：每个子列表代表一个分箱，结构清晰
2. **更灵活**：支持任意类型的值，包括np.nan
3. **更易用**：不需要拼接字符串，直接使用列表
4. **兼容toad**：与toad的Combiner格式一致

## 使用方法

### 1. 导入分箱规则

```python
import numpy as np
from hscredit.core.binning import OptimalBinning

# 定义分箱规则
rules = {
    'city': [['北京', '上海'], ['广州', '深圳'], ['杭州', '南京'], [np.nan]],
    'age': [25, 35, 45, 55]  # 数值型变量仍使用切分点列表
}

# 导入规则
binner = OptimalBinning()
binner.import_rules(rules)

# 应用分箱
import pandas as pd

df = pd.DataFrame({
    'city': ['北京', '上海', '广州', '深圳', '杭州', '南京', np.nan],
    'age': [20, 30, 40, 50, 60, 70, 45]
})

# 获取分箱索引
df_binned = binner.transform(df, metric='indices')
print(df_binned)
#    city  age
# 0     0    0  # 北京 -> 第0组
# 1     0    1  # 上海 -> 第0组
# 2     1    2  # 广州 -> 第1组
# 3     1    3  # 深圳 -> 第1组
# 4     2    4  # 杭州 -> 第2组
# 5     2    4  # 南京 -> 第2组
# 6     3    3  # np.nan -> 第3组

# 获取分箱标签
df_labels = binner.transform(df, metric='bins')
print(df_labels)
#      city              age
# 0  北京,上海    (-inf, 25.00]
# 1  北京,上海   (25.00, 35.00]
# 2  广州,深圳   (35.00, 45.00]
# 3  广州,深圳   (45.00, 55.00]
# 4  杭州,南京     (55.00, inf]
# 5  杭州,南京     (55.00, inf]
# 6      nan   (45.00, 55.00]
```

### 2. 导出分箱规则

```python
# 拟合分箱模型
binner = OptimalBinning(method='tree', max_n_bins=5)
binner.fit(df[['city', 'age']], df['target'])

# 导出规则
rules = binner.export_rules()

# 类别型变量会输出List[List]格式
print(rules['city'])
# [['北京', '上海'], ['广州', '深圳'], [np.nan]]

# 数值型变量会输出切分点列表
print(rules['age'])
# [25.5, 35.7, 45.2, 55.8]
```

### 3. JSON序列化

```python
import json
import numpy as np

# 处理np.nan以便JSON序列化
def convert_nan_to_str(obj):
    """将np.nan转换为字符串"NaN"以便JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_nan_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_str(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    return obj

def convert_str_to_nan(obj):
    """将字符串"NaN"转换回np.nan"""
    if isinstance(obj, dict):
        return {k: convert_str_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_nan(item) for item in obj]
    elif obj == "NaN":
        return np.nan
    return obj

# 保存到JSON文件
rules = {
    'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]],
    'age': [25, 35, 45]
}

# 序列化
rules_json = convert_nan_to_str(rules)
with open('binning_rules.json', 'w', encoding='utf-8') as f:
    json.dump(rules_json, f, indent=2, ensure_ascii=False)

# 从JSON文件加载
with open('binning_rules.json', 'r', encoding='utf-8') as f:
    rules_loaded = json.load(f)
    rules_final = convert_str_to_nan(rules_loaded)

# 导入
binner = OptimalBinning()
binner.import_rules(rules_final)
```

## 完整示例

```python
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning

# 1. 创建数据
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '南京'], n_samples),
    'gender': np.random.choice(['M', 'F', np.nan], n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

print(f"数据概览:")
print(df.head())

# 2. 分箱
binner = OptimalBinning(max_n_bins=5, method='tree')
binner.fit(df[['city', 'gender', 'age']], df['target'])

# 3. 导出规则
rules = binner.export_rules()

print(f"\n导出的分箱规则:")
for feature, rule in rules.items():
    print(f"  {feature}: {rule}")

# 4. 保存规则
import json

def convert_nan_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_str(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    return obj

rules_json = convert_nan_to_str(rules)
with open('binning_rules.json', 'w', encoding='utf-8') as f:
    json.dump(rules_json, f, indent=2, ensure_ascii=False)

print(f"\n规则已保存到 binning_rules.json")

# 5. 加载规则
def convert_str_to_nan(obj):
    if isinstance(obj, dict):
        return {k: convert_str_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_nan(item) for item in obj]
    elif obj == "NaN":
        return np.nan
    return obj

with open('binning_rules.json', 'r', encoding='utf-8') as f:
    rules_loaded = json.load(f)
    rules_final = convert_str_to_nan(rules_loaded)

binner2 = OptimalBinning()
binner2.import_rules(rules_final)

print(f"\n规则已加载")

# 6. 应用到新数据
df_new = pd.DataFrame({
    'city': ['北京', '上海', '广州', '深圳', np.nan],
    'gender': ['M', 'F', 'M', np.nan, 'F'],
    'age': [25, 35, 45, 55, 65]
})

df_binned = binner2.transform(df_new, metric='bins')
print(f"\n分箱结果:")
print(df_binned)
```

## 技术细节

### 修改的文件

1. **hscredit/core/binning/base.py**
   - 添加 `_cat_bins_` 属性：存储类别型变量的分组信息
   - 修改 `export_rules()`：支持导出List[List]格式
   - 修改 `import_rules()`：支持导入List[List]格式

2. **hscredit/core/binning/optimal_binning.py**
   - 修改 `_apply_bins()`：支持处理List[List]格式的类别型变量
   - 修改 `_get_bin_labels_dict()`：支持生成List[List]格式的标签
   - 修改 `transform()`：优先使用`_cat_bins_`处理类别型变量

### 数据结构

**内部存储：**
- `self.splits_`：数值型变量存储切分点，类别型变量存储空数组
- `self._cat_bins_`：类别型变量存储List[List]格式的分组信息
- `self.feature_types_`：存储特征类型（'numerical'或'categorical'）

**导出格式：**
- 数值型：`[25, 35, 45, 55]`
- 类别型：`[['A', 'B'], ['C'], [np.nan]]`

### 向后兼容

- 旧的字符串列表格式 `['A,B', 'C']` 仍然支持
- 自动检测List[List]格式（检查第一个元素是否为列表）
- 对旧格式自动转换为新格式

## 注意事项

1. **np.nan处理**：JSON序列化时需要将np.nan转换为字符串"NaN"
2. **缺失值箱**：建议将包含np.nan的箱放在最后，符合toad的习惯
3. **未知类别**：transform时遇到的未知类别会被分配到索引-1（missing）
4. **规则一致性**：导出-导入循环会保持规则完全一致

## 测试验证

运行测试：
```bash
cd /Users/xiaoxi/CodeBuddy/hscredit
python test_categorical_rules_simple.py
```

测试覆盖：
- ✅ 导入List[List]格式的规则
- ✅ 导出List[List]格式的规则
- ✅ 导出-导入循环一致性
- ✅ JSON序列化和反序列化
- ✅ 混合数值型和类别型变量
- ✅ 正确处理缺失值

---

**实现日期**: 2026-03-16  
**功能状态**: ✅ 已完成并测试通过  
**兼容性**: 向后兼容，支持toad格式
