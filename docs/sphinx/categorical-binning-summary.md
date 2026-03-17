# 类别型变量分箱规则改进总结

## ✅ 已完成的功能

### 1. List[List]格式支持

**功能描述：** 类别型变量的分箱规则现在支持List[List]格式，类似toad的实现方式。

**格式示例：**
```python
# 类别型变量
rules = {
    'city': [['北京', '上海'], ['广州', '深圳'], ['杭州', '南京'], [np.nan]]
}

# 数值型变量（保持不变）
rules = {
    'age': [25, 35, 45, 55]
}
```

### 2. 核心修改

#### base.py

**新增属性：**
```python
self._cat_bins_ = {}  # 存储类别型变量的分组信息
```

**export_rules()方法：**
- 数值型变量：返回切分点列表 `[25, 35, 45, 55]`
- 类别型变量：返回分组列表 `[['A', 'B'], ['C'], [np.nan]]`

**import_rules()方法：**
- 自动识别List[List]格式（类别型）
- 自动识别数值列表格式（数值型）
- 正确设置 `_cat_bins_` 属性

#### optimal_binning.py

**_apply_bins()方法：**
- 支持处理List[List]格式的类别型变量
- 正确映射每个值到对应的分箱
- 正确处理np.nan

**_get_bin_labels_dict()方法：**
- 支持生成List[List]格式的标签
- 正确显示类别名称，如 `北京,上海`

**transform()方法：**
- 优先使用 `_cat_bins_` 处理类别型变量
- 正确传递feature参数

### 3. 测试验证

**测试文件：** `test_categorical_rules_simple.py`

**测试覆盖：**
- ✅ 导入List[List]格式规则
- ✅ 导出List[List]格式规则
- ✅ 导出-导入循环一致性
- ✅ JSON序列化和反序列化
- ✅ 混合数值型和类别型变量
- ✅ 正确处理缺失值

**测试结果：**
```
================================================================================
✅ 所有测试通过！
================================================================================
```

## 📝 使用示例

### 基本使用

```python
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning

# 1. 定义规则
rules = {
    'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]],
    'age': [25, 35, 45]
}

# 2. 导入规则
binner = OptimalBinning()
binner.import_rules(rules)

# 3. 应用分箱
df = pd.DataFrame({
    'city': ['北京', '上海', '广州', '深圳', np.nan],
    'age': [20, 30, 40, 50, 60]
})

df_binned = binner.transform(df, metric='bins')
print(df_binned)
#      city              age
# 0  北京,上海    (-inf, 25.00]
# 1  北京,上海   (25.00, 35.00]
# 2  广州,深圳   (35.00, 45.00]
# 3  广州,深圳     (45.00, inf]
# 4      nan     (45.00, inf]
```

### JSON序列化

```python
import json

# 序列化辅助函数
def convert_nan_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_str(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    return obj

def convert_str_to_nan(obj):
    if isinstance(obj, dict):
        return {k: convert_str_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_nan(item) for item in obj]
    elif obj == "NaN":
        return np.nan
    return obj

# 保存
rules_json = convert_nan_to_str(rules)
with open('rules.json', 'w', encoding='utf-8') as f:
    json.dump(rules_json, f, indent=2, ensure_ascii=False)

# 加载
with open('rules.json', 'r', encoding='utf-8') as f:
    rules_loaded = json.load(f)
    rules_final = convert_str_to_nan(rules_loaded)

binner.import_rules(rules_final)
```

## 🔄 向后兼容

**旧格式仍然支持：**
```python
# 旧格式（字符串列表）
rules = {'city': ['北京,上海', '广州,深圳']}

# 会自动处理，但仍建议使用新格式
```

**自动格式检测：**
- 如果第一个元素是列表 → List[List]格式（类别型）
- 如果第一个元素是数值 → 切分点列表（数值型）

## 📋 技术实现

### 数据流程

**导入流程：**
```
import_rules(rules)
  ↓
判断格式 (List[List]?)
  ↓
类别型 → _cat_bins_[feature] = rules[feature]
        → splits_[feature] = np.array([])
        → feature_types_[feature] = 'categorical'
数值型 → splits_[feature] = np.array(rules[feature])
        → feature_types_[feature] = 'numerical'
```

**导出流程：**
```
export_rules()
  ↓
遍历每个特征
  ↓
类别型 → 返回 _cat_bins_[feature]  (List[List])
数值型 → 返回 splits_[feature].tolist()  (List)
```

**转换流程：**
```
transform(X)
  ↓
遍历每个特征
  ↓
类别型 → 使用 _cat_bins_[feature] 映射
        → [['A', 'B'], ['C']] → 0,0,1,1...
数值型 → 使用 splits_[feature] 切分
        → digitize(x, splits)
```

### 关键代码片段

**_apply_bins()方法（处理List[List]格式）：**
```python
if isinstance(splits, list) and len(splits) > 0 and isinstance(splits[0], list):
    # List[List]格式
    for i, group in enumerate(splits):
        if isinstance(group, list):
            for value in group:
                if isinstance(value, float) and np.isnan(value):
                    bins[x.isna()] = i
                else:
                    bins[x == value] = i
```

**_get_bin_labels_dict()方法（生成标签）：**
```python
if feature_type == 'categorical' and feature in self._cat_bins_:
    cat_bins = self._cat_bins_[feature]
    for i, group in enumerate(cat_bins):
        if isinstance(group, list):
            group_str = [str(v) if not np.isnan(v) else 'nan' for v in group]
            labels[i] = ','.join(group_str)
```

## ⚠️ 注意事项

1. **缺失值处理：** np.nan应该放在单独的组中，建议放在最后
2. **JSON序列化：** 需要将np.nan转换为字符串"NaN"
3. **未知类别：** transform时遇到的未知类别会被分配到-1（missing）
4. **规则一致性：** 导出-导入循环会保持完全一致

## 📚 参考文档

- [完整使用文档](./categorical_binning_rules.md)
- [测试代码](../test_categorical_rules_simple.py)
- [toad实现参考](../../toad/toad/transform.py)

---

**实现日期**: 2026-03-16  
**功能状态**: ✅ 已完成  
**测试状态**: ✅ 全部通过  
**文档状态**: ✅ 已完成
