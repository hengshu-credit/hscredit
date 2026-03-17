# feature_bin_stats 灰客户剔除功能实现

## 功能概述

`feature_bin_stats` 函数现已支持自动判断target并剔除灰客户，参考 scorecardpipeline (scp) 的实现。

## 核心改进

### 1. 自动判断target

当传入 `overdue` 和 `dpd` 参数时，函数会自动生成target：

```python
# overdue='MOB1', dpd=7
# 自动生成: target = (data['MOB1'] > 7).astype(int)
# 坏样本定义: MOB1 > 7
```

### 2. 灰客户剔除

**参数**: `del_grey`

**说明**:
- `del_grey=False`: 保留所有样本，包括逾期天数在(0, dpd]的灰客户
- `del_grey=True`: 剔除灰客户，只保留好样本(overdue==0)和坏样本(overdue>dpd)

**实现逻辑** (参考scp):
```python
# 剔除条件
mask = (data[overdue] > dpd) | (data[overdue] == 0)
data = data[mask].reset_index(drop=True)
```

**样本分类**:
- **好样本**: overdue == 0
- **灰客户**: 0 < overdue <= dpd (当del_grey=True时剔除)
- **坏样本**: overdue > dpd

### 3. merge_columns动态调整

参考scp实现，根据`del_grey`动态调整合并列：

```python
if isinstance(del_grey, bool) and del_grey:
    # 剔除灰客户时，不同目标下样本数不同
    # 不合并"样本总数"和"样本占比"列
    merge_cols = ['指标名称', '指标含义', '分箱', '分箱标签']
else:
    # 保留所有样本时，可以合并样本数相关列
    merge_cols = ['指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比']
```

## 使用示例

### 示例1: 单目标分析（不剔除灰客户）

```python
from hscredit.analysis import feature_bin_stats

table = feature_bin_stats(
    data=df,
    feature='score',
    overdue='MOB1',
    dpd=7,
    del_grey=False,  # 保留所有样本
    method='quantile',
    max_n_bins=5
)

# 样本数: 1000
# 包含: 好样本 + 灰客户 + 坏样本
```

### 示例2: 单目标分析（剔除灰客户）

```python
table = feature_bin_stats(
    data=df,
    feature='score',
    overdue='MOB1',
    dpd=7,
    del_grey=True,  # 剔除灰客户
    method='quantile',
    max_n_bins=5
)

# 样本数: 700 (假设灰客户有300个)
# 仅包含: 好样本 + 坏样本
```

### 示例3: 多目标分析（剔除灰客户）

```python
table = feature_bin_stats(
    data=df,
    feature='score',
    overdue=['MOB1', 'MOB1'],
    dpd=[7, 15],
    del_grey=True,  # 剔除灰客户
    method='quantile',
    max_n_bins=5,
    verbose=1
)

# 输出:
# MOB1_7+: 样本数 700
# MOB1_15+: 样本数 650 (不同目标下样本数不同)

# 分箱详情列不包含"样本总数"和"样本占比"
```

## 测试验证

### 测试结果

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit
python test_feature_bin_stats_grey.py
```

**输出**:
```
数据概况:
总样本数: 1000
好样本数 (MOB1 == 0): 400
灰客户数 (0 < MOB1 <= 7): 300
坏样本数 (MOB1 > 7): 300

不剔除灰客户 - 样本总数: 1000
剔除灰客户 - 样本总数: 700
剔除的灰客户数: 300

✅ 所有测试通过！
```

### 验证要点

1. **样本数正确**: 剔除灰客户后样本数减少
2. **坏样本率变化**: 剔除灰客户后坏样本率上升
3. **分箱结果**: 分箱切分点可能不同（因为训练数据不同）
4. **多目标合并**: 当del_grey=True时，不合并样本数列

## 实现细节

### 修改的文件

1. **hscredit/hscredit/analysis/feature_analyzer.py**
   - 修改 `feature_bin_stats` 函数
   - 修改 `_merge_multi_target_tables` 函数
   - 添加灰客户剔除逻辑
   - 动态调整merge_columns

### 关键代码

#### 灰客户剔除逻辑

```python
if isinstance(del_grey, bool) and del_grey:
    # 只保留好样本和坏样本
    mask = (data[overdue] > dpd) | (data[overdue] == 0)
    data = data[mask].reset_index(drop=True)
```

#### merge_columns动态调整

```python
if isinstance(del_grey, bool) and del_grey:
    merge_cols = ['指标名称', '指标含义', '分箱', '分箱标签']
else:
    merge_cols = ['指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比']
```

#### verbose输出

```python
if verbose > 0:
    n_samples = len(analysis_data)
    n_bad = y.sum()
    bad_rate = y.mean()
    print(f"特征 {feat} - 目标 {target_name}: 样本数 {n_samples}, 坏样本数 {n_bad}, 坏样本率 {bad_rate:.4f}")
```

## 与scp对比

| 特性 | hscredit | scorecardpipeline |
|------|----------|-------------------|
| 自动判断target | ✅ | ✅ |
| 灰客户剔除 | ✅ | ✅ |
| merge_columns动态调整 | ✅ | ✅ |
| 多目标分析 | ✅ | ✅ |
| verbose输出 | ✅ | ✅ |
| 实现逻辑 | 与scp一致 | 参考基准 |

## 注意事项

1. **参数依赖**: `del_grey` 参数仅在 `overdue` 和 `dpd` 参数存在时有效
2. **样本数变化**: 剔除灰客户后，样本数会减少，需要确保剩余样本足够
3. **多目标分析**: 当 `del_grey=True` 时，不同目标下样本数可能不同
4. **分箱训练**: 第一个目标用于训练分箱器，后续目标复用分箱结果

## 相关文档

- 测试脚本: `test_feature_bin_stats_grey.py`
- 示例notebook: `hscredit/examples/04_feature_bin_stats.ipynb`
- 参考实现: `scorecardpipeline/scorecardpipeline/processing.py`

## 更新日志

**2026-03-16**:
- ✅ 实现灰客户剔除功能
- ✅ 自动判断target（overdue + dpd）
- ✅ 动态调整merge_columns
- ✅ 添加verbose输出
- ✅ 更新示例notebook
- ✅ 添加测试脚本
