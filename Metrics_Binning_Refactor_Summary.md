# hscredit 分箱逻辑重构总结

## 重构目标
将 hscredit 库中分散的分箱逻辑统一收口到 `OptimalBinning`，分箱统计计算统一收口到 `compute_bin_stats`。

## 已替换的函数

### 1. `hscredit/core/metrics/lift.py`

#### `lift_table()`
**变更前：**
- 内置等频/等距分箱逻辑
- 手动计算分箱统计

**变更后：**
- 使用 `OptimalBinning` 进行分箱
- 使用 `compute_bin_stats` 计算分箱统计
- 参数风格与 `feature_bin_stats` 一致：
  - `method`: 分箱方法（如 'quantile', 'mdlp', 'best_iv' 等）
  - `max_n_bins`: 最大分箱数
  - `min_bin_size`: 每箱最小样本占比
  - `**kwargs`: 其他传递给 OptimalBinning 的参数

### 2. `hscredit/core/metrics/badrate.py`

#### `badrate_by_score_bin()`
**变更前：**
- 手动等频/等距分箱逻辑
- 手动计算每箱统计

**变更后：**
- 使用 `OptimalBinning` 进行分箱
- 使用 `compute_bin_stats` 计算分箱统计
- 参数风格统一：
  - `method`: 分箱方法
  - `max_n_bins`: 最大分箱数
  - `min_bin_size`: 每箱最小样本占比

### 3. `hscredit/core/metrics/classification.py`

#### `KS_bucket()`
**变更前：**
- 手动等频分箱逻辑
- 手动计算KS统计

**变更后：**
- 使用 `OptimalBinning` 进行分箱
- 使用 `compute_bin_stats` 计算分箱统计
- 参数风格统一：
  - `method`: 分箱方法
  - `max_n_bins`: 最大分箱数
  - `min_bin_size`: 每箱最小样本占比

### 4. `hscredit/core/viz/binning_plots.py`

#### `_compute_bin_stats_from_raw_data()`
**变更前：**
- 内置 quantile/uniform 分箱逻辑
- 调用 `compute_bin_stats` 计算统计

**变更后：**
- 使用 `OptimalBinning` 进行分箱（支持所有分箱方法）
- 使用 `compute_bin_stats` 计算分箱统计
- 参数变更：
  - `n_bins` -> `max_n_bins`
  - `method` 支持所有 OptimalBinning 方法
  - 新增 `min_bin_size` 参数

## 使用示例

### lift_table
```python
from hscredit.core.metrics import lift_table

# 等频分箱（默认）
result = lift_table(y_true, y_prob, max_n_bins=10)

# 使用MDLP分箱
result = lift_table(y_true, y_prob, method='mdlp', max_n_bins=5)

# 使用最优IV分箱
result = lift_table(y_true, y_prob, method='best_iv', max_n_bins=5, min_bin_size=0.05)
```

### badrate_by_score_bin
```python
from hscredit.core.metrics import badrate_by_score_bin

# 等频分箱（默认）
result = badrate_by_score_bin(y_true, score, max_n_bins=10)

# 使用CART分箱
result = badrate_by_score_bin(y_true, score, method='cart', max_n_bins=5)
```

### KS_bucket
```python
from hscredit.core.metrics import KS_bucket

# 等频分箱（默认）
result = KS_bucket(y_true, y_prob, max_n_bins=10)

# 使用最优KS分箱
result = KS_bucket(y_true, y_prob, method='best_ks', max_n_bins=5)
```

## 优势

1. **逻辑统一**：所有分箱操作都通过 `OptimalBinning` 完成，避免重复实现
2. **功能丰富**：支持所有 OptimalBinning 提供的分箱方法（mdlp、best_iv、cart等）
3. **维护简化**：分箱算法优化只需修改 OptimalBinning，无需改动各处
4. **参数一致**：分箱相关参数命名和风格统一

## 向后兼容性

- `lift_table`、`badrate_by_score_bin`、`KS_bucket` 的原有调用方式仍然兼容
- `_compute_bin_stats_from_raw_data` 的参数有调整（`n_bins` -> `max_n_bins`）
