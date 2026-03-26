# hscredit/core/metrics 模块重构完成总结

## 重构概述

已完成 `hscredit/core/metrics` 模块的重新设计，解决了原有结构混乱的问题。

---

## 新模块结构

```
hscredit/core/metrics/
├── __init__.py           # 统一导出所有公共API
├── _base.py              # 内部工具函数（不对外暴露）
├── _binning.py           # 分箱统计内部实现
├── classification.py     # 分类模型评估指标
├── feature.py            # 特征评估指标（IV、卡方、重要性等）
├── stability.py          # 稳定性指标（PSI、CSI等）
├── finance.py            # 金融风控专用指标（Lift、坏账率等）
└── regression.py         # 回归模型评估指标
```

---

## 主要改进

### 1. 命名规范统一

**函数命名**：全部使用小写+下划线
- `ks`, `auc`, `gini` （原 `KS`, `AUC`, `Gini`）
- `iv`, `iv_table` （原 `IV`, `IV_table`）
- `psi`, `csi` （原 `PSI`, `CSI`）
- `mse`, `mae`, `rmse`, `r2` （原 `MSE`, `MAE`, `RMSE`, `R2`）

### 2. 模块职责清晰

| 模块 | 职责 | 函数数量 |
|------|------|---------|
| `_base.py` | 内部工具函数 | 4个内部函数 |
| `_binning.py` | 分箱统计计算 | 3个函数 |
| `classification.py` | 分类模型评估 | 11个函数 |
| `feature.py` | 特征评估 | 6个函数 |
| `stability.py` | 稳定性评估 | 7个函数 |
| `finance.py` | 金融风控指标 | 11个函数 |
| `regression.py` | 回归评估 | 4个函数 |

### 3. 功能合并

- **lift.py + badrate.py → finance.py**
  - 合并金融风控相关指标
  - 新增 `score_stats`, `score_stability`

- **importance.py + chi2.py → feature.py**
  - 合并特征评估相关功能
  - 新增 `feature_summary`

- **binning_metrics.py → _base.py + _binning.py**
  - 拆分为内部工具和对外接口

### 4. 新增功能

- `classification.py`: 新增 `accuracy`, `precision`, `recall`, `f1`
- `stability.py`: 新增 `psi_rating`, `batch_psi`
- `finance.py`: 新增 `score_stats`, `score_stability`
- `feature.py`: 新增 `feature_summary`

---

## 统一参数规范

### 分箱相关参数
```python
method: str = 'quantile'      # 分箱方法
max_n_bins: int = 10          # 最大分箱数
min_bin_size: float = 0.01    # 最小箱占比
```

### 所有函数支持
- 输入: `np.ndarray` 或 `pd.Series`
- 自动处理缺失值
- 统一的参数验证

---

## 使用示例

```python
from hscredit.core import metrics

# 分类指标
metrics.ks(y_true, y_prob)
metrics.auc(y_true, y_prob)
metrics.accuracy(y_true, y_pred)

# 特征评估
metrics.iv(y_true, feature)
metrics.iv_table(y_true, feature, method='mdlp', max_n_bins=5)
metrics.chi2_test(feature, y_true)

# 稳定性
metrics.psi(score_train, score_test)
metrics.psi_rating(0.15)  # 返回评级描述

# 金融风控
metrics.lift(y_true, y_prob)
metrics.lift_table(y_true, y_prob, method='best_iv')
metrics.badrate_by_score_bin(y_true, score)

# 回归指标
metrics.mse(y_true, y_pred)
metrics.r2(y_true, y_pred)
```

---

## 向后兼容性

为了保持向后兼容，旧命名仍然可用（标记为Deprecated）：

```python
# 以下命名仍然可用，但建议使用新命名
metrics.KS  ->  metrics.ks
metrics.AUC ->  metrics.auc
metrics.IV  ->  metrics.iv
metrics.PSI ->  metrics.psi
metrics.MSE ->  metrics.mse
# ... 等
```

---

## 删除的文件

- `lift.py` → 合并到 `finance.py`
- `badrate.py` → 合并到 `finance.py`
- `chi2.py` → 合并到 `feature.py`
- `importance.py` → 合并到 `feature.py`
- `binning_metrics.py` → 拆分为 `_base.py` + `_binning.py`

---

## 设计原则

1. **单一职责**：每个模块只负责一类指标
2. **命名一致**：全部小写+下划线
3. **参数统一**：分箱相关参数命名一致
4. **内部/外部分离**：`_` 前缀模块为内部实现
5. **向后兼容**：保留旧命名别名
