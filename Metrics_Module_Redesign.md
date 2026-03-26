# hscredit/core/metrics 模块重构设计

## 当前问题分析

### 1. 命名风格不一致
- 模块名：classification（名词）、chi2（缩写）、lift（指标名）
- 函数名：KS（大写）、lift_table（小写）、IV_table（驼峰+下划线）

### 2. 模块拆分过细
- lift.py 只有 4 个函数，badrate.py 只有 4 个函数，可以合并

### 3. 功能重叠
- importance.py 和 binning_metrics.py 都有 IV 计算逻辑
- stability.py 的 PSI 与 binning_metrics 有重叠

### 4. 内部/外部函数混杂
- 大量 `_` 前缀的内部函数暴露在同文件中

---

## 新模块结构设计

```
hscredit/core/metrics/
├── __init__.py           # 统一导出
├── _base.py              # 内部工具函数（文件前缀表示内部）
├── _binning.py           # 分箱统计内部实现
├── classification.py     # 分类模型评估指标
├── feature.py            # 特征评估指标（IV、重要性等）
├── stability.py          # 稳定性指标（PSI、CSI等）
├── finance.py            # 金融风控专用指标（Lift、坏账率等）
└── regression.py         # 回归模型评估指标
```

---

## 详细设计

### 1. `_base.py` - 内部工具函数
```python
# 所有内部工具函数，不对外暴露
# - _woe_iv_vectorized
# - _validate_inputs
# - _handle_missing_values
# - _create_bin_edges
# 等
```

### 2. `_binning.py` - 分箱统计内部实现
```python
# 分箱统计计算，供其他模块使用
# - compute_bin_stats        (对外)
# - _ks_by_bin               (内部)
# - _chi2_by_bin             (内部)
# - _divergence_by_bin       (内部)
# 等
```

### 3. `classification.py` - 分类模型评估
**命名规范：小写+下划线**

```python
# 核心指标
ks(y_true, y_prob) -> float
auc(y_true, y_prob) -> float
gini(y_true, y_prob) -> float

# 分桶统计
ks_bucket(y_true, y_prob, method='quantile', max_n_bins=10) -> DataFrame

# 曲线数据
roc_curve(y_true, y_prob) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

# 混淆矩阵和报告
confusion_matrix(y_true, y_pred) -> np.ndarray
classification_report(y_true, y_pred) -> Union[str, dict]

# 新增：常规分类指标
accuracy(y_true, y_pred) -> float
precision(y_true, y_pred) -> float
recall(y_true, y_pred) -> float
f1(y_true, y_pred) -> float
```

### 4. `feature.py` - 特征评估指标
**命名规范：小写+下划线**

```python
# IV相关
iv(y_true, feature, method='quantile', max_n_bins=10) -> float
iv_table(y_true, feature, method='quantile', max_n_bins=10) -> DataFrame

# 分箱统计
bin_stats(bins, y, bin_labels=None) -> DataFrame  # 从 _binning 导入

# 卡方检验
chi2_test(x, y) -> Tuple[float, float]  # (stat, p_value)
cramers_v(x, y) -> float

# 特征重要性（基于树模型）
feature_importance(X, y, method='gini') -> pd.Series

# 单特征统计
feature_summary(feature, y=None) -> DataFrame  # 描述统计+目标关系
```

### 5. `stability.py` - 稳定性指标
**命名规范：小写+下划线**

```python
# PSI
psi(expected, actual, method='quantile', max_n_bins=10) -> float
psi_table(expected, actual, method='quantile', max_n_bins=10) -> DataFrame
psi_rating(psi_value) -> str  # 返回稳定性评级

# CSI (PSI的别名)
csi(expected, actual) -> float
csi_table(expected, actual) -> DataFrame

# 新增：批量PSI
batch_psI(X_train, X_test, features=None) -> DataFrame
```

### 6. `finance.py` - 金融风控专用指标
**命名规范：小写+下划线**

```python
# ========== Lift 相关 ==========
lift(y_true, y_prob, threshold=0.5) -> float
lift_table(y_true, y_prob, method='quantile', max_n_bins=10) -> DataFrame
lift_curve(y_true, y_prob, percentages=[0.02, 0.05, ...]) -> DataFrame
rule_lift(y_true, rule_mask, amount=None) -> Dict

# ========== 坏账率相关 ==========
badrate(y_true, weights=None) -> float
badrate_by_group(y_true, group, weights=None) -> DataFrame
badrate_trend(y_true, date, freq='M') -> DataFrame
badrate_by_score_bin(y_true, score, method='quantile', max_n_bins=10) -> DataFrame

# ========== 评分卡相关 ==========
score_stats(score, y_true=None) -> Dict  # 评分统计信息
score_stability(score_train, score_test) -> DataFrame
```

### 7. `regression.py` - 回归指标（保持不变）
```python
mse(y_true, y_pred) -> float
mae(y_true, y_pred) -> float
rmse(y_true, y_pred) -> float
r2(y_true, y_pred) -> float
```

---

## __init__.py 导出设计

```python
"""指标计算模块 - 统一的模型评估指标入口

指标分类:
- 分类指标: ks, auc, gini, accuracy, precision, recall, f1, ks_bucket
- 特征评估: iv, iv_table, chi2_test, cramers_v, feature_importance, bin_stats
- 稳定性: psi, psi_table, csi, csi_table, batch_psi
- 金融风控: lift, lift_table, lift_curve, badrate, badrate_by_group
- 回归指标: mse, mae, rmse, r2

使用示例:
    >>> from hscredit.core import metrics
    >>> metrics.ks(y_true, y_prob)
    >>> metrics.iv(y_true, feature)
    >>> metrics.psi(score_train, score_test)
"""

# 分类指标
from .classification import (
    ks, auc, gini,
    accuracy, precision, recall, f1,
    ks_bucket, roc_curve,
    confusion_matrix, classification_report
)

# 特征评估
from .feature import (
    iv, iv_table,
    chi2_test, cramers_v,
    feature_importance, bin_stats,
    feature_summary
)

# 稳定性
from .stability import (
    psi, psi_table, psi_rating,
    csi, csi_table,
    batch_psi
)

# 金融风控
from .finance import (
    lift, lift_table, lift_curve, rule_lift,
    badrate, badrate_by_group, badrate_trend, badrate_by_score_bin,
    score_stats, score_stability
)

# 回归指标
from .regression import (
    mse, mae, rmse, r2
)

__all__ = [
    # 分类
    'ks', 'auc', 'gini',
    'accuracy', 'precision', 'recall', 'f1',
    'ks_bucket', 'roc_curve',
    'confusion_matrix', 'classification_report',
    # 特征
    'iv', 'iv_table',
    'chi2_test', 'cramers_v',
    'feature_importance', 'bin_stats',
    'feature_summary',
    # 稳定性
    'psi', 'psi_table', 'psi_rating',
    'csi', 'csi_table',
    'batch_psi',
    # 金融
    'lift', 'lift_table', 'lift_curve', 'rule_lift',
    'badrate', 'badrate_by_group', 'badrate_trend', 'badrate_by_score_bin',
    'score_stats', 'score_stability',
    # 回归
    'mse', 'mae', 'rmse', 'r2',
]
```

---

## 命名规范统一

### 函数命名
- **全部使用小写+下划线**：`ks`, `auc`, `iv_table`, `lift_curve`
- **避免混合格式**：不再使用 `KS`, `IV_table`, `AUC`

### 参数命名
- 分箱参数统一：
  - `method`: 分箱方法（'quantile', 'mdlp', 'best_iv' 等）
  - `max_n_bins`: 最大分箱数
  - `min_bin_size`: 最小箱占比
  - `bins`: 分箱边界（仅用于自定义分箱场景）

### 列名规范
- 中文列名保持一致：
  - `分箱`, `分箱标签`
  - `样本总数`, `好样本数`, `坏样本数`
  - `坏样本率`, `样本占比`
  - `分档WOE值`, `分档IV值`, `指标IV值`
  - `LIFT值`, `坏账改善`
  - `累积LIFT值`, `累积坏账改善`

---

## 向后兼容性

为了保持向后兼容，在 `__init__.py` 中保留旧命名别名（标记为Deprecated）：

```python
# 向后兼容（Deprecated）
from .classification import ks as KS, auc as AUC, gini as Gini
from .feature import iv as IV, iv_table as IV_table
from .stability import psi as PSI, csi as CSI

__all__.extend([
    'KS', 'AUC', 'Gini',  # Deprecated, use lowercase instead
    'IV', 'IV_table',      # Deprecated
    'PSI', 'CSI',          # Deprecated
])
```

---

## 实施步骤

1. **创建新文件**：`_base.py`, `_binning.py`, `finance.py`
2. **重构现有文件**：
   - `classification.py`: 重命名函数为小写，添加常规分类指标
   - `feature.py`: 合并 importance + chi2 + binning_metrics 对外接口
   - `stability.py`: 重命名函数为小写
3. **更新 `__init__.py`**: 统一导出
4. **删除旧文件**：`lift.py`, `badrate.py`, `chi2.py`, `importance.py`, `binning_metrics.py`
5. **测试验证**：确保所有函数正常工作
