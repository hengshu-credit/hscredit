# hscredit 指标计算统一整合规划

## 版本信息
- **版本**：v1.0
- **日期**：2026-03-26
- **目标**：统一分散的指标计算逻辑，统一收口到`hscredit.core.metrics`

---

## 一、现状分析

### 1.1 指标分散情况

| 位置 | 指标 | 状态 |
|------|------|------|
| `hscredit.core.metrics.classification` | KS, AUC, Gini, KS_bucket, ROC_curve | ✅ 已统一 |
| `hscredit.core.metrics.stability` | PSI, CSI, PSI_table, CSI_table | ✅ 已统一 |
| `hscredit.core.metrics.importance` | IV, IV_table, gini_importance, entropy_importance | ✅ 已统一 |
| `hscredit.core.metrics.binning_metrics` | woe_iv_vectorized, compute_bin_stats, ks_by_bin, chi2_by_bin | ✅ 已统一 |
| `hscredit.core.rules.mining.metrics` | RuleMetrics类(含lift, badrate等) | ⚠️ 需整合 |
| `hscredit.core.viz.binning_plots` | _compute_bin_stats_from_raw_data | ⚠️ 需整合 |
| `hscredit.core.eda.*` | 各类中重复实现的IV/PSI计算 | ⚠️ 需整合 |
| 外部依赖(scorecardpy等) | IV, PSI, KS等 | ⚠️ 需迁移 |

### 1.2 问题诊断

1. **重复实现**：EDA模块的类中重复实现了IV/PSI计算逻辑
2. **分散维护**：rules/mining/metrics.py中独立实现了lift/badrate等指标
3. **外部依赖**：部分代码仍依赖scorecardpy等外部库的指标计算
4. **接口不统一**：不同模块的指标函数参数风格不一致

---

## 二、整合目标

```
┌─────────────────────────────────────────────────────────────────┐
│                     指标计算统一架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐     │
│   │           hscredit.core.metrics (统一入口)            │     │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │     │
│   │  │classification│ │  stability  │ │   importance    │  │     │
│   │  │  KS/AUC/Gini │ │ PSI/CSI     │ │ IV/WOE/Gini     │  │     │
│   │  └─────────────┘ └─────────────┘ └─────────────────┘  │     │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │     │
│   │  │   binning   │ │    lift     │ │   regression    │  │     │
│   │  │分箱指标/Chi2 │ │ Lift/Lift_table│ │ MSE/MAE/RMSE/R2 │  │     │
│   │  └─────────────┘ └─────────────┘ └─────────────────┘  │     │
│   └──────────────────────────────────────────────────────┘     │
│                              │                                   │
│           ┌──────────────────┼──────────────────┐               │
│           ▼                  ▼                  ▼               │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │  core.viz    │  │  core.eda    │  │ rules.mining │         │
│   │  可视化调用   │  │  EDA分析调用  │  │ 规则挖掘调用  │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、整合方案

### 3.1 新增指标模块

#### 3.1.1 lift.py - Lift指标模块（新增）

从`rules/mining/metrics.py`和外部代码中提取Lift计算逻辑。

```python
"""Lift指标计算模块.

提供模型/规则提升度评估指标。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


def lift(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5
) -> float:
    """计算Lift值.
    
    Lift = 命中样本坏账率 / 总体坏账率
    
    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率
    :param threshold: 分类阈值，默认0.5
    :return: Lift值
    """
    pass


def lift_table(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    bin_type: str = 'quantile'
) -> pd.DataFrame:
    """计算Lift详细统计表.
    
    按分箱计算每箱的Lift值、累积Lift值、坏账改善等指标。
    
    Returns:
        DataFrame: 列包括[分箱, 样本数, 好样本数, 坏样本数, 坏样本率, 
                         Lift值, 坏账改善, 累积Lift值, 累积坏账改善]
    """
    pass


def lift_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    percentages: List[float] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
) -> pd.DataFrame:
    """计算Lift曲线数据.
    
    在不同百分比切分点计算Lift值。
    
    :param percentages: 百分比切分点列表，默认[2%, 5%, 10%, 20%, 30%, 50%]
    :return: Lift曲线数据
    """
    pass


def rule_lift(
    y_true: Union[np.ndarray, pd.Series],
    rule_mask: Union[np.ndarray, pd.Series],
    amount: Optional[Union[np.ndarray, pd.Series]] = None
) -> dict:
    """计算规则的Lift指标.
    
    专为规则挖掘场景设计的Lift计算。
    
    :param y_true: 真实标签
    :param rule_mask: 规则命中掩码 (True/False或1/0)
    :param amount: 金额数据（可选），用于计算金额Lift
    :return: 包含lift、badrate、hit_rate等指标的字典
    """
    pass
```

#### 3.1.2 chi2.py - 卡方检验模块（新增）

从`binning_metrics.py`中提取并扩展chi2相关指标。

```python
"""卡方检验指标计算模块.

提供特征与目标变量的卡方独立性检验。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.stats import chi2_contingency


def chi2_test(
    feature: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    bins: int = 10
) -> Tuple[float, float, int]:
    """计算卡方统计量.
    
    :param feature: 特征变量
    :param y_true: 目标变量 (0/1)
    :param bins: 分箱数
    :return: (chi2统计量, p值, 自由度)
    """
    pass


def chi2_by_bin(
    bins: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """按分箱计算卡方统计量.
    
    :param bins: 分箱索引数组
    :param y: 目标变量
    :return: (chi2值, p值, 期望频数数组)
    """
    pass


def cramers_v(
    feature: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series]
) -> float:
    """计算Cramer's V系数.
    
    衡量分类变量与目标变量的关联强度，范围[0, 1]。
    
    :return: Cramer's V值
    """
    pass
```

#### 3.1.3 badrate.py - 坏账率指标模块（新增）

专为金融风控场景设计的坏账率计算模块。

```python
"""坏账率指标计算模块.

提供金融风控场景下的坏账率统计指标。
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List


def badrate(
    y_true: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """计算总体坏账率.
    
    :param y_true: 真实标签 (0/1)
    :param weights: 样本权重（可选）
    :return: 坏账率
    """
    pass


def badrate_by_group(
    y_true: Union[np.ndarray, pd.Series],
    group: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None
) -> pd.DataFrame:
    """按分组计算坏账率.
    
    :param group: 分组标签
    :return: 各组坏账率统计
    """
    pass


def badrate_trend(
    y_true: Union[np.ndarray, pd.Series],
    date: Union[np.ndarray, pd.Series],
    freq: str = 'M'
) -> pd.DataFrame:
    """计算坏账率时间趋势.
    
    :param date: 日期数据
    :param freq: 时间频率，'D'日/'W'周/'M'月
    :return: 时间趋势数据
    """
    pass
```

### 3.2 现有模块增强

#### 3.2.1 classification.py 增强

```python
# 新增函数
def accuracy(y_true, y_pred) -> float
def precision(y_true, y_pred) -> float
def recall(y_true, y_pred) -> float
def f1(y_true, y_pred) -> float
def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame  # DataFrame格式
```

#### 3.2.2 importance.py 增强

```python
# 新增函数
def iv_rating(iv: float) -> str:
    """根据IV值返回预测能力评级."""
    if iv < 0.02:
        return '无预测能力'
    elif iv < 0.1:
        return '弱预测能力'
    elif iv < 0.3:
        return '中等预测能力'
    elif iv < 0.5:
        return '强预测能力'
    else:
        return '极强(需检查)'


def batch_iv(df, features, target, n_bins=10) -> pd.DataFrame:
    """批量计算IV（从binning_metrics迁移并增强）."""
    pass
```

#### 3.2.3 stability.py 增强

```python
# 新增函数
def psi_rating(psi: float) -> str:
    """根据PSI值返回稳定性评级."""
    if psi < 0.1:
        return '非常稳定'
    elif psi < 0.25:
        return '相对稳定'
    else:
        return '不稳定'


def batch_psi(df, features, date_col, base_period, compare_periods) -> pd.DataFrame:
    """批量计算PSI."""
    pass
```

### 3.3 统一导出接口

更新`hscredit/core/metrics/__init__.py`：

```python
"""指标计算模块.

提供统一的模型评估指标计算功能，所有指标计算统一收口于此模块。

指标分类:
- 分类指标: KS, AUC, Gini, Accuracy, Precision, Recall, F1
- 稳定性指标: PSI, CSI, PSI_table, CSI_table
- 特征重要性: IV, WOE, IV_table, Gini_importance
- 分箱指标: compute_bin_stats, woe_iv_vectorized, ks_by_bin, chi2_by_bin
- Lift指标: lift, lift_table, lift_curve, rule_lift
- 坏账率指标: badrate, badrate_by_group, badrate_trend
- 回归指标: MSE, MAE, RMSE, R2
- 卡方检验: chi2_test, chi2_by_bin, cramers_v

所有指标计算支持:
- 单特征/单模型计算
- 批量计算
- 详细统计表输出
- 向量化高效计算
"""

# 分类指标
from .classification import (
    KS, AUC, Gini, KS_bucket, ROC_curve,
    accuracy, precision, recall, f1,
    confusion_matrix, classification_report
)

# 稳定性指标
from .stability import (
    PSI, CSI, PSI_table, CSI_table,
    psi_rating, batch_psi
)

# 特征重要性
from .importance import (
    IV, IV_table, iv_rating, batch_iv,
    gini_importance, entropy_importance
)

# 分箱指标
from .binning_metrics import (
    woe_iv_vectorized, compute_bin_stats,
    ks_by_bin, chi2_by_bin, divergence_by_bin,
    iv_for_splits, ks_for_splits
)

# Lift指标 (新增)
from .lift import (
    lift, lift_table, lift_curve, rule_lift
)

# 坏账率指标 (新增)
from .badrate import (
    badrate, badrate_by_group, badrate_trend
)

# 卡方检验 (新增)
from .chi2 import (
    chi2_test, chi2_by_bin, cramers_v
)

# 回归指标
from .regression import (
    MSE, MAE, RMSE, R2
)

__all__ = [
    # 分类指标
    "KS", "AUC", "Gini", "KS_bucket", "ROC_curve",
    "accuracy", "precision", "recall", "f1",
    "confusion_matrix", "classification_report",
    
    # 稳定性指标
    "PSI", "CSI", "PSI_table", "CSI_table",
    "psi_rating", "batch_psi",
    
    # 特征重要性
    "IV", "IV_table", "iv_rating", "batch_iv",
    "gini_importance", "entropy_importance",
    
    # 分箱指标
    "woe_iv_vectorized", "compute_bin_stats",
    "ks_by_bin", "chi2_by_bin", "divergence_by_bin",
    "iv_for_splits", "ks_for_splits",
    
    # Lift指标
    "lift", "lift_table", "lift_curve", "rule_lift",
    
    # 坏账率指标
    "badrate", "badrate_by_group", "badrate_trend",
    
    # 卡方检验
    "chi2_test", "chi2_by_bin", "cramers_v",
    
    # 回归指标
    "MSE", "MAE", "RMSE", "R2",
]
```

---

## 四、依赖迁移路径

### 4.1 EDA模块迁移

| 原代码 | 迁移后 |
|--------|--------|
| `hscredit.core.eda.DataOverview`中的指标计算 | 删除，直接使用`metrics` |
| `hscredit.core.eda.FeatureLabelRelationship.calculate_iv()` | 改为调用`metrics.IV_table()` |
| `hscredit.core.eda.StabilityAnalysis.calculate_psi()` | 改为调用`metrics.PSI_table()` |

### 4.2 Rules模块迁移

| 原代码 | 迁移后 |
|--------|--------|
| `hscredit.core.rules.mining.metrics.RuleMetrics._calculate_metrics()`中的lift计算 | 改为调用`metrics.rule_lift()` |
| `hscredit.core.rules.mining.metrics.RuleMetrics._calculate_psi()` | 改为调用`metrics.PSI()` |

### 4.3 Viz模块迁移

| 原代码 | 迁移后 |
|--------|--------|
| `hscredit.core.viz.binning_plots._compute_bin_stats_from_raw_data()` | 改为调用`metrics.compute_bin_stats()` |

---

## 五、实施计划

### 阶段1：新增指标模块（3天）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `metrics/lift.py` | 1天 | 从rules模块提取并增强 |
| `metrics/badrate.py` | 0.5天 | 新建坏账率指标 |
| `metrics/chi2.py` | 0.5天 | 从binning_metrics提取并增强 |
| `metrics/__init__.py` | 1天 | 更新统一导出接口 |

### 阶段2：现有模块增强（2天）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `metrics/classification.py` | 0.5天 | 新增accuracy/precision/recall/f1 |
| `metrics/importance.py` | 0.5天 | 新增iv_rating, batch_iv |
| `metrics/stability.py` | 0.5天 | 新增psi_rating, batch_psi |
| `metrics/binning_metrics.py` | 0.5天 | 优化并补充文档 |

### 阶段3：依赖迁移（3天）

| 模块 | 工作量 | 说明 |
|------|--------|------|
| `core.eda.*` | 1.5天 | 移除重复实现，改为调用metrics |
| `core.rules.mining.metrics` | 1天 | 改为调用metrics |
| `core.viz.binning_plots` | 0.5天 | 改为调用metrics |

### 阶段4：测试验证（2天）

- 指标计算结果一致性验证
- 性能测试
- 边界情况处理测试

**总计：10天**

---

## 六、使用示例

### 6.1 统一导入方式

```python
# 推荐：统一从metrics导入所有指标
from hscredit.core import metrics

# 计算IV
iv_value = metrics.IV(y_true, feature)
iv_table = metrics.IV_table(y_true, feature)

# 计算PSI
psi_value = metrics.PSI(expected, actual)

# 计算KS
ks_value = metrics.KS(y_true, y_prob)

# 计算Lift
lift_table = metrics.lift_table(y_true, y_prob)

# 计算坏账率
br = metrics.badrate(y_true)
br_trend = metrics.badrate_trend(y_true, date)

# 批量IV分析
iv_results = metrics.batch_iv(df, features=['age', 'income'], target='fpd15')
```

### 6.2 评级使用

```python
# IV评级
iv = metrics.IV(y_true, feature)
rating = metrics.iv_rating(iv)  # '中等预测能力'

# PSI评级
psi = metrics.PSI(expected, actual)
rating = metrics.psi_rating(psi)  # '非常稳定'
```

---

## 七、文件位置

本规划文档位于：`/Users/xiaoxi/CodeBuddy/hscredit/hscredit/Metrics_Integration_Plan.md`

---

**文档结束**
