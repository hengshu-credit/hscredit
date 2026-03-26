# hscredit 指标计算整合完成总结

## 整合日期
2026-03-26

---

## 一、新增指标模块

### 1.1 lift.py - Lift指标模块
**路径**: `hscredit/core/metrics/lift.py`

| 函数 | 功能 | 说明 |
|------|------|------|
| `lift()` | 计算Lift值 | 命中样本坏账率/总体坏账率 |
| `lift_table()` | Lift详细统计表 | 按分箱计算Lift、累积Lift、坏账改善 |
| `lift_curve()` | Lift曲线数据 | 不同覆盖率切分点的Lift值 |
| `rule_lift()` | 规则Lift指标 | 专为规则挖掘设计，支持金额加权 |

### 1.2 badrate.py - 坏账率指标模块
**路径**: `hscredit/core/metrics/badrate.py`

| 函数 | 功能 | 说明 |
|------|------|------|
| `badrate()` | 总体坏账率 | 支持样本加权 |
| `badrate_by_group()` | 分组坏账率 | 按维度分组统计 |
| `badrate_trend()` | 坏账率趋势 | 时间维度趋势分析 |
| `badrate_by_score_bin()` | 评分分箱坏账率 | 评估评分卡排序性 |

### 1.3 chi2.py - 卡方检验模块
**路径**: `hscredit/core/metrics/chi2.py`

| 函数 | 功能 | 说明 |
|------|------|------|
| `chi2_test()` | 卡方独立性检验 | 检验特征与目标是否独立 |
| `chi2_by_bin()` | 按分箱计算卡方 | 用于分箱合并检验 |
| `cramers_v()` | Cramer's V系数 | 衡量关联强度[0,1] |
| `chi2_merge_test()` | 两箱合并检验 | 卡方分箱算法使用 |
| `chi2_independence_test()` | DataFrame检验 | 对DataFrame列进行检验 |

---

## 二、统一导出接口更新

**路径**: `hscredit/core/metrics/__init__.py`

已更新统一导出接口，新增以下指标：

```python
# Lift指标
from .lift import lift, lift_table, lift_curve, rule_lift

# 坏账率指标
from .badrate import badrate, badrate_by_group, badrate_trend, badrate_by_score_bin

# 卡方检验
from .chi2 import chi2_test, chi2_by_bin, cramers_v, chi2_merge_test, chi2_independence_test
```

---

## 三、模块迁移完成

### 3.1 rules/mining/metrics.py 迁移
**状态**: ✅ 已完成

**变更内容**:
- 从`hscredit.core.metrics`导入统一指标
- `RuleMetrics._calculate_metrics()`改为调用`rule_lift()`
- `calculate_ks()`改为调用`KS()`
- `calculate_iv()`改为调用`IV()`
- `calculate_gini()`改为调用`Gini()`
- `calculate_lift_chart()`改为调用`lift_table()`
- 移除了内部重复实现的PSI计算

**导入语句**:
```python
from ....core.metrics import (
    KS, AUC, Gini,
    PSI, PSI_table,
    IV, IV_table,
    lift as metrics_lift,
    lift_table as metrics_lift_table,
    rule_lift,
    badrate
)
```

### 3.2 viz/binning_plots.py 迁移
**状态**: ✅ 已完成

**变更内容**:
- 添加导入`from ..metrics import compute_bin_stats`
- `_compute_bin_stats_from_raw_data()`改为包装函数
- 内部调用统一的`compute_bin_stats()`计算分箱统计
- 保留列名转换以兼容viz模块的期望

---

## 四、指标使用指南

### 4.1 统一导入方式

```python
# 推荐：统一从metrics导入所有指标
from hscredit.core import metrics

# 计算IV
iv_value = metrics.IV(y_true, feature)
iv_table = metrics.IV_table(y_true, feature)

# 计算PSI
psi_value = metrics.PSI(expected, actual)
psi_table = metrics.PSI_table(expected, actual)

# 计算KS
ks_value = metrics.KS(y_true, y_prob)

# 计算Lift
lift_value = metrics.lift(y_true, y_prob)
lift_table = metrics.lift_table(y_true, y_prob)

# 计算坏账率
br = metrics.badrate(y_true)
br_trend = metrics.badrate_trend(y_true, date)

# 卡方检验
chi2, p, dof = metrics.chi2_test(feature, y_true)
cv = metrics.cramers_v(feature, y_true)
```

### 4.2 各模块调用metrics的方式

| 模块 | 调用方式 | 示例 |
|------|---------|------|
| rules.mining | 从metrics导入 | `from ....core.metrics import rule_lift` |
| viz | 从metrics导入 | `from ..metrics import compute_bin_stats` |
| eda (新) | 从metrics导入 | `from ..metrics import IV, PSI, lift` |

---

## 五、指标清单汇总

### 5.1 完整指标列表（按类别）

| 类别 | 指标数量 | 指标列表 |
|------|---------|---------|
| 分类指标 | 7 | KS, AUC, Gini, KS_bucket, ROC_curve, confusion_matrix, classification_report |
| 稳定性指标 | 4 | PSI, CSI, PSI_table, CSI_table |
| 特征重要性 | 4 | IV, IV_table, gini_importance, entropy_importance |
| 分箱指标 | 9 | woe_iv_vectorized, compute_bin_stats, ks_by_bin, chi2_by_bin, divergence_by_bin, iv_for_splits, ks_for_splits, batch_iv, compare_splits_iv, compare_splits_ks |
| **Lift指标** | **4** | **lift, lift_table, lift_curve, rule_lift** |
| **坏账率指标** | **4** | **badrate, badrate_by_group, badrate_trend, badrate_by_score_bin** |
| **卡方检验** | **5** | **chi2_test, chi2_by_bin, cramers_v, chi2_merge_test, chi2_independence_test** |
| 回归指标 | 4 | MSE, MAE, RMSE, R2 |
| **总计** | **41** | |

---

## 六、后续建议

### 6.1 代码清理
- [ ] 清理`core.eda`模块中的重复指标实现（等待eda重写）
- [ ] 检查其他子模块是否还有分散的指标实现
- [ ] 更新文档中的指标使用示例

### 6.2 测试验证
- [ ] 验证rules模块的指标计算结果一致性
- [ ] 验证viz模块的分箱统计结果一致性
- [ ] 性能测试对比

### 6.3 文档更新
- [ ] 更新API文档
- [ ] 添加指标使用教程
- [ ] 添加指标选择指南

---

## 七、文件清单

### 新增文件
1. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/metrics/lift.py`
2. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/metrics/badrate.py`
3. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/metrics/chi2.py`

### 修改文件
1. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/metrics/__init__.py` - 更新统一导出
2. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/rules/mining/metrics.py` - 迁移到统一metrics
3. `/Users/xiaoxi/CodeBuddy/hscredit/hscredit/hscredit/core/viz/binning_plots.py` - 迁移到统一metrics

---

**整合完成！**
