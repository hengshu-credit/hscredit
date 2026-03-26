# hscredit EDA模块整合重构规划 v3.0

## 版本信息
- **版本**：v3.0
- **日期**：2026-03-26
- **核心目标**：整合现有功能 + 函数式API重构 + 金融场景增强

---

## 一、现有功能盘点与复用策略

### 1.1 hscredit已实现的强大基础

#### 1.1.1 核心指标计算 (`hscredit.core.metrics`)

| 已有函数 | 功能 | 状态 | 复用方式 |
|---------|------|------|---------|
| `IV()` | 计算IV值 | ✅ 成熟 | 直接调用 |
| `IV_table()` | IV详细统计表 | ✅ 成熟 | 直接调用 |
| `PSI()` | 计算PSI值 | ✅ 成熟 | 直接调用 |
| `PSI_table()` | PSI详细统计表 | ✅ 成熟 | 直接调用 |
| `CSI()` | CSI特征稳定性 | ✅ 成熟 | 直接调用 |
| `KS()` | KS统计量 | ✅ 成熟 | 直接调用 |
| `AUC()` | AUC计算 | ✅ 成熟 | 直接调用 |
| `Gini()` | Gini系数 | ✅ 成熟 | 直接调用 |
| `batch_iv()` | 批量IV计算 | ✅ 已有 | 复用/包装 |
| `compute_bin_stats()` | 分箱统计计算 | ✅ 成熟 | 直接调用 |

#### 1.1.2 可视化 (`hscredit.core.viz`)

| 已有函数 | 功能 | 状态 | 复用方式 |
|---------|------|------|---------|
| `bin_plot()` | 分箱图 | ✅ 成熟 | 直接调用 |
| `ks_plot()` | KS/ROC曲线 | ✅ 成熟 | 直接调用 |
| `corr_plot()` | 相关性热力图 | ✅ 成熟 | 直接调用 |
| `hist_plot()` | 分布直方图 | ✅ 成熟 | 直接调用 |
| `psi_plot()` | PSI稳定性图 | ✅ 成熟 | 直接调用 |
| `distribution_plot()` | 时间分布图 | ✅ 成熟 | 直接调用 |

#### 1.1.3 数据描述 (`hscredit.utils.describe`)

| 已有函数 | 功能 | 状态 | 复用方式 |
|---------|------|------|---------|
| `feature_describe()` | 单特征描述统计 | ✅ 成熟 | 直接调用 |
| `groupby_feature_describe()` | 分组描述统计 | ✅ 成熟 | 直接调用 |

#### 1.1.4 EDA模块现有类 (`hscredit.core.eda`)

| 已有类 | 功能 | 状态 | 处理方式 |
|-------|------|------|---------|
| `DataOverview` | 数据概览 | ✅ 已实现 | 拆解为独立函数 |
| `TargetAnalysis` | 目标变量分析 | ✅ 已实现 | 拆解为独立函数 |
| `FeatureAnalysis` | 特征分析 | ✅ 已实现 | 拆解为独立函数 |
| `FeatureLabelRelationship` | 特征标签关系 | ✅ 已实现 | 拆解为独立函数 |
| `StabilityAnalysis` | 稳定性分析 | ✅ 已实现 | 拆解为独立函数 |
| `CorrelationAnalysis` | 相关性分析 | ✅ 已实现 | 拆解为独立函数 |
| `EDAReport` | EDA报告 | ✅ 已实现 | 拆解为独立函数 |

---

## 二、重构设计理念

### 2.1 核心设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                    EDA模块重构设计原则                        │
├─────────────────────────────────────────────────────────────┤
│ 1. 函数式API                                                │
│    - 去除所有类封装                                          │
│    - 独立函数，输入输出清晰                                   │
├─────────────────────────────────────────────────────────────┤
│ 2. 最大化复用                                                │
│    - 复用metrics/viz/describe的核心功能                       │
│    - EDA层只做批量整合和结果增强                              │
├─────────────────────────────────────────────────────────────┤
│ 3. DataFrame优先                                            │
│    - 所有分析结果以DataFrame返回                              │
│    - 统一中文列名                                            │
├─────────────────────────────────────────────────────────────┤
│ 4. 金融场景定制                                              │
│    - 专注信贷风控特殊需求                                     │
│    - 逾期率/Vintage/滚动率等特有分析                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块架构（函数式）

```
hscredit/core/eda_v3/
├── __init__.py                 # 统一导出所有函数
├── overview.py                 # 数据概览（5个函数）
├── target.py                   # 目标变量分析（6个函数）
├── feature.py                  # 特征分析（8个函数）
├── relationship.py             # 特征标签关系（7个函数）
├── correlation.py              # 相关性分析（4个函数）
├── stability.py                # 稳定性分析（5个函数）
├── vintage.py                  # Vintage分析（3个函数）
├── report.py                   # 综合报告（4个函数）
└── utils.py                    # 工具函数
```

---

## 三、详细函数设计

### 3.1 overview.py - 数据概览模块

基于 `DataOverview` 类拆解，复用 `feature_describe`

| 函数名 | 输入参数 | 输出 | 已有基础 | 新增内容 |
|--------|---------|------|---------|---------|
| `data_info()` | df | DataFrame | `DataOverview.basic_info()` | 转为函数式 |
| `missing_analysis()` | df, threshold=0 | DataFrame | `DataOverview.missing_analysis()` | 转为函数式 |
| `feature_summary()` | df, features=None | DataFrame | `feature_describe` | 批量处理 |
| `numeric_summary()` | df | DataFrame | `DataOverview.numeric_summary()` | 转为函数式 |
| `category_summary()` | df, max_categories=10 | DataFrame | `DataOverview.category_summary()` | 转为函数式 |

```python
def feature_summary(
    df: pd.DataFrame,
    features: List[str] = None,
    include_types: List[str] = None
) -> pd.DataFrame:
    """
    批量特征描述统计
    
    复用 hscredit.utils.describe.feature_describe，支持批量处理
    
    Returns:
        DataFrame: 列包括[特征名, 数据类型, 样本数, 非空数, 查得率, 
                         最小值, 最大值, 均值, 中位数, 标准差, 偏度, 峰度]
    """
    pass
```

### 3.2 target.py - 目标变量分析模块

基于 `TargetAnalysis` 类拆解，金融场景特有

| 函数名 | 输入参数 | 输出 | 说明 |
|--------|---------|------|------|
| `target_distribution()` | df, target_col | DataFrame | 目标变量分布 |
| `bad_rate_overall()` | df, target_col | float | 整体逾期率 |
| `bad_rate_by_dimension()` | df, target_col, dim_col | DataFrame | 分维度逾期率 |
| `bad_rate_trend()` | df, target_col, date_col | DataFrame | 逾期率时间趋势 |
| `bad_rate_by_bins()` | df, target_col, score_col, n_bins | DataFrame | 评分分箱逾期率 |
| `sample_distribution()` | df, date_col, target_col | DataFrame | 样本时间分布 |

```python
def bad_rate_trend(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    freq: str = 'M',
    dimensions: List[str] = None
) -> pd.DataFrame:
    """
    逾期率时间趋势分析
    
    支持按时间周期（月/周/日）统计逾期率，支持分维度对比
    
    Returns:
        DataFrame: 列包括[时间周期, 样本数, 好样本数, 坏样本数, 
                         逾期率(%), 环比变化, 同比变化]
    """
    pass
```

### 3.3 feature.py - 特征分析模块

基于 `FeatureAnalysis` 类拆解，复用分箱统计计算

| 函数名 | 输入参数 | 输出 | 已有基础 | 新增内容 |
|--------|---------|------|---------|---------|
| `feature_type_inference()` | df | DataFrame | - | 自动识别特征类型 |
| `numeric_distribution()` | df, feature | DataFrame | - | 数值分布统计 |
| `categorical_distribution()` | df, feature | DataFrame | - | 类别分布统计 |
| `outlier_detection()` | df, features, method='iqr' | DataFrame | - | 异常值检测 |
| `rare_category_detection()` | df, features, threshold=0.01 | DataFrame | - | 稀有类别检测 |
| `concentration_analysis()` | df, features | DataFrame | - | 集中度分析(Gini) |
| `feature_stability_over_time()` | df, features, date_col | DataFrame | - | 特征时序稳定性 |

### 3.4 relationship.py - 特征标签关系模块

基于 `FeatureLabelRelationship` 类拆解，复用 `IV_table` 和 `compute_bin_stats`

| 函数名 | 输入参数 | 输出 | 已有基础 | 新增内容 |
|--------|---------|------|---------|---------|
| `iv_analysis()` | df, feature, target | DataFrame | `IV_table()` | 单变量IV分析 |
| `batch_iv_analysis()` | df, features, target | DataFrame | `batch_iv()` | 批量IV+评级 |
| `woe_analysis()` | df, feature, target | DataFrame | `IV_table()` | WOE分箱分析 |
| `binning_bad_rate()` | df, feature, target | DataFrame | `compute_bin_stats()` | 分箱逾期率 |
| `monotonicity_check()` | df, feature, target | DataFrame | 已有实现 | 单调性检验 |
| `univariate_auc()` | df, feature, target | DataFrame | 已有实现 | 单变量AUC |
| `feature_importance()` | df, features, target | DataFrame | 已有实现 | 综合重要性排序 |

```python
def batch_iv_analysis(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    n_bins: int = 10,
    method: str = 'quantile',
    return_details: bool = False
) -> pd.DataFrame:
    """
    批量IV分析
    
    复用 hscredit.core.metrics.IV_table，增加预测能力评级
    
    Returns:
        DataFrame: 列包括[特征名, IV值, 预测能力, 分箱数, 备注]
        
    预测能力评级标准：
        - IV < 0.02: 无预测能力
        - 0.02-0.1: 弱预测能力
        - 0.1-0.3: 中等预测能力
        - 0.3-0.5: 强预测能力
        - IV > 0.5: 极强(需检查)
    """
    pass
```

### 3.5 correlation.py - 相关性分析模块

基于 `CorrelationAnalysis` 类拆解

| 函数名 | 输入参数 | 输出 | 已有基础 |
|--------|---------|------|---------|
| `correlation_matrix()` | df, features, method='pearson' | DataFrame | `corr_plot`底层 |
| `high_correlation_pairs()` | df, features, threshold=0.8 | DataFrame | 已有实现 |
| `correlation_filter()` | df, features, target, threshold | List[str] | 已有实现 |
| `vif_analysis()` | df, features | DataFrame | - |

### 3.6 stability.py - 稳定性分析模块

基于 `StabilityAnalysis` 类拆解，复用 `PSI_table`

| 函数名 | 输入参数 | 输出 | 已有基础 | 新增内容 |
|--------|---------|------|---------|---------|
| `psi_analysis()` | base_df, curr_df, feature | DataFrame | `PSI_table()` | 单变量PSI |
| `batch_psi_analysis()` | base_df, curr_df, features | DataFrame | `PSI()` | 批量PSI+评级 |
| `csi_analysis()` | base_df, curr_df, feature, target | DataFrame | `CSI_table()` | CSI分析 |
| `time_psi_tracking()` | df, features, date_col, base_period | DataFrame | - | 时序PSI追踪 |
| `stability_report()` | base_df, curr_df, features | DataFrame | - | 综合稳定性报告 |

```python
def batch_psi_analysis(
    df: pd.DataFrame,
    features: List[str],
    date_col: str,
    base_period: str,
    compare_periods: List[str],
    n_bins: int = 10
) -> pd.DataFrame:
    """
    批量PSI稳定性分析
    
    复用 hscredit.core.metrics.PSI_table，增加稳定性评级
    
    Returns:
        DataFrame: 列包括[特征名, 基准期, 对比期, PSI值, 稳定性评级]
        
    稳定性评级标准：
        - PSI < 0.1: 非常稳定
        - 0.1-0.25: 相对稳定
        - PSI > 0.25: 不稳定
    """
    pass
```

### 3.7 vintage.py - Vintage分析模块

金融场景特有，基于本地代码审查发现的方法

| 函数名 | 输入参数 | 输出 | 说明 |
|--------|---------|------|------|
| `vintage_analysis()` | df, vintage_col, mob_col, target | DataFrame | Vintage账龄分析 |
| `vintage_summary()` | df, vintage_col, mob_col, target | DataFrame | Vintage汇总统计 |
| `plot_vintage()` | vintage_df | Figure | Vintage曲线图 |

```python
def vintage_analysis(
    df: pd.DataFrame,
    vintage_col: str,      # 放款批次列
    mob_col: str,          # 账龄列
    target_col: str,       # 目标变量
    max_mob: int = 12      # 最大账龄
) -> pd.DataFrame:
    """
    Vintage账龄分析
    
    追踪不同放款批次（Vintage）随账龄（MOB）的风险表现变化
    
    Returns:
        DataFrame: 列包括[Vintage批次, MOB, 开户数, 坏账户数, 累积坏账率(%)]
        
    计算公式：
        累积坏账率(MOB=n) = 截至MOB=n时该队列中坏账户数 / 该队列开户总账户数
    """
    pass
```

### 3.8 report.py - 综合报告模块

整合所有模块生成综合报告

| 函数名 | 输入参数 | 输出 | 说明 |
|--------|---------|------|------|
| `eda_summary()` | df, target, features | Dict[DataFrame] | EDA摘要 |
| `generate_report()` | df, target, features, config | Dict | 完整报告 |
| `export_report_to_excel()` | report_dict, filepath | - | 导出Excel |
| `report_to_html()` | report_dict, filepath | - | 导出HTML |

---

## 四、本地代码功能整合

### 4.1 从代码审查中提取的方法

| 原代码方法 | 新函数名 | 归属模块 | 说明 |
|-----------|---------|---------|------|
| `sample_analy()` | `sample_distribution()` | target.py | 样本分布分析 |
| `raw_bi_var()` | `binning_bad_rate()` | relationship.py | 双变量分箱分析 |
| `score_analy()` | `model_effect_summary()` | report.py | 评分效果综合评估 |
| `cal_bin()` | `score_bin_analysis()` | target.py | 评分等频分析 |
| `分月PSI_V.py` | `time_psi_tracking()` | stability.py | 分月PSI追踪 |
| `批量画等距分箱图` | `batch_binning_plots()` | report.py | 批量分箱图 |
| `ModelEffectReport()` | `model_effect_report()` | report.py | 模型效果报告 |

### 4.2 风控特有方法清单

```python
# 目标分析
def bad_rate_trend()           # 逾期率趋势
def score_bin_analysis()        # 评分分箱分析
def sample_distribution()       # 样本分布

# Vintage分析  
def vintage_analysis()          # Vintage账龄分析
def roll_rate_analysis()        # 滚动率分析

# 稳定性分析
def time_psi_tracking()         # 分月PSI追踪
def csi_analysis()              # CSI特征稳定性

# 报告生成
def model_effect_report()       # 模型效果报告
def batch_binning_plots()       # 批量分箱图
```

---

## 五、函数接口规范

### 5.1 统一参数风格

```python
# 标准参数命名
df: pd.DataFrame              # 主数据集
features: List[str]           # 特征列名列表
target: str / target_col: str # 目标变量列名
date_col: str                 # 日期列名
n_bins: int = 10              # 分箱数
method: str = 'quantile'      # 分箱/计算方法
threshold: float              # 阈值参数
```

### 5.2 统一输出列名（中文）

```python
# 基础列名
COLUMN_NAMES = {
    'feature': '特征名',
    'feature_type': '特征类型',
    'count': '样本数',
    'non_null': '非空数',
    'missing_rate': '缺失率(%)',
    'coverage': '查得率(%)',
    
    # 统计指标
    'min': '最小值',
    'max': '最大值',
    'mean': '均值',
    'median': '中位数',
    'std': '标准差',
    'skew': '偏度',
    'kurt': '峰度',
    
    # 金融指标
    'iv': 'IV值',
    'iv_level': '预测能力',
    'psi': 'PSI值',
    'psi_level': '稳定性评级',
    'ks': 'KS值',
    'auc': 'AUC值',
    'gini': 'Gini系数',
    'woe': 'WOE值',
    
    # 风控特有
    'bad_count': '坏样本数',
    'good_count': '好样本数',
    'bad_rate': '逾期率(%)',
    'lift': '提升度',
    'bin': '分箱',
    'bin_range': '分箱区间',
}
```

### 5.3 评级标准

```python
# IV预测能力评级
IV_RATING = {
    (0, 0.02): '无预测能力',
    (0.02, 0.1): '弱预测能力',
    (0.1, 0.3): '中等预测能力',
    (0.3, 0.5): '强预测能力',
    (0.5, float('inf')): '极强(需检查)',
}

# PSI稳定性评级
PSI_RATING = {
    (0, 0.1): '非常稳定',
    (0.1, 0.25): '相对稳定',
    (0.25, float('inf')): '不稳定',
}

# VIF共线性评级
VIF_RATING = {
    (0, 5): '无共线性',
    (5, 10): '中度共线性',
    (10, float('inf')): '严重共线性',
}
```

---

## 六、实施计划

### 阶段1：基础函数拆解（5天）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| overview.py | 1天 | 拆解DataOverview类 |
| target.py | 1天 | 拆解TargetAnalysis类 |
| feature.py | 1天 | 拆解FeatureAnalysis类 |
| relationship.py | 1天 | 拆解FeatureLabelRelationship类 |
| utils.py | 1天 | 公共工具函数 |

### 阶段2：高级功能开发（5天）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| vintage.py | 2天 | Vintage/滚动率分析 |
| stability.py | 1天 | 批量PSI/CSI分析 |
| correlation.py | 1天 | VIF/相关性筛选 |
| report.py | 1天 | 综合报告生成 |

### 阶段3：整合测试（3天）

- 函数接口统一测试
- 输出格式一致性检查
- 与metrics/viz模块集成测试

**总计：13天**

---

## 七、与现有代码的整合策略

### 7.1 直接复用（无需改动）

```python
# 从metrics复用
from ..metrics import IV, IV_table, PSI, PSI_table, CSI
from ..metrics import KS, AUC, Gini
from ..metrics import batch_iv, compute_bin_stats

# 从viz复用
from ..viz import bin_plot, ks_plot, corr_plot, psi_plot

# 从describe复用
from ...utils.describe import feature_describe
```

### 7.2 包装增强

```python
def batch_iv_analysis(df, features, target, n_bins=10):
    """批量IV分析 - 基于metrics.batch_iv增强"""
    # 1. 调用底层batch_iv
    iv_results = batch_iv(df, features, target, bins=n_bins)
    
    # 2. 增加预测能力评级
    iv_results['预测能力'] = iv_results['IV值'].apply(iv_rating)
    
    # 3. 统一列名（中文）
    iv_results.columns = ['特征名', 'IV值', '预测能力', '分箱数']
    
    return iv_results
```

### 7.3 现有类迁移路径

```
DataOverview.basic_info()        -> data_info()
DataOverview.missing_analysis()  -> missing_analysis()
DataOverview.numeric_summary()   -> numeric_summary()

TargetAnalysis.target_distribution() -> target_distribution()
TargetAnalysis.bad_rate_analysis()   -> bad_rate_by_dimension()

FeatureLabelRelationship.calculate_iv()      -> iv_analysis()
FeatureLabelRelationship.batch_iv_analysis() -> batch_iv_analysis()
FeatureLabelRelationship.woe_analysis()      -> woe_analysis()
```

---

## 八、使用示例

### 8.1 批量IV分析

```python
import hscredit.core.eda_v3 as eda

# 批量IV分析
iv_result = eda.batch_iv_analysis(
    df,
    features=['age', 'income', 'score', 'device_risk'],
    target='fpd15'
)
print(iv_result)
# 输出:
#      特征名    IV值    预测能力    分箱数
# 0   score  0.4521   强预测能力    10
# 1  device_risk  0.2343  中等预测能力    10
# 2     age  0.1567  中等预测能力    10
# 3  income  0.0345   弱预测能力    10
```

### 8.2 逾期率趋势分析

```python
# 分月逾期率趋势
trend = eda.bad_rate_trend(
    df,
    target_col='fpd15',
    date_col='apply_month',
    dimensions=['channel', 'product_type']
)
```

### 8.3 Vintage分析

```python
# Vintage账龄分析
vintage = eda.vintage_analysis(
    df,
    vintage_col='issue_month',    # 放款月份
    mob_col='mob',                # 账龄
    target_col='ever_dpd30'       # 目标
)
```

### 8.4 综合报告

```python
# 生成完整EDA报告
report = eda.generate_report(
    df,
    target='fpd15',
    features=feature_list,
    date_col='apply_date',
    config={
        'iv_threshold': 0.02,
        'psi_threshold': 0.1,
        'correlation_threshold': 0.8
    }
)

# 导出Excel
eda.export_report_to_excel(report, 'eda_report.xlsx')
```

---

## 九、函数清单汇总

### 9.1 完整函数清单（42个函数）

| 模块 | 函数数量 | 函数列表 |
|------|---------|---------|
| overview.py | 5 | data_info, missing_analysis, feature_summary, numeric_summary, category_summary |
| target.py | 6 | target_distribution, bad_rate_overall, bad_rate_by_dimension, bad_rate_trend, bad_rate_by_bins, sample_distribution |
| feature.py | 8 | feature_type_inference, numeric_distribution, categorical_distribution, outlier_detection, rare_category_detection, concentration_analysis, feature_stability_over_time, feature_quality_score |
| relationship.py | 7 | iv_analysis, batch_iv_analysis, woe_analysis, binning_bad_rate, monotonicity_check, univariate_auc, feature_importance |
| correlation.py | 4 | correlation_matrix, high_correlation_pairs, correlation_filter, vif_analysis |
| stability.py | 5 | psi_analysis, batch_psi_analysis, csi_analysis, time_psi_tracking, stability_report |
| vintage.py | 3 | vintage_analysis, vintage_summary, roll_rate_analysis |
| report.py | 4 | eda_summary, generate_report, export_report_to_excel, report_to_html |
| **总计** | **42** | |

---

## 十、与调研报告的对应关系

| 调研报告 | 整合内容 |
|---------|---------|
| 本地建模代码审查报告.md | 15个分析方法全部纳入，拆解为独立函数 |
| research_report_finance_eda_methods.md | 25个金融风控方法全部纳入，特别是Vintage/滚动率分析 |
| EDA_Redesign_Plan_v2.md | 复用策略完全遵循，最大化利用已有功能 |
| GitHub EDA库调研 | 通用EDA方法参考ydata-profiling，但优先复用hscredit已有功能 |

---

## 十一、文件位置

本规划文档位于：`/Users/xiaoxi/CodeBuddy/hscredit/hscredit/EDA_Module_Integration_Plan.md`

---

**文档结束**
