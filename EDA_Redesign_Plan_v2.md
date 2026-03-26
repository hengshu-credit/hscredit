# hscredit EDA模块重构规划设计 v2.0

## 版本信息
- 版本：v2.0
- 日期：2026-03-26
- 更新说明：基于hscredit现有功能重新设计，避免重复开发

---

## 1. hscredit现有功能盘点

### 1.1 数据描述统计 (`hscredit.utils.describe`)

| 函数 | 功能 | 输出 |
|------|------|------|
| `feature_describe()` | 单特征描述统计 | Series（中文指标名） |
| `groupby_feature_describe()` | 分组描述统计 | DataFrame（并行处理） |

**已有中文指标名**：样本数、非空数、查得率、最小值、平均值、最大值、1%-99%分位数

### 1.2 核心指标计算 (`hscredit.core.metrics`)

| 函数 | 功能 | 输出 |
|------|------|------|
| `IV()` | 计算IV值 | float |
| `IV_table()` | IV详细统计表 | DataFrame |
| `PSI()` | 计算PSI值 | float |
| `PSI_table()` | PSI详细统计表 | DataFrame |

### 1.3 可视化 (`hscredit.core.viz`)

| 函数 | 功能 | 输出 |
|------|------|------|
| `bin_plot()` | 分箱图（含分箱统计计算） | Figure |
| `corr_plot()` | 相关性热力图 | Figure |
| `ks_plot()` | KS/ROC曲线 | Figure |
| `hist_plot()` | 分布直方图 | Figure |
| `psi_plot()` | PSI稳定性图 | Figure/DataFrame |
| `distribution_plot()` | 时间分布图 | Figure/DataFrame |

### 1.4 分箱统计计算 (`hscredit.core.viz.binning_plots`)

| 函数 | 功能 | 说明 |
|------|------|------|
| `_compute_bin_stats_from_raw_data()` | 从原始数据计算分箱统计 | 内部函数，支持等频/等距分箱 |

**已有分箱统计列名**：分箱、样本总数、好样本数、坏样本数、坏样本率、样本占比

---

## 2. EDA模块新增功能设计

基于已有功能，EDA模块主要提供**批量处理**、**综合分析**和**报告生成**能力。

### 2.1 模块架构（精简版）

```
hscredit/core/eda/
├── __init__.py              # 统一导出
├── overview.py              # 数据概览（新增批量/综合功能）
├── target.py                # 目标变量分析（新增）
├── feature.py               # 特征分析（批量扩展）
├── metrics.py               # 批量指标计算（基于现有IV/PSI）
├── correlation.py           # 相关性分析（新增筛选功能）
├── stability.py             # 稳定性分析（批量扩展）
└── report.py                # 综合报告生成（新增）
```

### 2.2 功能映射表

| 规划功能 | 已有基础 | 新增内容 | 文件 |
|----------|----------|----------|------|
| 数据概览 | `feature_describe` | 批量处理、质量评分 | overview.py |
| 单特征描述 | `feature_describe` | 批量处理、自动类型识别 | feature.py |
| IV计算 | `IV`, `IV_table` | 批量计算、预测能力评级 | metrics.py |
| PSI计算 | `PSI`, `PSI_table` | 批量计算、分月分析 | stability.py |
| 分箱统计 | `_compute_bin_stats` | 批量处理、WOE计算 | feature.py |
| 分箱可视化 | `bin_plot` | 批量绘制、组合报告 | report.py |
| 相关性 | `corr_plot` | 高相关筛选、VIF | correlation.py |
| 目标分析 | - | 逾期率趋势、Vintage | target.py |

---

## 3. 详细函数设计

### 3.1 overview.py - 数据概览（基于已有功能扩展）

| 函数名 | 已有基础 | 新增功能 | 输出 |
|--------|----------|----------|------|
| `data_summary` | - | 数据集整体信息统计 | DataFrame |
| `missing_analysis` | - | 缺失值批量分析 | DataFrame |
| `feature_overview` | `feature_describe` | 批量特征描述 | DataFrame |
| `data_quality_report` | - | 综合质量评分报告 | DataFrame |

```python
def feature_overview(df: pd.DataFrame, 
                     features: List[str] = None,
                     n_jobs: int = -1) -> pd.DataFrame:
    """
    批量特征描述统计
    
    基于 hscredit.utils.describe.feature_describe 实现批量处理
    
    Returns:
        DataFrame: 列包括[特征名, 样本数, 非空数, 查得率, 数据类型, 
                         最小值, 平均值, 最大值, 50%分位数(中位数)]
    """
    pass
```

### 3.2 feature.py - 特征分析（分箱WOE扩展）

| 函数名 | 已有基础 | 新增功能 | 输出 |
|--------|----------|----------|------|
| `binning_stats` | `_compute_bin_stats` | 批量分箱统计 | DataFrame |
| `woe_analysis` | - | WOE编码与逆编码 | DataFrame |
| `feature_stats` | `feature_describe` | 数值/类别特征统计 | DataFrame |

```python
def binning_stats(df: pd.DataFrame,
                  feature: str,
                  target: str,
                  n_bins: int = 10,
                  method: str = 'quantile') -> pd.DataFrame:
    """
    分箱统计分析（含WOE计算）
    
    基于 hscredit.core.viz.binning_plots._compute_bin_stats_from_raw_data
    增加WOE值和IV值计算
    
    Returns:
        DataFrame: 列包括[分箱, 样本总数, 好样本数, 坏样本数, 
                         坏样本率, 样本占比, WOE值, IV值]
    """
    pass
```

### 3.3 metrics.py - 批量指标计算（基于IV/PSI）

| 函数名 | 已有基础 | 新增功能 | 输出 |
|--------|----------|----------|------|
| `batch_iv` | `IV`, `IV_table` | 批量IV计算 | DataFrame |
| `batch_psi` | `PSI`, `PSI_table` | 批量PSI计算 | DataFrame |
| `calculate_ks` | `ks_plot`内部逻辑 | 提取KS计算 | float |
| `model_metrics` | - | 模型综合指标 | DataFrame |

```python
def batch_iv(df: pd.DataFrame,
             features: List[str],
             target: str,
             n_bins: int = 10) -> pd.DataFrame:
    """
    批量IV计算
    
    基于 hscredit.core.metrics.IV 和 IV_table
    
    Returns:
        DataFrame: 列包括[特征名, IV值, 预测能力, 分箱数]
        
    预测能力评级：
        - <0.02: 无预测能力
        - 0.02-0.1: 弱预测能力
        - 0.1-0.3: 中等预测能力
        - 0.3-0.5: 强预测能力
        - >0.5: 极强(需检查)
    """
    pass

def batch_psi(df: pd.DataFrame,
              features: List[str],
              date_col: str,
              base_period: str,
              test_periods: List[str]) -> pd.DataFrame:
    """
    批量PSI计算
    
    基于 hscredit.core.metrics.PSI 和 PSI_table
    
    Returns:
        DataFrame: 列包括[特征名, 基准月, 对比月, PSI值, 稳定性]
        
    稳定性评级：
        - <0.1: 非常稳定
        - 0.1-0.25: 相对稳定
        - >0.25: 不稳定
    """
    pass
```

### 3.4 target.py - 目标变量分析（新增）

| 函数名 | 功能 | 输出 |
|--------|------|------|
| `target_distribution` | 目标变量分布统计 | DataFrame |
| `bad_rate_trend` | 逾期率时间趋势 | DataFrame |
| `vintage_analysis` | 账龄(Vintage)分析 | DataFrame |

### 3.5 correlation.py - 相关性分析（扩展）

| 函数名 | 功能 | 输出 |
|--------|------|------|
| `correlation_filter` | 高相关性特征筛选 | List[str] |
| `vif_analysis` | VIF多重共线性分析 | DataFrame |
| `high_corr_pairs` | 高相关性特征对 | DataFrame |

### 3.6 stability.py - 稳定性分析（扩展）

| 函数名 | 已有基础 | 新增功能 | 输出 |
|--------|----------|----------|------|
| `psi_analysis` | `PSI_table` | 分月PSI追踪 | DataFrame |
| `csi_analysis` | - | CSI特征稳定性 | DataFrame |
| `time_stability` | - | 时间稳定性评估 | DataFrame |

### 3.7 report.py - 综合报告（新增）

| 函数名 | 已有基础 | 新增功能 | 输出 |
|--------|----------|----------|------|
| `generate_report` | 各模块函数 | 综合EDA报告 | Dict[DataFrame] |
| `export_to_excel` | - | 导出Excel报告 | ExcelWriter |
| `plot_summary` | `bin_plot`等 | 批量可视化 | Figure |

---

## 4. 复用策略与新增比例

### 4.1 复用已有功能（约60%）

```python
# 示例：batch_iv 基于 IV_table 实现
from ..core.metrics import IV_table

def batch_iv(df, features, target, n_bins=10):
    results = []
    for feature in features:
        # 复用现有的IV_table
        iv_table = IV_table(df[target], df[feature], bins=n_bins)
        iv_value = iv_table['IV贡献'].sum()
        
        results.append({
            '特征名': feature,
            'IV值': round(iv_value, 4),
            '预测能力': iv_strength(iv_value),  # 新增评级
            '分箱数': len(iv_table),
        })
    return pd.DataFrame(results)
```

### 4.2 新增功能（约40%）

| 类别 | 新增函数 | 说明 |
|------|----------|------|
| 批量处理 | batch_iv, batch_psi, feature_overview | 基于已有功能批量处理 |
| 综合分析 | data_quality_report, model_metrics | 整合多维度指标 |
| 风控特有 | vintage_analysis, bad_rate_trend | 金融场景特有分析 |
| 报告生成 | generate_report, export_to_excel | 综合报告输出 |

---

## 5. 中文输出规范

### 5.1 统一列名标准

```python
# 基础统计列名（与feature_describe保持一致）
BASE_COLUMNS = {
    'feature': '特征名',
    'count': '样本数',
    'non_null': '非空数',
    'coverage': '查得率',
    'dtype': '数据类型',
    'min': '最小值',
    'max': '最大值',
    'mean': '平均值',
    'median': '中位数',
    'std': '标准差',
    'skew': '偏度',
    'kurt': '峰度',
}

# 金融指标列名
FINANCE_COLUMNS = {
    'iv': 'IV值',
    'psi': 'PSI值',
    'ks': 'KS值',
    'auc': 'AUC值',
    'woe': 'WOE值',
    'bin': '分箱',
    'bad_rate': '坏样本率',
    'good_count': '好样本数',
    'bad_count': '坏样本数',
    'lift': '提升度',
}

# 评级列名
RATING_COLUMNS = {
    'iv_level': '预测能力',
    'psi_level': '稳定性',
    'quality_score': '质量评分',
}
```

### 5.2 评级标准（中文）

```python
IV_RATING = {
    (0, 0.02): '无预测能力',
    (0.02, 0.1): '弱预测能力',
    (0.1, 0.3): '中等预测能力',
    (0.3, 0.5): '强预测能力',
    (0.5, float('inf')): '极强(需检查)',
}

PSI_RATING = {
    (0, 0.1): '非常稳定',
    (0.1, 0.25): '相对稳定',
    (0.25, float('inf')): '不稳定',
}
```

---

## 6. 使用示例

### 6.1 批量IV分析

```python
import hscredit.eda as eda

# 批量计算IV（复用metrics.IV）
iv_result = eda.batch_iv(
    df, 
    features=['age', 'income', 'score'],
    target='fpd15'
)
print(iv_result)
#    特征名   IV值    预测能力  分箱数
# 0   age  0.2343  中等预测能力    10
# 1 income 0.1567  中等预测能力    10
# 2  score 0.4521   强预测能力    10
```

### 6.2 分箱WOE分析

```python
# 分箱统计（复用viz._compute_bin_stats，增加WOE）
bin_report = eda.binning_stats(
    df,
    feature='age',
    target='fpd15'
)
print(bin_report[['分箱', '样本总数', '坏样本率', 'WOE值']])
```

### 6.3 数据质量报告

```python
# 综合数据质量报告（整合多个功能）
report = eda.data_quality_report(df)
# 包括：缺失率、常数特征、数据类型分布、质量评分
```

### 6.4 综合EDA报告

```python
# 生成完整EDA报告（调用各模块）
full_report = eda.generate_report(
    df,
    target='fpd15',
    date_col='apply_date'
)

# 导出Excel
eda.export_to_excel(full_report, 'eda_report.xlsx')
```

---

## 7. 实施计划（基于现有功能）

### 阶段1：基础批量功能（P0）

| 文件 | 函数 | 工作量 | 依赖 |
|------|------|--------|------|
| overview.py | feature_overview | 1天 | utils.describe |
| metrics.py | batch_iv | 1天 | core.metrics.IV |
| metrics.py | batch_psi | 1天 | core.metrics.PSI |
| feature.py | binning_stats | 1天 | viz.binning_plots |

### 阶段2：分析与报告（P1）

| 文件 | 函数 | 工作量 | 依赖 |
|------|------|--------|------|
| target.py | bad_rate_trend, vintage_analysis | 2天 | - |
| correlation.py | correlation_filter, vif_analysis | 2天 | - |
| stability.py | csi_analysis, time_stability | 2天 | core.metrics.PSI |
| overview.py | data_quality_report | 1天 | 阶段1 |

### 阶段3：综合报告（P2）

| 文件 | 函数 | 工作量 | 依赖 |
|------|------|--------|------|
| report.py | generate_report | 2天 | 所有模块 |
| report.py | export_to_excel | 1天 | generate_report |

**总工作量**：约12天（比v1.0减少50%，因大量复用已有功能）

---

## 8. 与现有代码的整合

### 8.1 直接复用（无需修改）

```python
# hscredit.utils.describe
from ..utils.describe import feature_describe, groupby_feature_describe

# hscredit.core.metrics
from ..core.metrics import IV, IV_table, PSI, PSI_table

# hscredit.core.viz（可视化在report中调用）
from ..core.viz import bin_plot, corr_plot, ks_plot, psi_plot
```

### 8.2 扩展包装（新增批量/评级功能）

```python
# 包装现有功能，增加批量处理和中文评级
def batch_iv(df, features, target):
    # 循环调用 IV_table
    # 增加中文评级列
    pass
```

---

## 9. 文件清单对比

### v1.0（原始规划）vs v2.0（基于现有功能）

| 文件 | v1.0函数数 | v2.0函数数 | 变化说明 |
|------|-----------|-----------|----------|
| overview.py | 6 | 4 | 复用describe，减少基础函数 |
| target.py | 5 | 3 | 精简合并 |
| feature.py | 5 | 3 | 复用分箱计算 |
| binning.py | 5 | 0 | 合并到feature.py |
| metrics.py | 8 | 4 | 复用IV/PSI，专注批量 |
| correlation.py | 5 | 3 | 精简 |
| stability.py | 5 | 3 | 复用PSI |
| visualization.py | 9 | 0 | 直接使用viz模块 |
| report.py | 0 | 3 | 新增综合报告 |
| **总计** | **48** | **23** | **减少52%** |

---

## 10. 总结

### v2.0核心改进

1. **大量复用已有功能**（约60%）
   - `utils.describe` - 特征描述统计
   - `core.metrics` - IV/PSI计算
   - `core.viz` - 可视化

2. **专注于新增价值**（约40%）
   - 批量处理能力
   - 中文评级标准
   - 综合报告生成
   - 金融场景特有分析

3. **减少开发工作量**（从25天到12天）

4. **保持API简洁**（从48个函数到23个函数）

### 关键设计原则

- **不要重复造轮子**：复用hscredit已有功能
- **专注批量与整合**：EDA模块专注批量处理和综合分析
- **中文输出**：统一中文指标名和评级标准
- **函数式API**：独立函数，DataFrame优先

---

**文档结束**
