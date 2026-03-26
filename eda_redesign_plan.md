# hscredit EDA模块重构规划设计

## 1. 项目背景与目标

### 1.1 现状分析
- 现有EDA模块采用**类封装**形式，用户需要实例化类后调用方法
- 本地历史代码提供了丰富的金融风控分析函数，但分散在不同文件中
- 缺乏统一的函数式API设计，使用门槛较高

### 1.2 重构目标
- **去类化**：将类方法拆解为独立的纯函数
- **DataFrame优先**：所有分析结果以DataFrame为主要输出格式
- **中文输出**：指标名称、列名、说明全部使用中文化
- **可视化集成**：统一可视化接口，支持matplotlib/plotly
- **金融场景**：专注于信贷风控领域的特殊需求

## 2. 模块架构设计

### 2.1 文件组织结构
```
hscredit/core/eda/
├── __init__.py              # 统一导出所有函数
├── overview.py              # 数据概览
├── target.py                # 目标变量分析
├── feature.py               # 特征分析
├── binning.py               # 分箱与WOE
├── metrics.py               # IV/PSI/KS等指标
├── correlation.py           # 相关性分析
├── stability.py             # 稳定性分析
├── visualization.py         # 可视化工具
└── utils.py                 # 工具函数
```

### 2.2 函数命名规范
- 统一使用`snake_case`命名
- 动词+名词形式，如`calculate_iv`、`plot_distribution`
- 金融专用指标保留英文缩写（IV/PSI/WOE/KS/AUC）

## 3. 功能模块详细设计

### 3.1 数据概览模块 (overview.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `data_summary` | 数据基础信息汇总 | df | DataFrame |
| `missing_analysis` | 缺失值分析 | df, threshold | DataFrame |
| `dtype_summary` | 数据类型统计 | df | DataFrame |
| `duplicate_analysis` | 重复值检测 | df, subset | DataFrame |
| `constant_features` | 常数/准常数特征 | df, threshold | DataFrame |
| `data_quality_report` | 数据质量综合报告 | df | DataFrame |

**输出列名（中文）**:
- 特征名、数据类型、缺失数量、缺失率、唯一值数量、零值占比

### 3.2 目标变量分析模块 (target.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `target_distribution` | 目标变量分布 | df, target | DataFrame |
| `target_by_time` | 目标变量时间趋势 | df, target, date_col | DataFrame |
| `target_by_group` | 分组逾期率分析 | df, target, group_col | DataFrame |
| `bad_rate_analysis` | 综合逾期率分析 | df, target, date_col | DataFrame |
| `vintage_analysis` | 账龄分析(Vintage) | df, target, date_col | DataFrame |

**输出列名（中文）**:
- 时间段、样本总数、坏样本数、逾期率、环比变化

### 3.3 特征分析模块 (feature.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `numeric_stats` | 数值特征统计 | df, features | DataFrame |
| `category_stats` | 类别特征统计 | df, features | DataFrame |
| `outlier_detection` | 异常值检测 | df, feature, method | DataFrame |
| `concentration_analysis` | 集中度分析(Gini) | df, feature | DataFrame |
| `distribution_plot` | 分布可视化 | df, feature | matplotlib.Figure |

**输出列名（中文）**:
- 特征名、均值、标准差、偏度、峰度、最小值、25分位数、中位数、75分位数、最大值

### 3.4 分箱与WOE模块 (binning.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `equal_freq_binning` | 等频分箱 | df, feature, n_bins | DataFrame |
| `equal_width_binning` | 等距分箱 | df, feature, n_bins | DataFrame |
| `optimal_binning` | 最优分箱(决策树) | df, feature, target | DataFrame |
| `woe_transform` | WOE转换 | df, feature, target, bins | DataFrame |
| `binning_report` | 分箱详细报告 | df, feature, target | DataFrame |

**输出列名（中文）**:
- 分箱编号、分箱区间、样本数、好样本数、坏样本数、逾期率、占比、WOE值、IV值

### 3.5 核心指标计算模块 (metrics.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `calculate_iv` | 计算IV值 | df, feature, target, bins | float |
| `batch_iv` | 批量IV计算 | df, features, target | DataFrame |
| `calculate_psi` | 计算PSI值 | base, test, bins | float |
| `batch_psi` | 批量PSI计算 | df, features, base_period | DataFrame |
| `calculate_ks` | 计算KS值 | y_true, y_pred | float |
| `calculate_auc` | 计算AUC值 | y_true, y_pred | float |
| `ks_table` | KS明细表 | y_true, y_pred, n_bins | DataFrame |
| `lift_table` | Lift表 | y_true, y_pred, n_bins | DataFrame |
| `gini_coefficient` | Gini系数 | y_true, y_pred | float |

**输出列名（中文）**:
- 特征名、IV值、PSI值、KS值、AUC值、预测能力评级、稳定性评级

### 3.6 相关性分析模块 (correlation.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `correlation_matrix` | 相关性矩阵 | df, features, method | DataFrame |
| `high_correlation_pairs` | 高相关特征对 | df, features, threshold | DataFrame |
| `vif_analysis` | VIF多重共线性 | df, features | DataFrame |
| `correlation_filter` | 相关性筛选 | df, features, threshold | List |
| `plot_correlation_heatmap` | 相关性热力图 | corr_matrix | matplotlib.Figure |

**输出列名（中文）**:
- 特征1、特征2、相关系数、绝对相关系数、VIF值

### 3.7 稳定性分析模块 (stability.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `psi_analysis` | PSI稳定性分析 | df, feature, date_col | DataFrame |
| `time_stability` | 时间稳定性分析 | df, feature, target, date_col | DataFrame |
| `distribution_shift` | 分布漂移检测 | df, feature, date_col | DataFrame |
| `csi_analysis` | CSI特征稳定性 | df, feature, target, date_col | DataFrame |
| `stability_report` | 稳定性综合报告 | df, features, date_col | DataFrame |

**输出列名（中文）**:
- 特征名、基准月、对比月、PSI值、稳定性评级、分布漂移程度

### 3.8 可视化模块 (visualization.py)

| 函数名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `plot_histogram` | 直方图 | df, feature | Figure |
| `plot_boxplot` | 箱线图 | df, feature | Figure |
| `plot_badrate_trend` | 逾期率趋势图 | df, feature, target | Figure |
| `plot_binning_chart` | 分箱图(柱状+折线) | binning_df | Figure |
| `plot_ks_curve` | KS曲线 | y_true, y_pred | Figure |
| `plot_roc_curve` | ROC曲线 | y_true, y_pred | Figure |
| `plot_lift_curve` | Lift曲线 | y_true, y_pred | Figure |
| `plot_psi_trend` | PSI趋势图 | psi_df | Figure |
| `plot_iv_chart` | IV条形图 | iv_df | Figure |

## 4. 函数API设计规范

### 4.1 通用参数设计
```python
# 数据参数
df: pd.DataFrame                    # 输入数据
features: Union[str, List[str]]     # 特征列
target: str                         # 目标变量列
date_col: str                       # 日期列

# 分箱参数
n_bins: int = 10                    # 分箱数
binning_method: str = 'quantile'    # 分箱方法
special_values: List = None         # 特殊值处理

# 输出参数
return_details: bool = False        # 是否返回明细
output_path: str = None             # 输出路径
```

### 4.2 返回值规范
- **分析类函数**：返回DataFrame，包含中文化的列名
- **指标类函数**：返回float或包含指标值的字典
- **可视化函数**：返回matplotlib Figure对象
- **批量处理函数**：返回DataFrame，每行一个特征的结果

### 4.3 DataFrame列名规范（中文）
```python
# 基础统计列名
COLUMN_NAMES = {
    'feature': '特征名',
    'count': '样本数',
    'missing': '缺失数',
    'missing_rate': '缺失率(%)',
    'unique': '唯一值数',
    'mean': '均值',
    'std': '标准差',
    'min': '最小值',
    '25%': '25分位数',
    '50%': '中位数',
    '75%': '75分位数',
    'max': '最大值',
    'skew': '偏度',
    'kurt': '峰度',
    
    # 金融指标列名
    'iv': 'IV值',
    'psi': 'PSI值',
    'ks': 'KS值',
    'auc': 'AUC值',
    'woe': 'WOE值',
    'bin': '分箱',
    'bad_rate': '逾期率',
    'lift': '提升度',
    
    # 评级列名
    'iv_level': '预测能力',
    'psi_level': '稳定性',
}
```

## 5. 关键功能实现方案

### 5.1 IV计算函数
```python
def calculate_iv(df: pd.DataFrame, 
                 feature: str, 
                 target: str,
                 n_bins: int = 10,
                 method: str = 'quantile',
                 special_values: List = None) -> Dict:
    """
    计算特征IV值
    
    Returns:
        Dict: {
            '特征名': str,
            'IV值': float,
            '预测能力': str,  # 无预测能力/弱/中等/强/极强
            '分箱明细': DataFrame,
        }
    """
```

### 5.2 分箱报告函数
```python
def binning_report(df: pd.DataFrame,
                   feature: str,
                   target: str,
                   n_bins: int = 10,
                   method: str = 'optimal') -> pd.DataFrame:
    """
    生成分箱详细报告
    
    Returns:
        DataFrame: 列包括[分箱编号, 分箱区间, 样本数, 好样本数, 坏样本数, 
                         逾期率, 占比, WOE值, IV值]
    """
```

### 5.3 批量PSI分析
```python
def batch_psi(df: pd.DataFrame,
              features: List[str],
              date_col: str,
              base_period: str,
              test_periods: List[str] = None,
              n_bins: int = 10) -> pd.DataFrame:
    """
    批量计算多特征多时间段PSI
    
    Returns:
        DataFrame: 列包括[特征名, 基准月, 对比月, PSI值, 稳定性评级]
    """
```

## 6. 使用示例

### 6.1 基础使用
```python
import hscredit.eda as eda

# 数据概览
summary = eda.data_summary(df)
missing = eda.missing_analysis(df, threshold=0.05)

# IV分析
iv_result = eda.batch_iv(df, features=feature_list, target='fpd15')
print(iv_result[['特征名', 'IV值', '预测能力']])

# PSI分析
psi_result = eda.batch_psi(df, features=feature_list, 
                           date_col='apply_date',
                           base_period='2023-01',
                           test_periods=['2023-02', '2023-03'])

# 分箱报告
bin_report = eda.binning_report(df, feature='age', target='fpd15')
print(bin_report[['分箱区间', '样本数', '逾期率', 'WOE值']])
```

### 6.2 可视化
```python
# 分箱图
fig = eda.plot_binning_chart(bin_report)
fig.savefig('age_binning.png')

# IV排序图
fig = eda.plot_iv_chart(iv_result)

# 相关性热力图
corr_matrix = eda.correlation_matrix(df, feature_list)
fig = eda.plot_correlation_heatmap(corr_matrix)
```

## 7. 实施计划

### 阶段1：基础模块 (优先级：高)
- [ ] overview.py - 数据概览
- [ ] metrics.py - IV/PSI/KS计算
- [ ] binning.py - 分箱与WOE

### 阶段2：分析模块 (优先级：高)
- [ ] target.py - 目标变量分析
- [ ] feature.py - 特征分析
- [ ] stability.py - 稳定性分析

### 阶段3：高级模块 (优先级：中)
- [ ] correlation.py - 相关性分析
- [ ] visualization.py - 可视化

### 阶段4：测试与文档 (优先级：高)
- [ ] 单元测试
- [ ] 使用文档
- [ ] 示例Notebook

## 8. 与现有代码的整合

### 8.1 复用现有功能
- `hscredit.core.metrics.IV_table` -> `calculate_iv`
- `hscredit.core.metrics.PSI` -> `calculate_psi`
- `funcs.caliv` -> `batch_iv`
- `funcs.calpsi` -> `batch_psi`

### 8.2 迁移本地代码
- `funcs.py:pltKsAuc` -> `metrics.calculate_ks`
- `funcs.py:calvif` -> `correlation.vif_analysis`
- `funcs.py:value2woe` -> `binning.woe_transform`

## 9. 质量要求

### 9.1 代码规范
- 类型注解完整
- Docstring遵循Google风格
- 异常处理完善

### 9.2 性能要求
- 支持大数据集(100万+样本)
- 批量计算使用向量化操作
- 提供进度条显示(tqdm)

### 9.3 测试覆盖
- 单元测试覆盖率>80%
- 边界条件测试
- 异常情况测试

## 10. 附录

### 10.1 IV预测能力分级
| IV范围 | 评级 | 说明 |
|--------|------|------|
| < 0.02 | 无预测能力 | 不建议入模 |
| 0.02-0.1 | 弱预测能力 | 谨慎使用 |
| 0.1-0.3 | 中等预测能力 | 可用 |
| 0.3-0.5 | 强预测能力 | 优先入模 |
| > 0.5 | 极强(需检查) | 可能存在过拟合 |

### 10.2 PSI稳定性分级
| PSI范围 | 评级 | 说明 |
|---------|------|------|
| < 0.1 | 非常稳定 | 无需关注 |
| 0.1-0.25 | 相对稳定 | 持续监控 |
| > 0.25 | 不稳定 | 需要处理 |

### 10.3 相关性处理标准
| 相关系数 | 处理方式 |
|----------|----------|
| > 0.9 | 必须剔除 |
| 0.8-0.9 | 建议剔除 |
| 0.7-0.8 | 保留其一 |
| VIF > 10 | 存在多重共线性 |
