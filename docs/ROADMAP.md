# hscredit 规划方案（2026 重新梳理版）

> 版本：2026-04-11  
> 定位：面向金融信贷场景的完整风控建模工具包，覆盖策略分析人员与模型开发人员的全链路需求。  
> 说明：在全面梳理现有代码基础上重新编写，按「功能主题」组织，不沿用旧的 Phase 编号。

---

## 一、现有功能全景（2026-04 实际状态）

### 1.1 代码结构总览

```
hscredit/
├── core/
│   ├── binning/          # 17种分箱算法 + OptimalBinning 统一接口  ✅ 成熟
│   ├── encoders/         # 9种编码器（WOE/Target/Count/OHE/OE/Quantile/CatBoost/GBM/Cardinality）  ✅ 成熟
│   ├── selectors/        # 23种特征筛选 + CompositeFeatureSelector + StabilityAwareSelector  ✅ 成熟
│   ├── models/           # 完整建模生态  ✅ 成熟
│   │   ├── boosting/     # XGBoost / LightGBM / CatBoost / NGBoost
│   │   ├── classical/    # LogisticRegression（扩展版）/ sklearn 包装
│   │   ├── scorecard/    # ScoreCard（含 export_deployment_code SQL/Python/Java）
│   │   ├── losses/       # 7种自定义损失函数（Focal/WeightedBCE/CostSensitive/BadDebt/ApprovalRate/ProfitMax）
│   │   ├── tuning/       # ModelTuner / AutoTuner / TuningObjective（含 lift_head / head_ks / ks_lift_combined 等）
│   │   ├── evaluation/   # ModelReport / QuickModelReport / ModelExplainer（SHAP）/ CalibratedModel
│   │   └── rules/        # RuleSet / RulesClassifier / LogicOperator
│   ├── metrics/          # 全面指标库  ✅ 成熟
│   │   ├── classification.py  # ks / auc / gini / accuracy / f1 / ks_bucket / roc_curve
│   │   ├── feature.py         # iv / iv_table / chi2_test / cramers_v / feature_importance
│   │   ├── stability.py       # psi / csi / batch_psi / psi_rating
│   │   └── finance.py         # lift / lift_at / lift_table / lift_curve / lift_monotonicity_check / badrate
│   ├── viz/              # 50+ 图表  ✅ 成熟
│   │   ├── binning_plots.py   # bin_plot / ks_plot / hist_plot / psi_plot / bin_trend_plot
│   │   ├── risk_plots.py      # roc_plot / lift_plot / score_dist_plot / vintage_plot / threshold_analysis_plot
│   │   ├── variable_plots.py  # variable_iv_plot / variable_woe_trend_plot / variable_psi_heatmap / variable_missing_badrate_plot
│   │   ├── score_plots.py     # score_ks_plot / score_distribution_comparison_plot / score_badrate_bin_plot / score_lift_plot
│   │   ├── strategy_plots.py  # feature_trend_by_time / feature_drift_comparison / feature_cross_heatmap / segment_scorecard_comparison
│   │   └── style.py           # set_style / reset_style / 主题管理 / 中文字体自动检测
│   ├── eda/              # 结构化探索性分析  ✅ 基本成熟
│   │   ├── overview.py        # data_info / missing_analysis / feature_summary / data_quality_report
│   │   ├── target.py          # target_distribution / bad_rate_overall / bad_rate_by_dimension / bad_rate_trend
│   │   ├── feature.py         # numeric_distribution / categorical_distribution / outlier_detection / feature_stability_over_time
│   │   ├── relationship.py    # iv_analysis / batch_iv_analysis / woe_analysis / binning_bad_rate / univariate_auc
│   │   ├── correlation.py     # correlation_matrix / high_correlation_pairs / vif_analysis
│   │   ├── stability.py       # psi_analysis / batch_psi_analysis / csi_analysis / time_psi_tracking / psi_cross_analysis
│   │   ├── vintage.py         # vintage_analysis / vintage_summary / roll_rate_analysis
│   │   └── report.py          # eda_summary / generate_report / export_report_to_excel
│   ├── rules/            # 规则引擎  ✅ 基础完整
│   │   ├── rule.py            # Rule / get_columns_from_query / RuleSet / RulesClassifier
│   │   └── expr_optimizer.py  # optimize_expr / beautify_expr
│   ├── feature_engineering/  # 特征工程  ⚠️ 当前仅有 NumExprDerive
│   └── financial/        # 金融计算（fv/pv/pmt/nper/npv/irr/mirr）  ✅ 基础完整
├── report/               # 报告模块  ✅ 成熟
│   ├── feature_analyzer.py   # feature_bin_stats / auto_feature_analysis（Excel 特征报告）
│   ├── model_report.py       # QuickModelReport / auto_model_report / compare_models
│   ├── rule_analysis.py      # ruleset_analysis / multi_label_rule_analysis
│   ├── swap_analysis.py      # SwapAnalyzer / swap_analysis（规则置换四象限分析）
│   ├── overdue_predictor.py  # OverduePredictor / overdue_prediction_report
│   ├── population_drift.py   # population_drift（客群偏移 Excel 报告）
│   └── mining/               # SingleFeatureRuleMiner / MultiFeatureRuleMiner / MultiLabelRuleMiner / TreeRuleExtractor
├── excel/                # ExcelWriter / dataframe2excel  ✅
└── utils/                # seed_everything / 数据集 / describe / pandas 扩展 / logger  ✅
```

### 1.2 已实现核心 API 速查

| 模块 | 主要类/函数 |
|------|-----------|
| 分箱 | `OptimalBinning(method=...)`, `BestIVBinning`, `MDLPBinning`, `MonotonicBinning`, `ORBinning`, `GeneticBinning` 等 17 种 |
| 编码 | `WOEEncoder`, `TargetEncoder`, `GBMEncoder`, `CatBoostEncoder` 等 9 种 |
| 筛选 | `IVSelector`, `PSISelector`, `StabilityAwareSelector`, `CompositeFeatureSelector`, `BorutaSelector`, `StepwiseSelector` 等 23 种 |
| 模型 | `XGBoostRiskModel`, `LightGBMRiskModel`, `LogisticRegression`, `ScoreCard` |
| 调参 | `ModelTuner(objective='ks'/'lift_head'/'lift_head_monotonic'/'head_ks')`, `AutoTuner` |
| 损失 | `FocalLoss`, `BadDebtLoss`, `ApprovalRateLoss`, `ProfitMaxLoss` |
| 解释 | `ModelExplainer`（SHAP）, `CalibratedModel` |
| 部署 | `ScoreCard.export_deployment_code(language='sql'/'python'/'java')` |
| 报告 | `auto_model_report(model, X_train, y_train, ...)`, `auto_feature_analysis(...)` |
| 指标 | `ks`, `auc`, `lift_at(ratios=[...])`, `lift_monotonicity_check`, `psi`, `iv` |
| 可视化 | `bin_plot`, `ks_plot`, `score_ks_plot`, `variable_iv_plot`, `feature_trend_by_time`, `feature_cross_heatmap` |
| 规则 | `Rule`, `RulesClassifier`, `MultiLabelRuleMiner`, `SwapAnalyzer`, `population_drift` |
| EDA | `batch_iv_analysis`, `bad_rate_trend`, `vintage_analysis`, `psi_cross_analysis`, `eda_summary` |

---

## 二、差距分析（真实缺口）

经过完整代码梳理，以下是 **实际尚未实现** 或 **明显薄弱** 的部分：

### 2.1 缺失：EDA 策略域与客群域

| 缺失文件 | 缺失函数 | 说明 |
|----------|----------|------|
| `core/eda/population.py` | `population_profile` / `population_shift_analysis` / `population_monitoring_report` / `segment_drift_analysis` / `feature_cross_segment_effectiveness` | 客群画像、多期偏移分析，当前 EDA 无此域 |
| `core/eda/strategy.py` | `approval_badrate_tradeoff` / `score_strategy_simulation` / `vintage_performance_summary` / `roll_rate_matrix` / `label_leakage_check` / `multi_label_correlation` | 策略仿真与风险决策分析，当前完全空白 |
| `core/eda/stability.py` 扩充 | `feature_drift_report` / `score_drift_report` | 当前 stability.py 有 PSI 分析但缺统一偏移报告函数 |

### 2.2 缺失：特征工程模块

`core/feature_engineering/` 目前仅有 `NumExprDerive`（表达式衍生），以下能力完全空白：

| 缺失文件 | 缺失类 | 说明 |
|----------|--------|------|
| `time_features.py` | `TimeFeatureGenerator` | 时序特征（年/月/周/季度/节假日/距关键时间点天数） |
| `cross_features.py` | `CrossFeatureGenerator` | 交叉特征（差/比/积/对数比），sklearn Pipeline 兼容 |
| `preprocessing.py` | `MissingValueImputer` / `OutlierClipper` / `FeatureScaler` | 缺失值、异常值、标准化 Transformer，保留 DataFrame 列名 |

### 2.3 薄弱：规则模块

| 项目 | 当前状态 | 缺口 |
|------|----------|------|
| 规则组合挖掘 | `MultiFeatureRuleMiner` 在 `report/mining/` 中，但核心 `core/rules/` 无复杂规则组合能力 | 规则集覆盖率模拟、规则冲突检测 |
| 规则解释 | 无 | 单条规则的各期、各客群有效性追踪 |
| 规则库管理 | 无 | 规则版本管理、规则 A/B 评估 |

### 2.4 薄弱：分箱模块扩展

| 项目 | 当前状态 | 缺口 |
|------|----------|------|
| 批量分箱 Excel 输出 | `auto_feature_analysis` 支持，但 `OptimalBinning` 本身无 `batch_to_excel()` | 缺独立的 `OptimalBinning.batch_to_excel()` 快捷方法 |
| PSI 导向分箱 | 无 | `BestPSIBinning`：最小化训练/测试分布差异的分箱 |
| 自动分箱数选择 | `auto_select_method` 已有，但基于样本量的自动 `max_n_bins` 推断无 | `auto_select_bins(X, y, feature)` |

### 2.5 薄弱：特征筛选

| 项目 | 当前状态 | 缺口 |
|------|----------|------|
| `LiftSelector` | 已有，但 `ratio` 固定 | 增加 `ratio` 参数支持 `lift@1%/5%` 等自定义比例 |
| 筛选报告 Excel | 无 | `CompositeFeatureSelector.to_excel()` 输出筛选全流程报告 |

### 2.6 薄弱：模型与报告

| 项目 | 当前状态 | 缺口 |
|------|----------|------|
| SHAP 导出 Excel | `ModelExplainer` 有 SHAP 分析，但 `auto_model_report` 中 SHAP Sheet 暂未集成 | `include_shap=True` 时生成 SHAP 汇总 Sheet |
| 损失函数 | 7 种已有 | `OrdinalRankLoss`（评分排序一致性）/ `LiftFocusedLoss`（头部 LIFT 导向） |
| ScoreCard 分析 | 已有分值表、部署代码 | `score_segment_analysis(df, target)` 各评分段客群特征分析 |
| 多模型比较报告 | `compare_models(...)` 存在 | 对比 Excel 报告格式待完善 |

---

## 三、迭代规划（按价值主题）

### Theme A：完善 EDA 策略分析体系（高优先，最大价值缺口）

**目标**：补齐 `core/eda/` 缺失的两大分析域，让策略分析人员有完整的客群与策略分析工具链。

#### A-1  新增 `core/eda/population.py`

```python
def population_profile(
    df: pd.DataFrame,
    features: List[str],
    segment_col: str = None,
    date_col: str = None,
    target: str = None,
) -> pd.DataFrame:
    """客群画像：各特征均值/分位数/坏率，支持按客群/时间分组.
    
    :return: 行=特征+统计量，列=客群/时间分组
    """

def population_shift_analysis(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str],
    target: str = None,
    psi_threshold: float = 0.1,
) -> pd.DataFrame:
    """客群偏移分析：各特征 PSI/均值变化/偏移等级，输出摘要表.
    
    :return: 含 特征名/PSI/均值变化/偏移等级/建议 的 DataFrame
    """

def population_monitoring_report(
    df_base: pd.DataFrame,
    df_compare_list: List[pd.DataFrame],
    compare_labels: List[str],
    features: List[str],
    target: str = None,
    output_path: str = 'population_monitor.xlsx',
) -> str:
    """多期客群监控 Excel 报告（各期规模+坏率趋势 / PSI 时序矩阵 / 偏移 Top10）."""

def segment_drift_analysis(
    df: pd.DataFrame,
    date_col: str,
    segment_col: str,
    features: List[str],
    target: str = None,
    base_period: str = None,
) -> pd.DataFrame:
    """分客群、分时间的特征偏移三维矩阵（特征×时间×客群）."""

def feature_cross_segment_effectiveness(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    segment_col: str,
    metric: str = 'iv',     # 'iv' / 'ks' / 'auc' / 'lift@5%'
) -> pd.DataFrame:
    """特征在不同客群下的有效性矩阵（行=特征，列=客群，格=IV/KS/AUC）."""
```

#### A-2  新增 `core/eda/strategy.py`

```python
def approval_badrate_tradeoff(
    y_true,
    score,
    n_points: int = 100,
) -> pd.DataFrame:
    """通过率-坏率权衡曲线：阈值/通过率/拒绝率/通过坏率/拒绝坏率/坏率改善."""

def score_strategy_simulation(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    thresholds: List[float],
    amount_col: str = None,
) -> pd.DataFrame:
    """多阈值策略仿真：每个阈值下通过率/坏率/损失额/利润对比."""

def vintage_performance_summary(
    df: pd.DataFrame,
    vintage_col: str,
    mob_col: str,
    target_col: str,
    mob_points: List[int] = [3, 6, 9, 12],
) -> pd.DataFrame:
    """Vintage 矩阵摘要：各放款批次在指定 MOB 节点的坏率."""

def roll_rate_matrix(
    df: pd.DataFrame,
    dpd_t0: str,
    dpd_t1: str,
    bins: List[int] = [0, 1, 7, 15, 30, 60, 90, 120],
) -> pd.DataFrame:
    """DPD 迁移率矩阵：判断资产质量演变趋势."""

def label_leakage_check(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    threshold_iv: float = 0.5,
    threshold_auc: float = 0.9,
) -> pd.DataFrame:
    """标签泄露检查：IV/AUC 异常高的特征预警列表."""

def multi_label_correlation(
    df: pd.DataFrame,
    labels: List[str],
) -> pd.DataFrame:
    """多标签相关性（FPD15/FPD30/MOB3@30/MOB6@30 等）Spearman 矩阵+一致性率."""
```

#### A-3  扩充 `core/eda/stability.py`

```python
def feature_drift_report(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str] = None,
    method: str = 'psi',     # 'psi' / 'ks' / 'wasserstein'
    psi_bins: int = 10,
) -> pd.DataFrame:
    """批量特征偏移报告：所有特征偏移指标，按偏移程度降序，标注等级."""

def score_drift_report(
    score_base: pd.Series,
    score_target: pd.Series,
    y_base: pd.Series = None,
    y_target: pd.Series = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """评分偏移综合报告：PSI + 分布对比 + 坏率变化（若传入标签）."""
```

#### A-4  更新 `eda/__init__.py` 和顶层 `hscredit/__init__.py`

---

### Theme B：扩充特征工程模块（高优先，当前空白最大）

**目标**：将 `core/feature_engineering/` 从单一的 `NumExprDerive` 扩展为完整的 sklearn-compatible Transformer 体系。

#### B-1  `core/feature_engineering/time_features.py`

```python
class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """时序特征生成器（sklearn Pipeline 兼容）.

    :param date_col: 日期列名
    :param features: 生成的时序特征列表，支持：
        'year' / 'month' / 'day' / 'weekday' / 'quarter' /
        'is_weekend' / 'is_month_end' / 'is_month_start' /
        'days_since_epoch' / 'days_to_year_end' /
        'days_since_{ref_col}' （计算与某参考日期列的天数差）
    :param ref_cols: 参考日期列名列表，用于计算天数差

    Example:
        >>> gen = TimeFeatureGenerator(
        ...     date_col='apply_date',
        ...     features=['year', 'month', 'weekday', 'is_weekend',
        ...               'days_since_epoch', 'quarter']
        ... )
        >>> X_new = gen.fit_transform(X)
    """
```

#### B-2  `core/feature_engineering/cross_features.py`

```python
class CrossFeatureGenerator(BaseEstimator, TransformerMixin):
    """交叉特征生成器.

    :param pairs: 特征对列表 [('income', 'debt'), ...]
    :param operations: 运算列表，支持 'ratio' / 'diff' / 'product' / 'log_ratio' / 'square_diff'
    :param naming: 命名规则，'auto' 自动命名，或自定义 lambda

    Example:
        >>> gen = CrossFeatureGenerator(
        ...     pairs=[('income', 'debt'), ('age', 'credit_limit')],
        ...     operations=['ratio', 'diff']
        ... )
        >>> X_new = gen.fit_transform(X)
    """
```

#### B-3  `core/feature_engineering/preprocessing.py`

```python
class MissingValueImputer(BaseEstimator, TransformerMixin):
    """缺失值填充器.

    :param strategy: 'mean' / 'median' / 'mode' / 'constant' / 'model'（用简单模型预测填充）
    :param fill_value: strategy='constant' 时的填充值
    :param infer_per_column: 是否对每列单独推断策略（True 时按列数据类型自动选择）

    保留 DataFrame 格式和列名，兼容 sklearn Pipeline。
    """

class OutlierClipper(BaseEstimator, TransformerMixin):
    """异常值截断.

    :param method: 'quantile' / 'iqr' / 'fixed'
    :param lower_quantile: 下截断分位数，默认 0.01
    :param upper_quantile: 上截断分位数，默认 0.99
    :param clip_value: method='fixed' 时的 (min, max) 边界

    保留 DataFrame 格式，fit 阶段记录截断边界，transform 阶段应用。
    """

class FeatureScaler(BaseEstimator, TransformerMixin):
    """特征标准化/归一化（封装 sklearn，保留 DataFrame 列名和格式）.

    :param method: 'standard' / 'minmax' / 'robust' / 'maxabs'
    :param features: 需要缩放的特征列表，None 时处理所有数值列
    """
```

#### B-4  更新 `core/feature_engineering/__init__.py`

---

### Theme C：补齐分箱模块实用功能（中优先）

#### C-1  `OptimalBinning.batch_to_excel()`

```python
def batch_to_excel(
    self,
    output_path: str,
    feature_map: Dict[str, str] = None,
) -> str:
    """批量分箱结果输出到 Excel（每特征一行，含分箱表和分箱图）.

    :param output_path: 输出 Excel 路径
    :param feature_map: 特征英文名→中文名映射
    :return: 输出路径
    """
```

#### C-2  新增 `BestPSIBinning`（`core/binning/best_psi_binning.py`）

```python
class BestPSIBinning(BaseBinning):
    """PSI 最优分箱：在约束条件下寻找使训练集/验证集 PSI 最小的切分点.

    适用场景：特征分布在训练集和测试集有偏移时，选择
    使两个数据集分布差异最小的分箱方案。

    :param max_n_bins: 最大分箱数，默认 5
    :param min_bin_size: 每箱最小样本占比，默认 0.05
    :param psi_target: PSI 约束（分箱方案总 PSI ≤ psi_target），默认 0.1

    Example:
        >>> binner = BestPSIBinning(max_n_bins=5)
        >>> binner.fit(X_train, y_train, X_val=X_val)
        >>> X_binned = binner.transform(X_test)
    """
```

---

### Theme D：强化规则模块（中优先）

当前规则模块的核心能力（Rule/RuleSet/RulesClassifier + 三类 RuleMiner）已具备，缺少的是**规则集运营工具**。

#### D-1  规则集覆盖率模拟（`report/rule_analysis.py` 扩充）

```python
def ruleset_coverage_simulation(
    df: pd.DataFrame,
    rules: List[str],    # pandas query 表达式列表
    target: str,
    labels: List[str] = None,
) -> pd.DataFrame:
    """多规则组合下的整体覆盖率/坏率改善仿真.

    计算各规则独立贡献和规则集叠加后的整体效果：
    - 各规则覆盖率、坏率、LIFT
    - 规则间覆盖重叠矩阵
    - 规则集合并后的最终效果（通过率/坏率/坏率改善）
    """

def rule_effectiveness_tracking(
    df_list: List[pd.DataFrame],
    labels: List[str],
    rule_expr: str,
    target: str,
) -> pd.DataFrame:
    """单条规则跨期有效性追踪（各期覆盖率、坏率、LIFT 趋势）."""
```

#### D-2  规则冲突检测（`core/rules/rule.py` 扩充）

```python
def detect_rule_conflicts(
    rules: List[Rule],
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """检测规则集中逻辑矛盾或高度重叠的规则对.

    输出规则对覆盖重叠率，标注高重叠（>80%）和逻辑矛盾。
    """
```

---

### Theme E：提升模型报告完整性（中优先）

当前 `auto_model_report` / `QuickModelReport` 已相当完整（6 个 Sheet），以下是高价值补充。

#### E-1  SHAP 报告集成

`QuickModelReport.to_excel(include_shap=True)` 时补充 Sheet：

```
Sheet 7 - SHAP 解释性（可选）
  - SHAP Summary Plot（蜂群图）
  - SHAP Bar Plot（平均|SHAP|特征重要性）
  - Top 5 特征 SHAP 依赖图
  - SHAP 均值汇总表（可导出）
```

需借助已有的 `ModelExplainer` 完成，`ModelExplainer.to_excel()` 方法一并实现。

#### E-2  `ScoreCard.score_segment_analysis()`

```python
def score_segment_analysis(
    self,
    df: pd.DataFrame,
    target: str,
    n_bins: int = 10,
    features: List[str] = None,
) -> pd.DataFrame:
    """各评分段的客群特征分布对比分析.

    输出各评分分箱下：样本量/坏率/主要特征均值，
    帮助理解模型在不同风险段的决策依据。
    """
```

#### E-3  `compare_models` Excel 报告完善

```python
def compare_models(
    models: Dict[str, Any],
    X_train, y_train,
    X_test=None, y_test=None,
    output_path: str = 'model_comparison.xlsx',
    metrics: List[str] = ['KS', 'AUC', 'LIFT@5%', 'PSI'],
) -> pd.DataFrame:
    """多模型性能对比报告（含雷达图/折线图/评分分布对比）."""
```

---

### Theme F：特征筛选器补强（低优先）

#### F-1  `LiftSelector` 增加 `ratio` 参数

```python
class LiftSelector(BaseFeatureSelector):
    def __init__(
        self,
        threshold: float = 1.5,
        ratio: float = 0.10,    # 新增：LIFT 计算的覆盖率（默认 10%）
        ascending: bool = False,
        ...
    ):
        """支持 lift@1%/5%/10% 等自定义比例."""
```

#### F-2  `CompositeFeatureSelector.to_excel()`

```python
def to_excel(
    self,
    output_path: str,
    feature_map: Dict[str, str] = None,
) -> str:
    """筛选全流程报告 Excel.

    每个筛选步骤一个 Sheet，输出：
    - 各步骤通过/剔除特征列表及剔除原因
    - 各特征完整筛选轨迹汇总表
    """
```

---

### Theme G：损失函数扩充（低优先）

#### G-1  新增 `OrdinalRankLoss`（`core/models/losses/`）

```python
class OrdinalRankLoss(BaseLoss):
    """序数损失：优化预测评分的排序一致性（最大化 AUC 代理目标）.

    适用于评分卡场景，直接优化样本对相对排序，
    比交叉熵损失更贴近"区分好坏"的业务目标。
    """
```

#### G-2  新增 `LiftFocusedLoss`（`core/models/losses/`）

```python
class LiftFocusedLoss(BaseLoss):
    """头部 LIFT 导向损失：对高风险样本施加更高惩罚.

    在高概率端不正确预测的惩罚远大于低概率端，
    引导模型在头部风险区间有更强区分能力。

    :param top_ratio: 头部比例，默认 0.10
    :param penalty_factor: 头部惩罚倍数，默认 3.0
    """
```

---

## 四、版本规划

| 版本 | 主要内容 | 涉及 Theme | 预期状态 |
|------|----------|-----------|---------|
| `v0.1.x` | 现有代码稳定版（Bug修复、测试补全） | — | ✅ 当前 |
| `v0.2.0` | EDA 策略域与客群域补齐 | Theme A | 待开发 |
| `v0.3.0` | 特征工程模块大幅扩充 | Theme B | 待开发 |
| `v0.4.0` | 分箱批量导出 + BestPSIBinning + 规则运营工具 | Theme C + D | 待开发 |
| `v0.5.0` | 模型报告 SHAP 集成 + compare_models 完善 + ScoreCard 分析扩充 | Theme E | 待开发 |
| `v0.6.0` | 筛选器补强 + 损失函数扩充 | Theme F + G | 待开发 |

---

## 五、实现顺序建议

```
Theme A（EDA 策略/客群）
  → 策略分析师最迫切需要，无其他模块依赖
  → 纯函数实现，风险低，预计 4-6 天

Theme B（特征工程）
  → 模型开发链路的基础短板，sklearn 标准接口
  → 独立模块，预计 3-5 天

Theme C（分箱扩展）+ Theme F（筛选器补强）
  → 在现有代码上追加，改动小
  → 可并行，合计 2-3 天

Theme D（规则运营）
  → 依赖现有 report/mining，在其基础上追加函数
  → 预计 2-3 天

Theme E（报告完善）
  → 依赖 Theme B（特征工程完善后报告更完整）
  → 依赖已有 ModelExplainer，SHAP 集成相对独立
  → 预计 3-4 天

Theme G（损失函数）
  → 独立数学组件，随时可实现
  → 预计 1-2 天
```

---

## 六、代码规范（继承现有风格）

1. **命名规范**：类名 PascalCase，函数/方法 snake_case，常量 UPPER_SNAKE_CASE
2. **Docstring**：Google 风格，每个 public API 含最小可运行示例
3. **输入处理**：所有 public API 入参走 `hscredit.utils.input_utils` 统一处理
4. **输出格式**：统计表列名使用中文，与现有模块保持一致
5. **Optional 依赖**：缺失时给出清晰安装提示，如 `pip install shap`
6. **只增不改**：不破坏现有 API，参数别名保留向后兼容
7. **Excel 报告**：统一使用 `ExcelWriter`，样式参考 `feature_analyzer.py`
8. **sklearn 兼容**：所有 Transformer 类实现 `fit` / `transform` / `get_params` / `set_params`，通过 `check_estimator` 验证
9. **类型注解**：新增代码全部加 Python 类型注解
10. **测试覆盖**：每个新模块对应 `tests/` 下的测试文件，核心路径覆盖率 ≥ 80%
