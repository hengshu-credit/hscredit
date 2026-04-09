# hscredit 重新设计规划方案

> 版本：2026-03-27（修订版 v2）  
> 定位：面向金融信贷场景的完整风控建模工具包，覆盖策略分析人员与模型开发人员的全链路需求。  
> 修订说明：根据用户反馈，对原规划方案进行全面修订，聚焦7大核心修正点。  
> 最近更新：完成 Bug 修复、参数名统一、部分 Phase 1-5 功能实现。

---

## 已完成事项 ✅

### Bug 修复
- ✅ `BaseFeatureSelector._get_feature_names()` 补充 `feature_names_in_` 属性，修复 sklearn Pipeline 兼容性
- ✅ `NumExprDerive._transform_frame()` 分离数值/非数值列处理，修复 datetime/object 列 TypeError
- ✅ `ExcelWriter.__init__` 优雅处理模板文件缺失，fallback 到空 Workbook
- ✅ `examples/12_complete_workflow.ipynb` 修复 JSON 转义错误

### 参数名统一
- ✅ `eda/target.py` 5个函数添加 `target` 别名（兼容 `target_col`）
- ✅ `eda/target.py` `bad_rate_by_dimension` 添加 `segment_col` 别名（兼容 `dim_col`）
- ✅ `viz/binning_plots.py` `corr_plot` 添加 `figsize` 别名（兼容 `figure_size`）
- ✅ `viz/binning_plots.py` `ks_plot` 添加 `ax` 别名（兼容 `axes`）

### 功能新增
- ✅ **#6 多标签规则挖掘**：`report/mining/multi_label.py` `MultiLabelRuleMiner` 类 + `report/rule_analysis_report.py` `multi_label_rule_report` 函数
- ✅ **#7 调参目标扩展**：`models/tuning.py` 新增 `ks_lift_combined` 和 `tail_purity_ks` 内置目标
- ✅ **#9 评分卡部署代码导出**：`models/scorecard.py` 新增 `export_deployment_code()` 支持 SQL/Python/Java
- ✅ **#11 viz 统一样式系统**：`viz/style.py` 提供 `set_style(theme)` 主题管理、配色方案、中文字体自动检测
- ✅ **#12 客群偏移监控报告**：`report/population_drift_report.py` `population_drift_report()` 生成 PSI 总览 + 特征分布对比 + 逾期率对比 + 评分分布 Excel 报告
- ✅ **#15 StabilityAwareSelector**：`selectors/stability_selector.py` 综合 IV 有效性与 PSI 稳定性的加权评分特征筛选器

---

## 修订说明

原规划方案整体方向正确，本次修订针对以下7点进行调整：

1. **Sphinx API 文档暂缓**：库尚未成型，不生成文档站点，节省精力专注功能实现
2. **已有功能继续扩充**：分箱/特征筛选/模型/自定义loss 仍有完善空间，持续迭代
3. **画图模块重新设计**：API 统一重设计，方法扩充，聚焦「模型报告变量分析/评分分析」和「策略人员跨时间/客群/交叉特征有效性和偏移分析」
4. **EDA 体系化重构**：当前方法多但不成体系，金融策略/模型相关分析不够，客群监控和偏移分析专项补强
5. **规则挖掘整合进 Report**：支持多标签联合规则挖掘（如 MOB3@30 + MOB6@30），形成可读的规则有效性分析报告
6. **模型报告完整设计**：参考 `examples/模型报告.xlsx` 和 `examples/建模参考代码.ipynb`，设计更完整的模板和内容
7. **指标和调参扩充**：LIFT@1%/3%/5%/10%/任意值、头/尾部区分能力目标、LIFT单调性约束调参

---

## 一、现有功能全景（只增不改原则）

### 1.1 模块现状速览

```
hscredit/
├── core/
│   ├── binning/          # 17种分箱算法 + OptimalBinning  ✅ 持续扩充
│   ├── encoders/         # 8种编码器  ✅
│   ├── selectors/        # 22种特征筛选 + 组合器  ✅ 新增 StabilityAwareSelector
│   ├── models/           # 7种模型+评分卡+损失函数+调参+解释  ✅ 调参目标已扩充，评分卡部署代码已实现
│   ├── metrics/          # 分类/特征/稳定性/金融/回归指标  ⚠️ LIFT待扩充
│   ├── viz/              # 40+图表 + 统一样式系统  ✅ style.py 主题管理已实现
│   ├── eda/              # 多个EDA函数  ⚠️ 待体系化补强（population.py / strategy.py）
│   ├── rules/            # 规则引擎+挖掘  ✅ 多标签规则挖掘已整合进Report
│   ├── feature_engineering/  # NumExprDerive  ⚠️ 待扩充（已修复非数值列bug）
│   └── financial/        # 金融计算  ✅
├── report/               # 特征/规则/置换/偏移监控报告  ✅ 新增客群偏移报告、多标签规则报告
└── utils/                # 工具函数  ✅
```

---

## 二、差距分析与优先级

### 2.1 高优先级差距（本次规划重点）

| 模块 | 现状 | 差距 | 优先级 |
|------|------|------|--------|
| `metrics/finance.py` | `lift_curve` 仅固定比例 | **缺 LIFT@1%/3%/5%/10%/任意值；缺LIFT单调性检验** | P0 |
| `models/tuning.py` | ✅ 已扩充 `ks_lift_combined` / `tail_purity_ks` | ~~缺头/尾部区分能力目标~~ 已完成 | ✅ Done |
| `viz/` | ✅ 40+图表 + `style.py` 统一样式 | ~~缺统一绘图入口~~ 已完成样式系统 | ✅ Done (样式) |
| `eda/` | 方法多但不成体系 | **缺金融策略分析体系、客群监控、特征偏移专项分析** | P0 |
| `report/model_report.py` | 不存在 | **缺完整模型报告（参考examples/模型报告.xlsx）** | P0 |
| `rules/` + `report/` | ✅ 多标签挖掘已整合进Report | ~~缺多标签联合挖掘~~ 已完成 | ✅ Done |
| `feature_engineering/` | NumExprDerive（已修复bug） | 时序特征/交叉特征/预处理Transformer | P2 |

### 2.2 中低优先级（持续迭代）

| 模块 | 内容 | 优先级 |
|------|------|--------|
| `binning/` | 分箱结果批量Excel输出完善 | P2 |
| `selectors/` | ✅ StabilityAwareSelector 已实现；筛选全流程Pipeline示例完善 | P2 |
| `models/scorecard.py` | ✅ SQL/Python/Java 部署代码生成已实现 | ✅ Done |
| `models/interpretability.py` | SHAP 结果落 Excel | P2 |
| `models/tuning.py` | 调参报告 Excel 输出 | P2 |
| `models/losses/` | 损失函数进一步完善 | P2 |
| `report/` | ✅ 客群偏移监控报告已实现；策略对比报告待补充 | P2 |

---

## 三、实现步骤（按优先级分阶段）

---

### Phase 1 —— 指标扩充：LIFT@任意比例 + 单调性检验（1-2天）

**目标**：`lift_at` 支持任意比例 LIFT，`lift_monotonicity_check` 支持头/尾部单调性检验，为调参目标和模型报告提供基础指标。

#### 1.1 新增 `lift_at` 函数（`core/metrics/finance.py`）

```python
def lift_at(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    ratios: Union[float, List[float]] = [0.01, 0.03, 0.05, 0.10],
    ascending: bool = False,
) -> Union[float, pd.DataFrame]:
    """计算指定覆盖率下的LIFT值.

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率
    :param ratios: 覆盖率，如 0.05 或 [0.01, 0.03, 0.05, 0.10]
    :param ascending: False=高概率排前（风险模型头部），True=低概率排前（尾部分析）
    :return: 单个 float（ratios 为标量时）或 DataFrame（ratios 为列表时）

    Example:
        >>> lift_at(y_true, y_prob, ratios=0.05)          # 单值
        3.42
        >>> lift_at(y_true, y_prob, ratios=[0.01, 0.03, 0.05, 0.10])
           覆盖率  样本数  坏样本数  坏样本捕获率  LIFT值
        0    1%      50      35       12.5%    5.83
        1    3%     150      98       35.0%    4.83
        2    5%     250     155       55.4%    4.11
        3   10%     500     285       101.8%   3.76
    """
```

扩充现有 `lift_curve`：
- 默认 `percentages=[0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]`
- 新增 `tail` 参数（`tail=True` 时从低概率端截取，分析尾部低风险客群）

#### 1.2 新增 `lift_monotonicity_check` 函数（`core/metrics/finance.py`）

```python
def lift_monotonicity_check(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    direction: str = 'both',   # 'head' / 'tail' / 'both'
) -> Dict[str, Any]:
    """检查LIFT单调性.

    风控场景理想状态：高风险端（头部）坏率单调递减，低风险端（尾部）坏率单调递增。
    违反单调性意味着评分中间段区分能力弱，可作为调参约束。

    :return: {
        'head_monotonic': bool,
        'tail_monotonic': bool,
        'head_lift_values': list,       # 各分箱LIFT（由高风险到低风险）
        'tail_lift_values': list,
        'head_violations': list,        # 违反单调性的分箱 [(bin_i, bin_j, 差值)]
        'tail_violations': list,
        'head_violation_ratio': float,  # 违反比例 0.0~1.0
        'tail_violation_ratio': float,
    }

    Example:
        >>> result = lift_monotonicity_check(y_true, y_prob, n_bins=10)
        >>> print(result['head_monotonic'])        # True/False
        >>> print(result['head_violation_ratio'])  # 0.0
    """
```

#### 1.3 扩充 `BaseRiskModel.evaluate()` 返回字典（`core/models/base.py`）

```python
# 原有
{'KS': ..., 'AUC': ..., 'Gini': ...}

# 新增
{
    'KS': ..., 'AUC': ..., 'Gini': ...,
    'LIFT@1%': ..., 'LIFT@3%': ..., 'LIFT@5%': ..., 'LIFT@10%': ...,
    'LIFT_HEAD_MONOTONIC': True/False,
}
```

导出路径：更新 `core/metrics/__init__.py` 和 `hscredit/__init__.py`。

---

### Phase 2 —— 调参扩充：头尾区分能力 + LIFT单调性约束（2-3天）

**目标**：`ModelTuner` 支持以头部/尾部区分能力或 LIFT 单调性为优化目标，适配「重点管控头部高风险客群」的风控建模需求。

#### 2.1 新增内置调参目标类（`core/models/tuning.py`）

```python
class TuningObjective:
    """内置调参目标函数集合.

    所有目标函数签名：(y_true, y_prob) -> float（值越大越好）
    """

    @staticmethod
    def ks(y_true, y_prob) -> float:
        """标准KS目标."""

    @staticmethod
    def auc(y_true, y_prob) -> float:
        """AUC目标."""

    @staticmethod
    def lift_head(y_true, y_prob, ratio: float = 0.10) -> float:
        """头部LIFT目标：优化预测概率最高 ratio 比例样本的 LIFT 值."""

    @staticmethod
    def lift_tail(y_true, y_prob, ratio: float = 0.10) -> float:
        """尾部LIFT目标：优化预测概率最低 ratio 比例样本（低风险客群）的纯净度."""

    @staticmethod
    def lift_head_monotonic(
        y_true, y_prob,
        n_bins: int = 10,
        penalty: float = 0.5,
    ) -> float:
        """头部单调LIFT目标：KS × (1 - 违反单调性比例 × penalty).

        单调性违反比例越低，目标越高；完全单调时等同于 KS 目标。
        """

    @staticmethod
    def ks_with_lift_constraint(
        y_true, y_prob,
        min_lift_ratio: float = 0.05,
        min_lift_value: float = 2.0,
    ) -> float:
        """KS + LIFT约束：满足头部 min_lift_ratio 处 LIFT >= min_lift_value 前提下最大化KS."""

    @staticmethod
    def head_ks(
        y_true, y_prob,
        ratio: float = 0.30,
    ) -> float:
        """头部KS：仅计算预测概率前 ratio 比例样本的KS（头部区分能力）."""
```

#### 2.2 `ModelTuner` 扩充参数

```python
class ModelTuner:
    def __init__(
        self,
        model_class,
        param_space: dict = None,
        objective: Union[str, Callable] = 'ks',
        # 支持字符串：'ks' / 'auc' / 'lift_head' / 'lift_tail' /
        #              'lift_head_monotonic' / 'ks_with_lift_constraint' / 'head_ks'
        # 或自定义函数 (y_true, y_prob) -> float
        objective_kwargs: dict = None,  # 透传给目标函数的额外参数，如 ratio=0.05
        n_trials: int = 100,
        direction: str = 'maximize',
        cv: int = 0,                  # >0 启用交叉验证
        eval_ratios: List[float] = [0.01, 0.03, 0.05, 0.10],  # 调参过程中额外追踪的 LIFT 比例
        ...
    )
```

调参过程中，每个 trial 的 history 除记录主目标值外，还记录 `LIFT@1%/3%/5%/10%`，方便事后分析「用哪个目标调参更能满足业务需要」。

---

### Phase 3 —— 画图模块重新设计（3-5天）

**目标**：统一 API 风格，扩充模型报告变量分析/评分分析图，以及策略人员需要的跨时间/客群/交叉特征有效性和偏移分析图。

#### 3.1 设计原则

- **统一函数签名**：所有绘图函数遵循 `func(data, ..., ax=None, figsize=(...), title=..., save=None) -> Figure`
- **统一样式系统**：通过 `viz/style.py` 管理调色板/字体/网格，提供 `set_style(theme='risk')` 入口
- **支持批量输出**：所有函数支持 `save` 参数，图片保存后供 Excel 报告插入
- **中英文支持**：标题和轴标签默认中文，可通过 `lang='en'` 切换

#### 3.2 新增：模型报告变量分析图（`core/viz/variable_plots.py`）

```python
def variable_iv_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    top_n: int = 20,
    ax=None, figsize=(12, 8), title='特征IV值排名', save=None,
) -> Figure:
    """特征IV值横向柱状图，按IV降序排列，标注IV阈值参考线（0.02/0.1/0.3）."""

def variable_woe_trend_plot(
    bin_table: pd.DataFrame,
    feature: str = None,
    ax=None, figsize=(10, 5), title=None, save=None,
) -> Figure:
    """WOE折线图+坏率柱状图（双轴），用于变量分析报告."""

def variable_psi_heatmap(
    psi_matrix: pd.DataFrame,
    ax=None, figsize=(14, 8), title='特征PSI热力图', save=None,
) -> Figure:
    """特征PSI矩阵热力图，颜色反映偏移程度（绿/黄/红）."""

def variable_importance_grouped_plot(
    importance_df: pd.DataFrame,
    group_col: str = 'category',
    value_col: str = 'importance',
    ax=None, figsize=(12, 8), title='分类特征重要性', save=None,
) -> Figure:
    """按特征类别（资产/负债/行为/人口学等）分组的重要性堆叠图."""

def variable_missing_badrate_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    ax=None, figsize=(12, 6), title='缺失率 vs 坏账率', save=None,
) -> Figure:
    """散点图：横轴=缺失率，纵轴=缺失样本坏账率（用于评估缺失值信息价值）."""
```

#### 3.3 新增：评分分析图（`core/viz/score_plots.py`）

```python
def score_ks_plot(
    y_true, y_prob,
    datasets: Dict[str, Tuple] = None,  # {'训练集': (y_true, y_prob), '测试集': ...}
    ax=None, figsize=(10, 6), title='KS曲线', save=None,
) -> Figure:
    """KS曲线图，支持多数据集叠加（训练集/测试集/OOT）."""

def score_distribution_comparison_plot(
    scores: Dict[str, np.ndarray],  # {'训练集': array, '测试集': array}
    ax=None, figsize=(10, 5), title='评分分布对比', save=None,
) -> Figure:
    """多数据集评分分布对比（KDE + 直方图），直观反映分布偏移."""

def score_badrate_bin_plot(
    y_true, score,
    n_bins: int = 10,
    show_psi: bool = True,
    ax=None, figsize=(12, 6), title='评分分箱坏率图', save=None,
) -> Figure:
    """评分分箱图：柱=样本量，折线=坏率，支持同时展示PSI."""

def score_lift_plot(
    y_true, y_prob,
    ratios: List[float] = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50],
    datasets: Dict[str, Tuple] = None,
    ax=None, figsize=(10, 6), title='LIFT曲线', save=None,
) -> Figure:
    """LIFT曲线图，支持多数据集叠加，标注 1%/5%/10% 参考点."""

def score_approval_badrate_curve(
    y_true, score,
    ax=None, figsize=(10, 6), title='通过率-坏率权衡曲线', save=None,
) -> Figure:
    """审批通过率 vs 坏账率曲线（策略人员必备），同时展示不同阈值下的通过数量."""
```

#### 3.4 新增：策略分析图 - 跨时间/客群/交叉（`core/viz/strategy_plots.py`）

```python
def feature_trend_by_time(
    df: pd.DataFrame,
    feature: str,
    date_col: str,
    target: str = None,
    stat: str = 'mean',   # 'mean' / 'median' / 'psi' / 'badrate'
    ax=None, figsize=(12, 5), title=None, save=None,
) -> Figure:
    """特征随时间的均值/中位数/PSI/坏率趋势图，用于检测特征偏移."""

def feature_drift_comparison(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str],
    top_n: int = 20,
    ax=None, figsize=(12, 8), title='特征分布偏移（PSI）', save=None,
) -> Figure:
    """多特征偏移瀑布图：横轴=PSI值，颜色标注偏移等级（绿/黄/红），快速定位偏移特征."""

def feature_effectiveness_by_segment(
    df: pd.DataFrame,
    feature: str,
    target: str,
    segment_col: str,
    metric: str = 'iv',   # 'iv' / 'ks' / 'auc' / 'lift@5%'
    ax=None, figsize=(10, 6), title=None, save=None,
) -> Figure:
    """特征在不同客群（渠道/产品/时间段）下的有效性对比热力图或分组柱状图."""

def feature_cross_heatmap(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    target: str,
    stat: str = 'badrate',   # 'badrate' / 'count' / 'lift'
    ax=None, figsize=(10, 8), title=None, save=None,
) -> Figure:
    """两个特征交叉分析热力图：行=feature_x分箱，列=feature_y分箱，格=坏率/样本数/LIFT."""

def population_drift_monitor(
    df_list: List[pd.DataFrame],
    labels: List[str],
    features: List[str],
    target: str = None,
    ax=None, figsize=(14, 10), title='客群偏移监控', save=None,
) -> Figure:
    """多期客群特征分布偏移监控大图：上方为PSI矩阵，下方为关键特征分布叠加对比图."""

def segment_scorecard_comparison(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    segment_col: str,
    ax=None, figsize=(14, 6), title='分客群评分效果对比', save=None,
) -> Figure:
    """按客群分组的评分KS/AUC/LIFT对比柱状图，用于验证模型在不同客群的稳定性."""
```

#### 3.5 更新 `viz/__init__.py` 统一导出

将 `variable_plots`、`score_plots`、`strategy_plots` 中所有新增函数统一导出至 `core/viz/__init__.py` 和顶层 `hscredit/__init__.py`。

---

### Phase 4 —— EDA 体系化重构（3-5天）

**目标**：将现有 EDA 函数重新组织为「金融策略体系」，新增客群监控、特征偏移、金融指标分析三大专项模块，形成有章可循的分析框架。

#### 4.1 EDA 分析体系设计

现有 EDA 函数按金融风控场景重组为以下5个分析域：

```
core/eda/
├── overview.py       # 域1：数据质量与概览  ✅ 已有，持续完善
├── target.py         # 域2：目标变量分析  ✅ 已有，持续完善
├── feature.py        # 域3：特征分析  ✅ 已有，持续完善
├── relationship.py   # 域4：特征-标签关系  ✅ 已有，持续完善
├── correlation.py    # 域5：相关性分析  ✅ 已有
├── stability.py      # 域6：稳定性分析  ✅ 已有，待扩充
├── vintage.py        # 域7：Vintage分析  ✅ 已有
├── population.py     # 域8：客群监控与偏移分析  🆕 新增
├── strategy.py       # 域9：金融策略分析  🆕 新增
└── report.py         # 综合报告  ✅ 已有，待扩充
```

#### 4.2 新增：客群监控与偏移分析（`core/eda/population.py`）🆕

```python
def population_profile(
    df: pd.DataFrame,
    features: List[str],
    segment_col: str = None,
    date_col: str = None,
    target: str = None,
) -> pd.DataFrame:
    """客群画像分析：各特征均值/分位数/坏率，支持按客群/时间分组."""

def population_shift_analysis(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str],
    target: str = None,
    psi_threshold: float = 0.1,
) -> pd.DataFrame:
    """客群偏移分析：计算各特征PSI/KS/均值变化，标注偏移等级，输出偏移摘要表.

    :return: DataFrame，含特征名/PSI/均值变化/偏移等级/建议
    """

def population_monitoring_report(
    df_base: pd.DataFrame,
    df_compare_list: List[pd.DataFrame],
    compare_labels: List[str],
    features: List[str],
    target: str = None,
    output_path: str = 'population_monitor.xlsx',
) -> str:
    """多期客群监控报告（Excel输出）.

    报告包含：
    - 各期客群规模和坏率趋势
    - 特征PSI时序矩阵（热力图）
    - 偏移特征Top10详细分布对比
    - 目标变量趋势（若传入）
    """

def segment_drift_analysis(
    df: pd.DataFrame,
    date_col: str,
    segment_col: str,
    features: List[str],
    target: str = None,
    base_period: str = None,
) -> pd.DataFrame:
    """分客群、分时间的特征偏移分析.

    每个 (segment, period) 组合计算相对于基准期的特征PSI，
    输出三维矩阵：特征 × 时间 × 客群。
    """

def feature_cross_segment_effectiveness(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    segment_col: str,
    metric: str = 'iv',   # 'iv' / 'ks' / 'auc' / 'lift@5%'
) -> pd.DataFrame:
    """各特征在不同客群下的有效性矩阵（行=特征，列=客群，格=IV/KS/AUC）."""
```

#### 4.3 新增：金融策略分析（`core/eda/strategy.py`）🆕

```python
def approval_badrate_tradeoff(
    y_true: Union[np.ndarray, pd.Series],
    score: Union[np.ndarray, pd.Series],
    n_points: int = 100,
) -> pd.DataFrame:
    """通过率-坏率权衡曲线数据.

    :return: DataFrame，含阈值/通过率/拒绝率/通过样本坏率/拒绝样本坏率/整体坏率改善
    """

def score_strategy_simulation(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    thresholds: List[float],
    amount_col: str = None,
) -> pd.DataFrame:
    """评分策略模拟：给定多个阈值，计算各策略下的通过率/坏率/损失额/利润."""

def vintage_performance_summary(
    df: pd.DataFrame,
    vintage_col: str,
    mob_col: str,
    target_col: str,
    mob_points: List[int] = [3, 6, 9, 12],
) -> pd.DataFrame:
    """Vintage表现摘要：各放款批次在指定 MOB 节点的坏率，输出矩阵表."""

def roll_rate_matrix(
    df: pd.DataFrame,
    dpd_t0: str,
    dpd_t1: str,
    bins: List[int] = [0, 1, 7, 15, 30, 60, 90, 120],
) -> pd.DataFrame:
    """滚动率矩阵：从 DPD_t0 到 DPD_t1 的迁移率矩阵，用于判断资产质量变化."""

def label_leakage_check(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    threshold_iv: float = 0.5,
    threshold_auc: float = 0.9,
) -> pd.DataFrame:
    """标签泄露检查：IV/AUC 异常高的特征预警，输出疑似泄露特征列表."""

def multi_label_correlation(
    df: pd.DataFrame,
    labels: List[str],
) -> pd.DataFrame:
    """多标签相关性分析（如 FPD15/FPD30/MOB3@30/MOB6@30 相互关系），
    输出标签间 Spearman 相关矩阵和一致性率表."""
```

#### 4.4 完善 `stability.py`：增加偏移专项分析

```python
def feature_drift_report(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str] = None,
    method: str = 'psi',   # 'psi' / 'ks' / 'wasserstein'
    psi_bins: int = 10,
) -> pd.DataFrame:
    """批量特征偏移报告：计算所有特征的偏移指标，按偏移程度排序，标注等级."""

def score_drift_report(
    score_base: pd.Series,
    score_target: pd.Series,
    y_base: pd.Series = None,
    y_target: pd.Series = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """评分偏移综合报告：PSI + 分布对比 + 坏率变化（若传入标签）."""
```

#### 4.5 更新 `eda/__init__.py`

将所有新增函数统一导出，按域分组注释，方便用户查找。

---

### Phase 5 —— 规则挖掘整合进 Report：多标签规则分析（3-5天）

**目标**：支持设置多个标签（如 MOB3@30 和 MOB6@30），挖掘规则时同时考虑多个标签的有效性，生成可解读的规则分析报告。

#### 5.1 多标签规则挖掘（`core/rules/mining/multi_label.py`）🆕

```python
class MultiLabelRuleMiner:
    """多标签规则挖掘器.

    支持同时针对多个标签（如短期标签 MOB3@30 和长期标签 MOB6@30）挖掘规则，
    并分析规则在不同标签下的有效性差异。

    典型应用场景：
    - 长短期标签都有效的强规则（稳定拒绝规则）
    - 仅短期标签有效（可能是偶发风险，谨慎使用）
    - 仅长期标签有效（长期风险，可做预警规则）
    - 两标签均无效（噪声规则，丢弃）

    Example:
        >>> miner = MultiLabelRuleMiner(
        ...     labels=['mob3_30', 'mob6_30'],
        ...     label_names=['短期标签(MOB3@30)', '长期标签(MOB6@30)'],
        ...     min_support=0.02,
        ...     min_lift=1.5,
        ... )
        >>> miner.fit(df, features=['age', 'income', 'credit_score'])
        >>> rules = miner.get_rules(effectiveness='both')  # 'both'/'short_only'/'long_only'/'any'
        >>> report = miner.get_report()
    """

    def fit(self, df, features=None): ...

    def get_rules(
        self,
        effectiveness: str = 'any',
        # 'both'=所有标签均有效, 'short_only'=仅短期有效,
        # 'long_only'=仅长期有效, 'any'=任一标签有效
        min_lift_per_label: float = 1.5,
        min_support: float = 0.01,
        top_n: int = None,
    ) -> pd.DataFrame:
        """获取筛选后的规则表.

        :return: DataFrame，含规则表达式/各标签支持度/各标签LIFT/规则类型/建议
        """

    def get_effectiveness_matrix(self) -> pd.DataFrame:
        """规则有效性矩阵：行=规则，列=各标签，格=LIFT值，高亮显示有效规则."""

    def get_report(self) -> pd.DataFrame:
        """完整规则分析报告，含规则分类和业务解读."""
```

#### 5.2 规则分析报告（`report/rule_analysis_report.py`）🆕

```python
def multi_label_rule_report(
    df: pd.DataFrame,
    features: List[str],
    labels: Dict[str, str],
    # 如 {'短期标签': 'mob3_30', '长期标签': 'mob6_30'}
    miner_params: dict = None,
    output_path: str = 'rule_analysis_report.xlsx',
) -> str:
    """多标签规则挖掘分析报告（Excel输出）.

    报告包含：
    Sheet 1 - 规则汇总
        - 各规则在每个标签下的覆盖率/坏率/LIFT/支持度
        - 规则有效性分类（长短期均有效/仅短期/仅长期/均无效）
        - 规则建议（强拒绝/预警/观察/放弃）
    Sheet 2 - 规则有效性矩阵热力图
        - 行=规则，列=标签，颜色=LIFT强度
    Sheet 3 - 单规则详细分析
        - 每条规则在各标签下的分箱分布和坏率
    Sheet 4 - 规则集模拟
        - 多条规则组合后的整体覆盖率/坏率改善
    Sheet 5 - 单特征规则概览
        - 各特征的最优切分点及双标签效果对比

    Example:
        >>> multi_label_rule_report(
        ...     df=df,
        ...     features=['age', 'income', 'credit_score'],
        ...     labels={'短期标签(MOB3@30)': 'mob3_30', '长期标签(MOB6@30)': 'mob6_30'},
        ...     output_path='rule_report.xlsx',
        ... )
    """
```

更新 `report/__init__.py` 导出 `multi_label_rule_report`。

---

### Phase 6 —— 模型报告完整实现（5-7天）

**目标**：参考 `examples/模型报告.xlsx` 和 `examples/建模参考代码.ipynb`，实现 `auto_model_report`，产出格式规范、内容详尽的模型评估 Excel 报告。

#### 6.1 报告结构设计（参考示例文件）

```
auto_model_report 输出的 Excel 结构：
├── Sheet 1：封面
│   - 模型名称、版本、生成时间
│   - 数据概况（训练/测试/OOT 样本量、坏率）
│   - 报告目录索引
│
├── Sheet 2：模型性能总览
│   - KS / AUC / Gini / LIFT@1%/3%/5%/10% 三数据集对比表
│   - 头部单调性标注（✓/✗）
│   - PSI 稳定性评级
│
├── Sheet 3：KS / ROC 曲线
│   - 训练集 + 测试集 + OOT KS曲线叠加图
│   - 训练集 + 测试集 + OOT ROC曲线叠加图
│
├── Sheet 4：LIFT 曲线
│   - 多数据集 LIFT 曲线叠加
│   - LIFT@1%/3%/5%/10% 标注点
│
├── Sheet 5：评分分布
│   - 训练集 / 测试集 / OOT 评分分布（KDE + 直方图）
│   - 好坏样本评分分布叠加（检验区分度）
│
├── Sheet 6：评分分箱坏率
│   - 训练集 / 测试集 / OOT 分箱坏率表（支持率/坏率/LIFT/累积KS）
│   - 分箱坏率图（柱+折线双轴）
│
├── Sheet 7：PSI 稳定性
│   - 训练集 vs 测试集 评分 PSI 分箱表
│   - 训练集 vs OOT 评分 PSI 分箱表（若有OOT）
│
├── Sheet 8：变量分析（特征有效性）
│   - 入模特征 IV 排名表（含 WOE 单调性）
│   - 特征重要性 Top20 图
│   - 每个入模特征的分箱表（WOE/坏率/样本量）
│
├── Sheet 9：变量稳定性
│   - 入模特征 PSI 汇总表（训练 vs 测试，含偏移等级）
│   - 偏移较大特征的分布对比图
│
└── Sheet 10：SHAP 解释性（可选，需 shap 库）
    - SHAP Summary Plot
    - SHAP Bar Plot
    - Top 5 特征 SHAP 依赖图
```

#### 6.2 `auto_model_report` 函数签名（`report/model_report.py`）

```python
def auto_model_report(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    X_oot: pd.DataFrame = None,
    y_oot: pd.Series = None,
    feature_map: Dict[str, str] = None,    # 特征英文名 -> 中文名映射
    feature_groups: Dict[str, List[str]] = None,  # 特征分组，如 {'行为类': ['f1','f2']}
    include_shap: bool = False,
    output_path: str = 'model_report.xlsx',
    model_name: str = None,
    model_version: str = 'v1.0',
) -> str:
    """一键生成模型评估 Excel 报告.

    :param model: 训练好的 BaseRiskModel 实例
    :param X_train/y_train: 训练集特征和标签
    :param X_test/y_test: 测试集（可选）
    :param X_oot/y_oot: OOT集（可选，跨时间验证集）
    :param feature_map: 特征中文名映射，用于报告展示
    :param feature_groups: 特征分组，用于分类展示重要性
    :param include_shap: 是否包含 SHAP 解释性 Sheet（需安装 shap）
    :param output_path: 输出路径
    :param model_name: 模型名称（用于封面）
    :param model_version: 模型版本（用于封面）
    :return: 输出文件路径

    Example:
        >>> from hscredit.report import auto_model_report
        >>> auto_model_report(
        ...     model=lgbm_model,
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test,   y_test=y_test,
        ...     X_oot=X_oot,     y_oot=y_oot,
        ...     feature_map={'age': '年龄', 'income': '收入'},
        ...     output_path='lgbm_report.xlsx',
        ...     model_name='贷后风险模型',
        ... )
    """
```

#### 6.3 `ModelReport` 类扩充（`core/models/report.py`）

在现有 `ModelReport` 基础上新增：

```python
class ModelReport:
    # 现有方法保持不变，新增：

    def get_lift_table_at(
        self,
        ratios: List[float] = [0.01, 0.03, 0.05, 0.10],
        dataset: str = 'test',
    ) -> pd.DataFrame:
        """获取指定比例的LIFT汇总表."""

    def get_feature_psi(
        self,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """获取入模特征在训练集 vs 测试集的 PSI 汇总表."""

    def get_woe_bin_tables(self) -> Dict[str, pd.DataFrame]:
        """获取各入模特征的分箱统计表（WOE/IV/坏率）."""

    def to_excel(
        self,
        output_path: str,
        include_oot: bool = True,
        include_shap: bool = False,
        writer: 'ExcelWriter' = None,
    ) -> str:
        """输出完整模型报告 Excel."""
```

---

### Phase 7 —— 特征工程扩充（持续迭代，P2）

**目标**：补强 `core/feature_engineering/`，支持时序特征、交叉特征和标准化预处理 Transformer。

#### 7.1 时序特征（`core/feature_engineering/time_features.py`）

```python
class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """时序特征生成器.

    Example:
        >>> gen = TimeFeatureGenerator(
        ...     date_col='apply_date',
        ...     features=['year', 'month', 'weekday', 'is_weekend',
        ...               'days_since_epoch', 'quarter', 'is_month_end']
        ... )
        >>> X_new = gen.fit_transform(X)
    """
```

#### 7.2 交叉特征（`core/feature_engineering/cross_features.py`）

```python
class CrossFeatureGenerator(BaseEstimator, TransformerMixin):
    """交叉特征生成器（加/减/乘/除/比率/对数比）.

    Example:
        >>> gen = CrossFeatureGenerator(
        ...     pairs=[('income', 'debt'), ('age', 'credit_limit')],
        ...     operations=['ratio', 'diff', 'product', 'log_ratio']
        ... )
        >>> X_new = gen.fit_transform(X)
    """
```

#### 7.3 预处理 Transformer（`core/feature_engineering/preprocessing.py`）

```python
class MissingValueImputer(BaseEstimator, TransformerMixin):
    """缺失值填充器（均值/中位数/众数/常数）."""

class OutlierClipper(BaseEstimator, TransformerMixin):
    """异常值截断（分位数/固定边界）."""

class FeatureScaler(BaseEstimator, TransformerMixin):
    """标准化/归一化（封装sklearn，保留列名和DataFrame格式）."""
```

---

### Phase 8 —— 已有功能持续扩充（持续迭代，P2）

#### 8.1 分箱模块扩充

- `OptimalBinning.batch_to_excel()`：批量分箱结果输出到 Excel
- `OptimalBinning` 增加 `auto_select_bins` 模式：基于样本量自动确定最优分箱数
- 增加 `BestPSIBinning`：最优 PSI 分箱（使训练/测试分布差异最小）

#### 8.2 特征筛选扩充

- `LiftSelector` 增加 `ratio` 参数（支持 LIFT@1%/5% 等自定义比例）
- `CompositeFeatureSelector` 增加 `to_excel()` 输出筛选报告
- ✅ ~~新增 `StabilityAwareSelector`~~ 已实现：`selectors/stability_selector.py`

#### 8.3 损失函数扩充

- `FocalLoss` 增加 `class_weights` 参数（支持非对称类别权重）
- 新增 `OrdinalRankLoss`：序数损失，优化评分的排序一致性
- 新增 `LiftFocusedLoss`：头部 LIFT 导向损失（对高风险样本施加更高惩罚）

#### 8.4 评分卡扩充

- ✅ ~~`ScoreCard.export_deployment_code(language='sql'/'python'/'java')`~~ 已实现：`models/scorecard.py`
- `ScoreCard.score_segment_analysis(df, target)`：各评分段的样本特征分析

---

## 四、新增 API 总览

### 4.1 `core/metrics/finance.py` 新增

| 函数 | 说明 |
|------|------|
| `lift_at(y_true, y_prob, ratios)` | LIFT@任意比例，支持标量或列表 |
| `lift_monotonicity_check(y_true, y_prob)` | 头/尾部 LIFT 单调性检验 |

### 4.2 `core/models/tuning.py` 新增

| 类/函数 | 说明 |
|---------|------|
| `TuningObjective` | 内置目标函数集合（ks/auc/lift_head/lift_tail/lift_head_monotonic/head_ks）|
| `ModelTuner.objective` 参数扩充 | 支持字符串名称或自定义函数 |

### 4.3 `core/viz/` 新增

| 文件 | 新增函数 |
|------|----------|
| `variable_plots.py` | `variable_iv_plot` / `variable_woe_trend_plot` / `variable_psi_heatmap` / `variable_importance_grouped_plot` / `variable_missing_badrate_plot` |
| `score_plots.py` | `score_ks_plot` / `score_distribution_comparison_plot` / `score_badrate_bin_plot` / `score_lift_plot` / `score_approval_badrate_curve` |
| `strategy_plots.py` | `feature_trend_by_time` / `feature_drift_comparison` / `feature_effectiveness_by_segment` / `feature_cross_heatmap` / `population_drift_monitor` / `segment_scorecard_comparison` |

### 4.4 `core/eda/` 新增

| 文件 | 新增函数 |
|------|----------|
| `population.py` 🆕 | `population_profile` / `population_shift_analysis` / `population_monitoring_report` / `segment_drift_analysis` / `feature_cross_segment_effectiveness` |
| `strategy.py` 🆕 | `approval_badrate_tradeoff` / `score_strategy_simulation` / `vintage_performance_summary` / `roll_rate_matrix` / `label_leakage_check` / `multi_label_correlation` |
| `stability.py` 扩充 | `feature_drift_report` / `score_drift_report` |

### 4.5 `core/rules/mining/` 新增

| 文件 | 新增类 |
|------|--------|
| `multi_label.py` ✅ | `MultiLabelRuleMiner` |

### 4.6 `report/` 新增

| 文件 | 新增函数 |
|------|----------|
| `model_report.py` 🆕 | `auto_model_report` |
| `rule_analysis_report.py` ✅ | `multi_label_rule_report` |
| `population_drift_report.py` ✅ | `population_drift_report` |

### 4.7 `core/selectors/` 新增

| 文件 | 新增类 |
|------|--------|
| `stability_selector.py` ✅ | `StabilityAwareSelector` |

### 4.8 `core/viz/` 新增

| 文件 | 新增内容 |
|------|----------|
| `style.py` ✅ | `set_style` / `reset_style` / `get_palette` / `get_font_sizes` / 主题系统 / 中文字体自动检测 |

### 4.9 `core/models/` 新增

| 文件 | 新增内容 |
|------|----------|
| `tuning.py` ✅ | `TuningObjective.ks_lift_combined` / `TuningObjective.tail_purity_ks` |
| `scorecard.py` ✅ | `ScoreCard.export_deployment_code()` (SQL/Python/Java) |

### 4.7 顶层 `hscredit/__init__.py` 新增导出

```python
# Phase 1
from .core.metrics.finance import lift_at, lift_monotonicity_check

# Phase 3
from .core.viz import (
    variable_iv_plot, variable_woe_trend_plot, variable_psi_heatmap,
    variable_importance_grouped_plot, variable_missing_badrate_plot,
    score_ks_plot, score_distribution_comparison_plot,
    score_badrate_bin_plot, score_lift_plot, score_approval_badrate_curve,
    feature_trend_by_time, feature_drift_comparison,
    feature_effectiveness_by_segment, feature_cross_heatmap,
    population_drift_monitor, segment_scorecard_comparison,
)

# Phase 4
from .core.eda import (
    population_profile, population_shift_analysis, population_monitoring_report,
    segment_drift_analysis, feature_cross_segment_effectiveness,
    approval_badrate_tradeoff, score_strategy_simulation,
    vintage_performance_summary, roll_rate_matrix,
    label_leakage_check, multi_label_correlation,
    feature_drift_report, score_drift_report,
)

# Phase 5
from .core.rules.mining import MultiLabelRuleMiner

# Phase 6
from .report import auto_model_report, multi_label_rule_report
```

---

## 五、版本规划

| 版本 | 主要内容 | Phase | 状态 |
|------|----------|-------|------|
| `v0.1.1` | Bug 修复 + 参数名统一 + 调参目标扩展 + 评分卡部署代码 + 多标签规则挖掘 + viz 统一样式 + 客群偏移监控报告 + StabilityAwareSelector | Phase 1-5 部分 | ✅ 已完成 |
| `v0.2.0` | LIFT@任意值 + 单调性检验 + 调参报告 | Phase 1-2 剩余 | 待开发 |
| `v0.3.0` | viz 模块持续扩充（变量分析/评分分析/策略分析图样式统一化） | Phase 3 | 待开发 |
| `v0.4.0` | EDA 体系化（population.py / strategy.py / 偏移专项） | Phase 4 | 待开发 |
| `v0.5.0` | 规则报告深化（单规则详细分析、规则集模拟Sheet） | Phase 5 | 待开发 |
| `v0.6.0` | 模型报告完整实现（10-sheet Excel模板） | Phase 6 | 待开发 |
| `v0.7.0` | 特征工程扩充 + 已有功能持续迭代 | Phase 7-8 | 持续 |

---

## 六、实现顺序建议

```
Phase 1（指标扩充）
    → 为 Phase 2（调参）和 Phase 6（报告）提供基础指标
    → 优先实现，1-2天可完成

Phase 2（调参扩充）
    → 依赖 Phase 1 的 lift_at 和 lift_monotonicity_check
    → 2-3天可完成

Phase 3（viz重设计）和 Phase 4（EDA补强）
    → 相互独立，可并行实现
    → 各3-5天

Phase 5（多标签规则报告）
    → 依赖 viz 的图表函数（Phase 3）
    → 3-5天

Phase 6（模型报告）
    → 依赖 Phase 1（LIFT指标）+ Phase 3（viz图表）
    → 5-7天，是阶段性最大交付物

Phase 7-8（持续迭代）
    → 无强依赖，随时穿插实现
```

---

## 七、代码规范（继承现有风格）

1. **命名规范**：类名 PascalCase，函数/方法 snake_case，常量 UPPER_SNAKE_CASE
2. **Docstring**：Google 风格，每个 public API 含最小可运行示例
3. **输入处理**：所有 public API 输入走 `hscredit.utils.input_utils` 统一处理
4. **输出格式**：统计表列名使用中文，与现有模块保持一致
5. **Optional 依赖**：缺失时给出清晰的安装提示（如 `pip install shap`）
6. **只增不改**：不破坏现有 API，大写别名保留，旧参数名兼容
7. **Excel 报告**：统一使用 `ExcelWriter`，样式参考 `feature_report.py`
