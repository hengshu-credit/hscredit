# hscredit 优化方案与迭代计划

> 版本：2026-04-12 v3
> 定位：面向金融信贷场景的完整风控建模工具包，覆盖策略分析人员与模型开发人员的全链路需求。
> 方法：基于 hscredit 代码风格审计 + toad / optbinning / scorecardpipeline 竞品对标分析编写。

---

## 一、代码风格与设计约定（供开发参考）

以下约定从现有代码中提炼，**所有新增代码必须遵循**。

### 1.1 命名与文档

| 维度 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `OptimalBinning`, `WOEEncoder`, `XGBoostRiskModel` |
| 函数/方法 | snake_case | `fit_transform`, `get_bin_table`, `bad_rate_trend` |
| 常量 | UPPER_SNAKE_CASE | `VALID_METHODS`, `XGBOOST_AVAILABLE` |
| 模块 docstring | 中文，描述模块职责 | `"""统一分箱接口 - 整合所有分箱方法."""` |
| 类 docstring 结构 | `**参数**` → `**属性**` → `**参考样例**` 三段式 | 见 `BaseBinning` / `OptimalBinning` |
| 参数文档 | reST 风格 `:param name:` + 中文说明 | `:param max_n_bins: 最大分箱数，默认为5` |
| 实践经验注释 | `- 内部经验: ...` 嵌在参数说明中 | 见 `XGBoostRiskModel` 各参数 |
| 输出 DataFrame 列名 | **中文** | `分箱`, `样本总数`, `坏样本率`, `分档WOE值`, `指标IV值` |

### 1.2 API 设计模式

```
双 API 风格（所有有监督组件统一支持）:

sklearn 风格:              binner.fit(X, y)         # X 为特征矩阵, y 为目标
scorecardpipeline 风格:    binner.fit(df)           # df 包含 target 列, 通过 target='target' 指定
混合风格:                  binner.fit(df, y=ext_y)  # y 参数优先于 df 中的 target 列
```

| 约定 | 说明 |
|------|------|
| 基类继承 | `BaseEstimator, TransformerMixin`（sklearn Pipeline 兼容） |
| `target` 参数 | 所有有监督组件构造函数第一参数，默认 `'target'` |
| 可选依赖 | `try/except ImportError` + `_AVAILABLE` 布尔标志 + 中文安装提示 |
| 异常体系 | `HSCreditError` → `ValidationError` / `NotFittedError` / `FeatureNotFoundError` / `DependencyError` |
| 错误消息语言 | 中文 |
| 类型注解 | 构造函数所有参数必须标注（`from typing import ...`） |
| 属性命名 | fit 后产生的属性以 `_` 结尾：`splits_`, `n_bins_`, `bin_tables_`, `iv_` |

### 1.3 模块组织原则

```
core/xxx/          → 核心算法（纯计算逻辑，不涉及 IO/报告）
report/            → 报告生成（Excel/HTML，可引用 core）
excel/             → Excel 底层工具（ExcelWriter / dataframe2excel）
utils/             → 通用工具（seed_everything / 数据集 / describe / pandas 扩展 / logger）
```

---

## 二、竞品对标分析

### 2.1 toad（amphibian-dev/toad）— 实用主义标杆

**核心流程**：`detect()` → `quality()` → `select()` → `Combiner` → `WOETransformer` → `ScoreCard` → `metrics`

| toad 功能 | hscredit 对应 | 状态 |
|-----------|-------------|------|
| `toad.detect()` 一行数据画像 | `eda.data_info()` + `eda.feature_summary()` | ✅ 已有（更丰富） |
| `toad.quality()` 一行 IV 筛查 | `eda.batch_iv_analysis()` | ✅ 已有 |
| `toad.selection.select()` 多条件筛选 | `CompositeFeatureSelector` (23 种) | ✅ 远超 |
| `toad.selection.stepwise()` 逐步回归 | `StepwiseSelector` | ✅ 已有 |
| `Combiner` 分箱（chi/dt/quantile/step/kmeans 5种） | `OptimalBinning` (17 种方法) | ✅ 远超 |
| `Combiner.export() / set_rules()` 规则导入导出 | `export() / load()` | ✅ 已有 |
| `WOETransformer` | `WOEEncoder` + `transform(metric='woe')` | ✅ 已有 |
| `toad.ScoreCard` PDO/Odds 转换 | `ScoreCard`（含 SQL/Python/Java 部署代码导出） | ✅ 更强 |
| `KS_bucket()` 分桶 KS 分析 | `ks_bucket()` | ✅ 已有 |
| `toad.plot` 基础图表 | `core.viz` 50+ 图表 | ✅ 远超 |
| `toad.nn` 深度学习 | — | ❌ 未有（低优先） |
| 拒绝推断 | — | ❌ **toad 也无** |

**结论：hscredit 在功能广度上已全面超越 toad。toad 的优势在于 API 极简（一行 `quality()`），hscredit 应保持专业深度的同时提供同等简洁的快捷入口。**

### 2.2 optbinning（guillermo-navas-palencia）— 学术严谨标杆

**核心特色**：MIP/CP 数学最优化求解，`BinningProcess` 批量处理，`BinningTable` 分析报告

| optbinning 功能 | hscredit 对应 | 状态 |
|----------------|-------------|------|
| MIP/CP 数学最优分箱 | `ORBinning`（基于 OR-Tools） | ✅ 已有 |
| 8 种单调性约束（ascending/descending/peak/valley/convex/concave/auto/auto_heuristic） | `BaseBinning.monotonic` (10 种) | ✅ 更多 |
| `BinningProcess` 批量分箱 + `.summary()` | `OptimalBinning` 支持 DataFrame | ✅ 已有 |
| `BinningTable.build()` 分箱表（Count/WoE/IV/JS） | `get_bin_table()` 中文分箱表（含 LIFT/坏账改善/累积指标） | ✅ 已有（列更丰富） |
| `BinningTable.analysis()` 复合质量评分（IV+Gini+HHI+单调性） | — | ❌ **应引入** |
| `BinningTable.plot()` 可视化 | `bin_plot()` + 多种图表 | ✅ 已有 |
| `Scorecard` (PDO/Odds 标准转换) | `ScoreCard` | ✅ 已有 |
| `OptimalBinning2D` 二维交互分箱 | — | ❌ **独特功能** |
| `ContinuousOptimalBinning` 连续目标分箱 | — | ❌ 可考虑 |
| `MulticlassOptimalBinning` 多分类分箱 | — | ❌ 可考虑 |
| `OptimalPWBinning` 分段多项式分箱 | — | ❌ 低优先 |
| `CounterfactualExplanation` 反事实解释 | — | ❌ **高价值 XAI** |
| solver 状态报告 (`OPTIMAL/FEASIBLE/INFEASIBLE`) | — | ⚠️ 可增强 |
| `max_pvalue` 相邻箱统计显著性约束 | — | ⚠️ 可增强 |
| `prebinning_method` 预分箱 | `prebinning` 参数 | ✅ 已有 |
| `special_codes` 特殊值分箱 | `special_codes` 参数 | ✅ 已有 |

**结论：optbinning 在 2D 分箱、连续/多分类目标支持、复合质量评分、反事实解释上有独到优势。hscredit 应选择性引入高价值功能。**

### 2.3 scorecardpipeline（itlubber）— 集成完整性标杆

**核心特色**：包装 toad + optbinning + scorecardpy 三引擎，全 Pipeline 兼容，`target` 随 DataFrame 传递

| scorecardpipeline 功能 | hscredit 对应 | 状态 |
|-----------------------|-------------|------|
| 全 sklearn Pipeline 兼容 | 所有组件继承 `BaseEstimator` | ✅ 已有 |
| `target` 在 DataFrame 中传递 | 双 API 风格 | ✅ 已有 |
| 多引擎分箱（toad+optbinning 8种方法） | 17 种原生方法（不依赖外部库） | ✅ 更强 |
| 17 种特征筛选器 + `dropped` 审计轨迹 | 23 种 + `SelectionReportCollector` | ✅ 更多 |
| `ITLubberLogisticRegression` 统计摘要（coef/p-value/VIF） | `LogisticRegression(calculate_stats=True)` | ✅ 已有 |
| `Rule` 规则代数 (`&`/`|`/`~`) + `report()` | `Rule` / `RuleSet` / `RulesClassifier` | ✅ 已有 |
| `DecisionTreeRuleExtractor` | `TreeRuleExtractor` | ✅ 已有 |
| `ExcelWriter` + `dataframe2excel()` 专业报告 | `ExcelWriter` + `dataframe2excel` | ✅ 已有 |
| `feature_bin_stats()` 多标签分析 | `feature_bin_stats()` | ✅ 已有 |
| `auto_data_testing_report()` 全自动化报告 | `auto_feature_analysis()` + `auto_model_report()` | ✅ 已有 |
| `NumExprDerive` 表达式衍生 | `NumExprDerive` | ✅ 已有 |
| 金融计算 (NPV/IRR/PMT) | `core.financial` | ✅ 已有 |
| `scorecard2pmml()` PMML 导出 | `pyproject.toml` 已配 pmml 依赖 | ⚠️ 集成可增强 |
| `BoxCoxScoreTransformer` 分数变换 | `StandardScoreTransformer` | ⚠️ 可增加变体 |
| `auto_eda_sweetviz()` 自动 EDA | `eda.eda_summary()` + `eda.generate_report()` | ✅ 原生实现 |
| `class_steps()` Pipeline 组件提取 | `ScoreCard(pipeline=...)` 自动提取 | ✅ 已有 |

**结论：hscredit 已覆盖 scorecardpipeline 的绝大多数功能，且核心算法为原生实现而非第三方包装。**

### 2.4 hscredit 独有优势（竞品均不具备）

| 独有功能 | 说明 |
|---------|------|
| **9 种风控专用损失函数** | Focal / AsymmetricFocal / WeightedBCE / CostSensitive / BadDebt / ApprovalRate / ProfitMax / OrdinalRank / LiftFocused |
| **EDA 客群分析** | `population_profile` / `population_shift_analysis` / `population_monitoring_report` / `segment_drift_analysis` / `feature_cross_segment_effectiveness` |
| **EDA 策略分析** | `approval_badrate_tradeoff` / `score_strategy_simulation` / `roll_rate_matrix` / `label_leakage_check` / `multi_label_correlation` |
| **EDA 偏移分析** | `feature_drift_report` / `score_drift_report` / `batch_psi_analysis` / `psi_cross_analysis` |
| **SwapAnalyzer** | 规则置换四象限分析（通过/拒绝 × 好/坏），982 行完整实现 |
| **OverduePredictor** | MOB 逾期率预测与报告 |
| **ModelTuner / AutoTuner** | 风控专用调参目标（ks / lift_head / head_ks / ks_lift_combined） |
| **CalibratedModel** | 模型概率校准 |
| **ScoreCard 部署代码导出** | `export_deployment_code(language='sql'/'python'/'java')` |
| **50+ 可视化** | binning_plots / risk_plots / score_plots / strategy_plots / variable_plots 五大类 |

---

## 三、真实缺口与优先级

| 优先级 | 缺口 | 竞品参考 | 价值说明 |
|--------|------|---------|---------|
| 🔴 P0 | **特征工程模块薄弱**（仅 NumExprDerive） | 竞品同样薄弱 | 三大竞品均无完整特征工程，**差异化机会最大** |
| 🔴 P0 | **拒绝推断（Reject Inference）** | **三大竞品均无** | 信贷建模刚需，**独家差异化** |
| 🟠 P1 | 分箱质量评分 & 批量导出 | optbinning `BinningTable.analysis()` | 提升分箱模块专业度 |
| 🟠 P1 | 规则运营工具（覆盖率仿真/跨期追踪/冲突检测） | — | 策略人员刚需 |
| 🟡 P2 | 二维交互分箱 | optbinning `OptimalBinning2D` | 交互效应分析 |
| 🟡 P2 | 模型报告 SHAP 集成 / ScoreCard 分析增强 | — | 报告完整度 |
| 🟡 P2 | 反事实解释 | optbinning `CounterfactualExplanation` | 监管合规 |
| 🔵 P3 | LiftSelector ratio | — | 小幅改进 |
| 🔵 P3 | README / info() 修正 / 测试补全 / CI·CD / API 文档 | — | 工程质量 |

---

## 四、迭代规划（按价值主题）

### Theme 1：特征工程模块扩充 🔴 P0

**目标**：将 `core/feature_engineering/` 从单一的 `NumExprDerive` 扩展为完整的 sklearn Pipeline 兼容 Transformer 体系。**三大竞品在此领域均薄弱，是 hscredit 建立差异化优势的最佳切入点。**

#### 1-1 新增 `core/feature_engineering/time_features.py`

```python
class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """时序特征生成器（sklearn Pipeline 兼容）.

    从日期列衍生多维时间特征，信贷场景高频使用。

    **参数**

    :param date_col: 日期列名
    :param features: 生成的时序特征列表，支持：
        'year' / 'month' / 'day' / 'weekday' / 'quarter' /
        'is_weekend' / 'is_month_end' / 'is_month_start' /
        'days_since_epoch' / 'days_to_year_end'
    :param ref_cols: 参考日期列名列表，自动生成 'days_since_{ref_col}' 天数差特征
    :param drop_original: 是否删除原始日期列，默认 True

    **参考样例**

    >>> gen = TimeFeatureGenerator(
    ...     date_col='apply_date',
    ...     features=['month', 'weekday', 'is_weekend', 'quarter'],
    ...     ref_cols=['id_issue_date']
    ... )
    >>> X_new = gen.fit_transform(X)
    """
```

#### 1-2 新增 `core/feature_engineering/cross_features.py`

```python
class CrossFeatureGenerator(BaseEstimator, TransformerMixin):
    """交叉特征生成器.

    对指定特征对进行交叉运算，生成衍生变量。
    信贷场景典型用法：收入/负债比、年龄×额度等。

    **参数**

    :param pairs: 特征对列表，如 [('income', 'debt'), ('age', 'credit_limit')]
    :param operations: 运算列表，支持 'ratio' / 'diff' / 'product' / 'log_ratio' / 'square_diff'
    :param auto_pairs: 若为 True 且未指定 pairs，自动枚举所有数值列组合
        - 内部经验: 特征数>30时不建议开启，组合数爆炸
    :param naming: 命名规则，'auto' 自动命名 (如 'income_div_debt')，或自定义 callable

    **参考样例**

    >>> gen = CrossFeatureGenerator(
    ...     pairs=[('income', 'debt'), ('age', 'credit_limit')],
    ...     operations=['ratio', 'diff']
    ... )
    >>> X_new = gen.fit_transform(X)  # 新增 income_div_debt, income_sub_debt, ...
    """
```

#### 1-3 新增 `core/feature_engineering/preprocessing.py`

```python
class MissingValueImputer(BaseEstimator, TransformerMixin):
    """缺失值填充器（保留 DataFrame 格式和列名）.

    **参数**

    :param strategy: 填充策略，默认 'median'
        - 'mean' / 'median' / 'mode' / 'constant' / 'model'
        - 'model': 使用 LightGBM 预测填充（需安装 lightgbm）
    :param fill_value: strategy='constant' 时的填充值，默认 -999
    :param infer_per_column: 是否按列数据类型自动选择策略，默认 True
        - True: 数值列用 median，类别列用 mode
    """

class OutlierClipper(BaseEstimator, TransformerMixin):
    """异常值截断器.

    fit 阶段记录截断边界，transform 阶段应用。

    **参数**

    :param method: 截断方法，默认 'quantile'
        - 'quantile': 按分位数截断
        - 'iqr': IQR 方法 (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - 'fixed': 固定边界
    :param lower_quantile: 下截断分位数，默认 0.01
    :param upper_quantile: 上截断分位数，默认 0.99
    :param clip_value: method='fixed' 时的 (min, max) 边界元组
    """

class FeatureScaler(BaseEstimator, TransformerMixin):
    """特征标准化/归一化（封装 sklearn，保留 DataFrame 列名和格式）.

    **参数**

    :param method: 标准化方法，默认 'standard'
        - 'standard' / 'minmax' / 'robust' / 'maxabs'
    :param features: 需要缩放的特征列表，None 时处理所有数值列
    """
```

#### 1-4 更新 `__init__.py` 导出

**预计工期**：3-5 天
**测试要求**：`tests/test_feature_engineering/` 目录，每个类独立测试文件

---

### Theme 2：拒绝推断模块 🔴 P0

**目标**：新增 `core/reject_inference/` 模块。**三大竞品（toad / optbinning / scorecardpipeline）均未提供此功能，是 hscredit 最大的独家差异化机会。**

#### 2-1 新增 `core/reject_inference/reject_inference.py`

```python
class RejectInference(BaseEstimator, TransformerMixin):
    """拒绝推断 - 处理审批偏差问题.

    信贷建模中，训练数据仅包含被批准的申请人，
    被拒绝的申请人没有表现标签。拒绝推断通过不同方法
    为被拒绝样本推断标签，减少样本选择偏差。

    **参数**

    :param method: 推断方法，默认 'hard_cutoff'
        - 'hard_cutoff': 硬截断法 — 用现有模型对拒绝样本打分，
          高于阈值标记为好，低于阈值标记为坏
        - 'fuzzy': 模糊增强法 — 用模型预测概率作为样本权重，
          加权加入训练集
        - 'parceling': 分组法 — 按评分段分组，假设拒绝样本的
          坏率是批准样本��率的 k 倍
        - 'twin': 孪生法 — 仅使用审批前可获得的特征建模
    :param cutoff: hard_cutoff 方法的分数阈值，默认 None（自动取批准样本中位数）
    :param reject_bad_rate_multiplier: parceling 方法中拒绝样本坏率倍数，默认 2.0
        - 内部经验: 通常取 1.5-3.0，需结合业务判断
    :param model: 用于打分的预训练模型，默认 None（自动训练 LR）
    :param random_state: 随机种子，默认 None

    **参考样例**

    基本使用::

        >>> from hscredit.core.reject_inference import RejectInference
        >>> ri = RejectInference(method='fuzzy')
        >>> # X_approved: 已审批样本特征, y_approved: 已审批样本标签
        >>> # X_rejected: 被拒绝样本特征
        >>> X_aug, y_aug = ri.fit_transform(X_approved, y_approved, X_rejected=X_rejected)

    在建模流程中使用::

        >>> ri = RejectInference(method='parceling', reject_bad_rate_multiplier=2.0)
        >>> X_combined, y_combined = ri.augment(X_approved, y_approved, X_rejected)
    """
```

**预计工期**：3-4 天
**测试要求**：每种推断方法独立测试 + 与模拟数据的效果验证

---

### Theme 3：分箱模块增强 🟠 P1

**参考 optbinning 的专业特性进行补强。**

#### 3-1 分箱质量评分（参考 optbinning `BinningTable.analysis()`）

在 `BaseBinning` 新增 `get_quality_report()` 方法：

```python
def get_quality_report(self, feature: str) -> pd.DataFrame:
    """分箱质量综合评估.

    参考 optbinning BinningTable.analysis()，输出复合质量指标。

    输出指标:
    - IV值 / IV 解读（无预测力/弱/中/强/可疑）
    - KS值
    - Gini系数
    - HHI 集中度指数（衡量各箱样本是否均匀）
    - 单调性方向
    - 综合质量评分（0-1，加权组合以上指标）

    :param feature: 特征名
    :return: 质量报告 DataFrame
    """
```

#### 3-2 `OptimalBinning.batch_to_excel()`

```python
def batch_to_excel(
    self,
    output_path: str,
    feature_map: Dict[str, str] = None,
    include_plots: bool = True,
    include_quality: bool = True,
) -> str:
    """批量分箱结果输出到 Excel.

    首个 Sheet 为汇总页（特征名/IV/KS/Gini/分箱数/单调性/质量评分），
    后续每个特征一个 Sheet，含分箱统计表 + 分箱图（可选）。

    :param output_path: 输出 Excel 路径
    :param feature_map: 特征英文名→中文名映射
    :param include_plots: 是否嵌入分箱图，默认 True
    :param include_quality: 是否包含质量评分列，默认 True
    :return: 输出文件路径
    """
```

#### 3-3 新增 `BestPSIBinning`（`core/binning/best_psi_binning.py`）

```python
class BestPSIBinning(BaseBinning):
    """PSI 最优分箱：在约束条件下寻找使训练集/验证集 PSI 最小的切分点.

    适用场景：特征分布在训练集和测试集有偏移时，选择
    使两个数据集分布差异最小的分箱方案。

    **参数**

    :param target: 目标变量列名，默认 'target'
    :param max_n_bins: 最大分箱数，默认 5
    :param min_bin_size: 每箱最小样本占比，默认 0.05
    :param psi_target: PSI 约束上限，默认 0.1
        - 内部经验: PSI < 0.1 为稳定，0.1-0.25 需关注，> 0.25 显著偏移

    **参考样例**

    >>> binner = BestPSIBinning(max_n_bins=5)
    >>> binner.fit(X_train, y_train, X_val=X_val)
    >>> X_binned = binner.transform(X_test)
    """
```

#### 3-4 `auto_select_bins()` 工具函数

```python
def auto_select_bins(
    X: pd.Series,
    y: pd.Series = None,
    min_bins: int = 3,
    max_bins: int = 10,
    criterion: str = 'iv',
) -> int:
    """基于样本量和数据特征自动推断最优分箱数.

    规则:
    - n < 5000: 推荐 3-5 箱
    - 5000 ≤ n < 50000: 推荐 min(8, 唯一值数/3) 箱
    - n ≥ 50000: 推荐最多 10 箱
    - 有监督模式(传入y)下: 交叉验证选择使 criterion 最大的箱数

    :param X: 特征序列
    :param y: 目标变量序列，可选
    :param min_bins: 最小箱数，默认 3
    :param max_bins: 最大箱数，默认 10
    :param criterion: 评估准则 'iv' / 'ks' / 'entropy'，默认 'iv'
    :return: 推荐分箱数
    """
```

**预计工期**：3-4 天

---

### Theme 4：规则运营工具链 🟠 P1

**补齐 `rule_analysis.py`（当前仅 155 行）的规则运营能力。**

#### 4-1 规则集覆盖率仿真


ruleset_analysis 增加规则间覆盖重叠矩阵分析

#### 4-2 单条规则跨期追踪

```python
def rule_effectiveness_tracking(
    df_list: List[pd.DataFrame],
    labels: List[str],
    rule_expr: str,
    target: str,
) -> pd.DataFrame:
    """单条规则跨期有效性追踪.

    :param df_list: 各期数据集列表
    :param labels: 各期标签（如 ['2025-01', '2025-02', ...]）
    :param rule_expr: pandas query 表达式
    :param target: 目标变量列名
    :return: 各期覆盖率/坏率/LIFT 趋势表
    """
```

#### 4-3 规则冲突检测

```python
def detect_rule_conflicts(
    rules: List[Rule],
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """检测规则集中逻辑矛盾或高度重叠的规则对.

    基于数据的覆盖重叠率计算，标注高重叠（>80%）、
    完全包含关系、逻辑矛盾。

    :param rules: Rule 对象列表
    :param df: 数据集，用于计算覆盖重叠率。None 时仅做表达式分析
    :return: 规则对分析 DataFrame（规则A/规则B/重叠率/关系类型）
    """
```

**预计工期**：2-3 天

---

### Theme 5：模型报告增强 🟡 P2

#### 5-1 SHAP 报告集成

在 `QuickModelReport.to_excel()` 增加 `include_shap` 参数，自动追加 Sheet：

```
Sheet N - SHAP 解释性（可选）
  - SHAP 均值汇总表（特征名/平均|SHAP|/排名）
  - SHAP Summary Plot（蜂群图）嵌入图片
  - SHAP Bar Plot（特征重要性）嵌入图片
  - Top 5 特征 SHAP 依赖图
```

联动已有的 `ModelExplainer`，新增 `ModelExplainer.to_excel()` 方法。

#### 5-2 `ScoreCard.score_segment_analysis()`

```python
def score_segment_analysis(
    self,
    df: pd.DataFrame,
    target: str,
    n_bins: int = 10,
    features: List[str] = None,
) -> pd.DataFrame:
    """各评分段的客群特征分布对比分析.

    :param df: 含评分和特征的数据集
    :param target: 目标变量列名
    :param n_bins: 评分分箱数，默认 10
    :param features: 分析的特征列表，None 时自动选取 Top 10
    :return: 评分段分析 DataFrame（评分段/样本量/坏率/各特征均值）
    """
```

#### 5-3 `compare_models()` 增强

为已有函数增加可视化（雷达图 / KS·LIFT 曲线叠加 / 评分分布对比），输出统一使用 `ExcelWriter`。

**预计工期**：3-4 天

---

### Theme 6：二维交互分箱 🟡 P2

**参考 optbinning `OptimalBinning2D`，提供交互效应分析能力。**

```python
class InteractionBinning(BaseBinning):
    """二维交互分箱.

    对两个特征进行联合分箱，捕获交互效应。

    **参数**

    :param target: 目标变量列名，默认 'target'
    :param method: 分箱方法，默认 'tree'
        - 'tree': 二维决策树分箱
        - 'grid': 网格搜索（各维度独立分箱后组合）
    :param max_n_bins_x: X 轴最大分箱数，默认 5
    :param max_n_bins_y: Y 轴最大分箱数，默认 5
    :param min_bin_size: 每格最小样本占比，默认 0.02

    **参考样例**

    >>> binner = InteractionBinning(max_n_bins_x=4, max_n_bins_y=4)
    >>> binner.fit(df[['income', 'age']], y)
    >>> table = binner.get_interaction_table()  # 二维坏率表
    >>> binner.plot_interaction()               # 交互热力图
    """
```

**预计工期**：3-4 天

---

### Theme 7：反事实解释 🟡 P2

**参考 optbinning `CounterfactualExplanation`，提供监管合规 XAI 能力。**

```python
class CounterfactualExplainer:
    """反事实解释器.

    回答"最小改变哪些特征值，可以翻转模型决策？"
    监管合规场景下用于向申请人解释拒绝原因和改善建议。

    **参数**

    :param model: 已训练模型（需支持 predict_proba）
    :param feature_names: 特征名列表
    :param feature_ranges: 各特征可变范围 Dict[str, Tuple[min, max]]
    :param immutable_features: 不可变特征列表（如 'age', 'gender'）
    :param method: 搜索方法，默认 'greedy'
        - 'greedy': 贪心搜索（快速）
        - 'genetic': 遗传算法（全局优化）

    **参考样例**

    >>> explainer = CounterfactualExplainer(model, feature_names=X.columns.tolist())
    >>> explainer.fit(X_train)
    >>> cf = explainer.explain(X_test.iloc[0], target_class=0)
    >>> print(cf)  # 输出最小特征变化方案
    """
```

**预计工期**：2-3 天

---

### Theme 8：筛选器补强 🔵 P3

#### 8-1 `LiftSelector` 增加 `ratio` 参数

```python
class LiftSelector(BaseFeatureSelector):
    def __init__(
        self,
        threshold: float = 1.5,
        ratio: float = 0.10,    # 新增：LIFT@比例，默认 10%
        ascending: bool = False,
        target: str = 'target',
        n_jobs: int = 1,
    ):
        """LIFT 筛选器，支持 lift@1%/5%/10% 等自定义比例.

        :param threshold: LIFT 阈值，低于此值的特征被剔除
        :param ratio: LIFT 计算的覆盖率，默认 0.10（即 LIFT@10%）
            - 内部经验: 风控场景常用 lift@5% 或 lift@10%
        """
```


**预计工期**：1-2 天

---

### Theme 9：工程质量与基础设施 🔵 P3

#### 9-1 README.md / `info()` 修正

- "7种损失函数" → "9种损失函数"
- 补充 EDA 客群/策略分析、拒绝推断等新能力
- `info()` 删除"待实现模块: core.metrics"

#### 9-2 测试补全

| 测试文件 | 覆盖模块 |
|----------|----------|
| `tests/test_eda/test_population.py` | population.py |
| `tests/test_eda/test_strategy.py` | strategy.py |
| `tests/test_feature_engineering/test_*.py` | 特征工程新类 |
| `tests/test_reject_inference/test_*.py` | 拒绝推断 |
| `tests/test_binning/test_best_psi_binning.py` | BestPSIBinning |
| `tests/test_rules/test_rule_operations.py` | 规则运营工具 |
| `tests/test_models/test_losses_ranking.py` | OrdinalRankLoss / LiftFocusedLoss |


## 五、版本规划

| 版本 | 主要内容 | Theme | 状态 |
|------|----------|-------|------|
| `v0.1.0` | 核心功能完整版 | — | ✅ 当前 |
| `v0.1.1` | README/info() 修正 + 已有模块测试补全 | 9-1/9-2 | 🔜 立即 |
| `v0.2.0` | **特征工程** + **拒绝推断**（独家差异化） | Theme 1 + 2 | 待开发 |
| `v0.3.0` | 分箱增强（质量评分/BestPSI/batch_to_excel） + 规则运营 | Theme 3 + 4 | 待开发 |
| `v0.4.0` | 报告增强 + 二维分箱 + 反事实解释 | Theme 5 + 6 + 7 | 待开发 |

---

## 六、实施路线图

```
v0.1.1 紧急修正（1-2 天）
  ├── README.md / info() 数据修正
  └── population/strategy/stability/ranking_loss 测试补全

         ↓

v0.2.0 核心扩充（6-9 天）  ← 最大价值
  ├── [特征工程] TimeFeatureGenerator / CrossFeatureGenerator
  ├── [特征工程] MissingValueImputer / OutlierClipper / FeatureScaler
  ├── [拒绝推断] RejectInference（hard_cutoff/fuzzy/parceling/twin）  ← 独家
  └── 对应测试 + examples notebook

         ↓

v0.3.0 专业增强（5-7 天，可并行）
  ├── [分箱] get_quality_report + batch_to_excel + BestPSIBinning + auto_select_bins
  ├── [规则] rule_effectiveness_tracking + detect_rule_conflicts
  └── 对应测试

         ↓

v0.4.0 分析增强（5-8 天，可并行）
  ├── [报告] SHAP Sheet 集成 + score_segment_analysis + compare_models 增强
  ├── [分箱] InteractionBinning 二维交互分箱
  ├── [XAI] CounterfactualExplainer 反事实解释
  └── 对应测试

         ↓

v0.5.0 筛选 + 工程化（3-5 天）
  ├── LiftSelector ratio
```

**总计预估**：22-30 个工作日（各 Theme 间可并行开发）

---

## 七、竞争定位总结

```
                    分箱算法  特征筛选  损失函数  EDA分析  规则引擎  报告生成  特征工程  拒绝推断  XAI解释
toad                ★★★☆     ★★☆☆     ☆☆☆☆    ★☆☆☆     ☆☆☆☆    ☆☆☆☆     ☆☆☆☆    ☆☆☆☆    ☆☆☆☆
optbinning          ★★★★     ★★☆☆     ☆☆☆☆    ☆☆☆☆     ☆☆☆☆    ★★★☆     ☆☆☆☆    ☆☆☆☆    ★★★☆
scorecardpipeline   ★★★☆     ★★★☆     ☆☆☆☆    ★☆☆☆     ★★★☆    ★★★★     ★☆☆☆    ☆☆☆☆    ☆☆☆☆
hscredit (当前)     ★★★★     ★★★★     ★★★★    ★★★★     ★★★☆    ★★★★     ★☆☆☆    ☆☆☆☆    ☆☆☆☆
hscredit (v0.5目标) ★★★★     ★★★★     ★★★★    ★★★★     ★★★★    ★★★★     ★★★★    ★★★★    ★★★☆
```

**hscredit 的战略定位**：在 toad 的实用性、optbinning 的专业性、scorecardpipeline 的完整性基础上做到全面超越，同时通过「拒绝推断」「完整特征工程」「反事实解释」建立**独家竞争壁垒**。

