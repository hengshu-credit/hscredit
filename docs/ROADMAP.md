# hscredit Roadmap

> 版本：2026-06-18
> 定位：面向金融信贷风控策略分析与评分模型研发的 Python 量化建模工具箱。
> 方法：基于当前代码结构、公开 API、业务场景和 toad / optbinning / scorecardpipeline / scorecardpy 等同类库对标整理。

## 一、产品定位

`hscredit` 的核心目标不是单点复刻某个开源库，而是围绕信贷风控团队的日常工作流，形成一套可计算、可解释、可交付的建模工具链。

信贷风控建模的真实链路通常包含：

1. 数据质量检查和客群画像。
2. 目标变量、逾期标签、Vintage 和 Roll Rate 分析。
3. 单变量有效性、分箱、WOE、IV、KS、Lift、PSI 分析。
4. 多维特征筛选和稳定性筛选。
5. 评分卡、机器学习模型和策略规则模型训练。
6. 模型评估、调参、概率校准、分数转换和漂移监控。
7. 策略规则挖掘、规则集效果评估和新旧策略 Swap 分析。
8. Excel 报告、图表和上线交付材料生成。

`hscredit` 当前已经覆盖上述链路中的大部分常见任务，但仍处于早期版本。下一阶段重点是增强工程可信度、文档可用性、测试覆盖、特征工程、拒绝推断、规则运营和可解释性能力。

## 二、当前代码结构与已实现能力

### 2.1 模块总览

```text
hscredit/
├── core/
│   ├── binning/              # 分箱算法
│   ├── encoders/             # 特征编码
│   ├── selectors/            # 特征筛选
│   ├── models/               # 风控模型、评分卡、损失函数、调参、评估
│   ├── metrics/              # 指标体系
│   ├── eda/                  # 数据探索、策略、Vintage、稳定性分析
│   ├── rules/                # 规则表达式
│   ├── financial/            # 金融计算
│   ├── feature_engineering/  # 表达式特征衍生
│   └── viz/                  # 可视化
├── report/                   # 规则挖掘、模型报告、Swap、逾期预测、漂移报告
├── excel/                    # Excel 输出
└── utils/                    # pandas 扩展、数据集、IO、日志、随机种子
```

### 2.2 已实现能力清单

| 领域 | 模块 | 已实现能力 |
|:---|:---|:---|
| 数据探索 | `core.eda` | 数据概览、缺失分析、目标分布、坏率趋势、特征分布、相关性、IV/WOE、PSI/CSI、客群迁移、策略仿真、Vintage、Roll Rate |
| 分箱 | `core.binning` | 等宽、等频、树、CART、卡方、Best IV、Best KS、Best Lift、MDLP、OR-Tools、CP-SAT、KMeans、单调、遗传算法、平滑、核密度、目标坏率、二维分箱 |
| 编码 | `core.encoders` | WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、Cardinality、GBM 编码 |
| 筛选 | `core.selectors` | 缺失率、众数率、基数、方差、相关性、VIF、IV、Lift、PSI、模型重要性、零重要性、RFE、序列选择、逐步回归、Boruta、互信息、卡方、F 检验、稳定性感知、组合筛选 |
| 模型 | `core.models` | LogisticRegression、RandomForest、ExtraTrees、GradientBoosting、XGBoost、LightGBM、CatBoost、NGBoost、ScoreCard、RuleSet、RulesClassifier |
| 损失函数 | `core.models.losses` | Focal、非对称 Focal、加权 BCE、成本敏感、坏账、审批率、利润最大化、排序、KS 聚焦、Top-K 捕获、金额加权、期望收益等 |
| 调参与评估 | `core.models.tuning` / `evaluation` | Optuna 调参、模型报告、概率校准、解释性分析 |
| 指标 | `core.metrics` | KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、分类/回归指标、分箱统计 |
| 规则 | `core.rules` / `report.mining` | Rule 表达式、规则优化、单特征规则、多特征规则、多标签规则、树规则提取、手工树分析、规则指标 |
| 策略分析 | `report.swap_analysis` / `report.rule_analysis` | 策略置换、规则集分析、多标签规则分析、规则 Swap 分析 |
| 逾期预测 | `report.overdue_predictor` | MOB 逾期预测、逾期预测报告 |
| 报告 | `report.model_report` / `excel` | 快速模型报告、模型对比、ExcelWriter、dataframe2excel、pandas 扩展输出 |
| 金融计算 | `core.financial` | FV、PV、PMT、NPER、IPMT、PPMT、RATE、NPV、IRR、MIRR |

### 2.3 设计约定

| 维度 | 约定 |
|:---|:---|
| API 风格 | 核心建模组件尽量兼容 `fit` / `transform` / `predict` / `predict_proba` |
| Pipeline | 主要分箱、编码、筛选和模型组件遵循 sklearn 风格，便于 Pipeline 与调参集成 |
| 目标列 | 支持 `X, y` 风格，也支持 DataFrame 内通过 `target` 参数指定目标列 |
| 输出语言 | 用户可见的列名、错误消息、报告内容尽量使用中文 |
| 可选依赖 | Boosting、深度学习、调参、解释、PMML 按 optional extras 安装 |
| 报告交付 | 重要分析结果优先支持 DataFrame 和 Excel 输出 |

## 三、业务场景映射

### 3.1 贷前评分卡建模

| 业务任务 | 推荐能力 |
|:---|:---|
| 数据质量检查 | `eda.data_info`、`eda.missing_analysis`、`eda.data_quality_report` |
| 变量有效性分析 | `eda.batch_iv_analysis`、`eda.woe_analysis`、`metrics.iv`、`metrics.lift` |
| 分箱 | `OptimalBinning`、`BestIVBinning`、`BestKSBinning`、`MonotonicBinning`、`GeneticBinning` |
| WOE 编码 | `WOEEncoder`、`OptimalBinning.transform(metric='woe')` |
| 特征筛选 | `IVSelector`、`VIFSelector`、`CorrSelector`、`PSISelector`、`CompositeFeatureSelector` |
| 建模 | `LogisticRegression`、`ScoreCard` |
| 评估 | `ks`、`auc`、`gini`、`lift_table`、`ks_bucket` |
| 交付 | `auto_model_report`、`ExcelWriter`、`dataframe2excel` |

### 3.2 机器学习风控模型

| 业务任务 | 推荐能力 |
|:---|:---|
| 树模型训练 | `RandomForestRiskModel`、`GradientBoostingRiskModel`、`XGBoostRiskModel`、`LightGBMRiskModel`、`CatBoostRiskModel` |
| 不平衡样本处理 | `FocalLoss`、`BalancedFocalLoss`、`WeightedBCELoss` |
| 业务目标优化 | `BadDebtLoss`、`ApprovalRateLoss`、`ProfitMaxLoss`、`TopKBadCaptureLoss`、`AmountWeightedLoss` |
| 模型调参 | `ModelTuner`、`AutoTuner`、`TuningObjective` |
| 概率校准 | `PlattCalibrator`、`IsotonicCalibrator`、`ProbabilityCalibrator`、`CalibratedModel` |
| 解释性 | `ModelExplainer`、`plot_feature_importance`、SHAP 可选依赖 |

### 3.3 策略规则挖掘与运营

| 业务任务 | 推荐能力 |
|:---|:---|
| 单变量规则发现 | `SingleFeatureRuleMiner` |
| 交叉规则发现 | `MultiFeatureRuleMiner` |
| 多标签规则分析 | `MultiLabelRuleMiner`、`multi_label_rule_analysis` |
| 树模型规则提取 | `TreeRuleExtractor`、`DecisionTreeAnalyzer`、`ManualTreeExtractor` |
| 规则指标 | `RuleMetrics`、`calculate_rule_metrics`、`rule_lift` |
| 规则分类器 | `RuleSet`、`RulesClassifier` |
| 策略置换 | `SwapAnalyzer`、`swap_analysis`、`rule_swap_analysis` |

### 3.4 贷后监控与稳定性

| 业务任务 | 推荐能力 |
|:---|:---|
| 客群漂移 | `population_shift_analysis`、`population_monitoring_report`、`population_drift_monitor` |
| 变量漂移 | `psi_analysis`、`batch_psi_analysis`、`feature_drift_report` |
| 分数漂移 | `score_drift_report`、`ScoreDriftCalibrator`、`QuantileAligner` |
| Vintage / MOB | `vintage_analysis`、`vintage_summary`、`OverduePredictor` |
| Roll Rate | `roll_rate_matrix`、`roll_rate_analysis` |
| 策略阈值仿真 | `approval_badrate_tradeoff`、`score_strategy_simulation`、`threshold_analysis_plot` |

## 四、竞品对标

### 4.1 toad

`toad` 是评分卡建模领域常用的实用型工具，典型流程包括 `detect`、`quality`、`select`、`Combiner`、`WOETransformer`、`ScoreCard` 和基础指标。

| 维度 | toad | hscredit 当前状态 |
|:---|:---|:---|
| 易用性 | API 极简，快速上手 | 需要提供更多快捷入口，降低学习成本 |
| 分箱 | 常用分箱方法成熟 | 已实现更多分箱方法，但需要持续验证边界条件和性能 |
| 特征筛选 | 支持常见筛选 | 已实现更丰富的筛选器和组合筛选报告 |
| 评分卡 | 成熟易用 | 已实现 ScoreCard，并支持评分转换和组件提取 |
| 报告 | 相对基础 | hscredit 更强调中文 Excel 报告和业务交付 |
| 策略分析 | 非核心能力 | hscredit 已实现更多规则、Swap、Vintage、客群分析能力 |

建议：学习 `toad` 的简洁入口，补充 `quality`、`detect`、`select` 等快捷 API 风格的高层封装。

### 4.2 optbinning

`optbinning` 是最优分箱方向的专业库，优势在数学规划分箱、分箱质量分析、二维分箱、连续/多分类目标和反事实解释。

| 维度 | optbinning | hscredit 当前状态 |
|:---|:---|:---|
| 数学最优分箱 | MIP/CP 体系成熟 | 已有 OR-Tools、CP-SAT 等实现，但质量报告和求解状态说明可增强 |
| 分箱质量分析 | `BinningTable.analysis()` 专业 | 需要补充分箱质量评分和统计显著性分析 |
| 二维分箱 | 支持 `OptimalBinning2D` | 已有 `OptimalBinning2D` 文件与导出，需要补充文档、示例和测试 |
| 多目标类型 | 连续、多分类等支持完善 | 当前重点仍是二分类信贷风控场景 |
| XAI | 提供反事实解释 | hscredit 尚需规划反事实解释和合规解释报告 |

建议：优先补充分箱质量分析、求解状态报告、二维分箱示例和反事实解释能力。

### 4.3 scorecardpipeline

`scorecardpipeline` 将 `toad`、`scorecardpy`、`optbinning` 等能力封装为 sklearn Pipeline 风格，强调端到端评分卡建模、Excel 报告、规则分析和 PMML 交付。

| 维度 | scorecardpipeline | hscredit 当前状态 |
|:---|:---|:---|
| Pipeline 风格 | 集成度高 | hscredit 继承该方向，核心组件尽量兼容 sklearn 风格 |
| target 传递 | DataFrame + target 体验好 | hscredit 多数监督组件支持 `target` 参数 |
| 报告 | Excel 报告成熟 | hscredit 已有 Excel 与模型报告，需继续补齐模板和示例 |
| 规则分析 | 较完整 | hscredit 已有 Rule、RuleSet、规则挖掘、树规则和 Swap 分析 |
| 依赖 | 底层依赖较多 | hscredit 更强调原生实现和低依赖基础安装 |
| 维护方向 | 与 hscredit 存在传承关系 | hscredit 应承接其成熟经验并减少多库拼装成本 |

建议：保留 scorecardpipeline 的 Pipeline、报告、PMML 和规则分析经验，同时强化 hscredit 的原生实现、文档和测试。

### 4.4 scorecardpy

`scorecardpy` 提供评分卡基础流程，适合快速完成分箱、WOE、评分卡建模。

| 维度 | scorecardpy | hscredit 当前状态 |
|:---|:---|:---|
| 评分卡主流程 | 简单直接 | hscredit 覆盖评分卡主流程，并扩展模型、策略、报告和稳定性分析 |
| 工程化 | 相对基础 | hscredit 更贴近 sklearn 和 Python 工程体系 |
| 中文报告 | 非重点 | hscredit 输出更贴近中文风控交付 |

## 五、竞争策略

### 5.1 不建议的外部表述

| 不建议写法 | 原因 | 建议写法 |
|:---|:---|:---|
| 全面超越 toad / optbinning / scorecardpipeline | 缺少统一基准，容易夸大 | 在信贷风控全流程集成和中文化交付上形成互补优势 |
| hscredit 全搞定，告别所有库 | 高级分箱、XAI、PMML、CI 等仍需增强 | 减少多库拼装成本，覆盖常见评分卡建模与策略分析流程 |
| 分箱能力超过 optbinning | optbinning 在数学规划和质量分析上仍很强 | 提供多种分箱方法，并持续增强质量分析和二维分箱能力 |
| 生产级全面成熟 | 当前早期版本，测试和文档仍需补强 | 面向生产实践设计，持续完善工程化能力 |
| 支持所有 Python 新版本和所有可选依赖 | 可选依赖对新版本支持不一定一致 | 支持 Python 3.8+，可选依赖兼容性以实际环境为准 |

### 5.2 推荐定位

`hscredit` 的竞争策略应围绕以下方向展开：

| 方向 | 说明 |
|:---|:---|
| 一体化 | 将 EDA、分箱、编码、筛选、建模、评估、策略和报告放在统一 API 下 |
| 中文化 | 面向国内风控团队的字段、报表、错误提示和业务解释 |
| 低依赖 | 基础安装轻量，增强能力通过 extras 按需安装 |
| Pipeline | 保持 sklearn 风格，便于与现有建模流水线集成 |
| 场景化 | 聚焦信贷评分卡、策略规则、逾期预测、稳定性监控和 Excel 交付 |
| 差异化 | 特征工程、拒绝推断、规则运营、反事实解释和模型上线交付 |

## 六、能力缺口与优先级

| 优先级 | 能力缺口 | 价值 | 参考 |
|:---|:---|:---|:---|
| P0 | 文档、测试、CI、示例和 README 真实性修正 | 建立项目可信度 | 所有成熟开源库 |
| P0 | 特征工程模块扩展 | 三方竞品普遍薄弱，适合形成差异化 | 内部业务实践 |
| P0 | 拒绝推断 | 处理审批样本选择偏差，是信贷建模强场景能力 | 信贷建模实践 |
| P1 | 分箱质量分析和批量导出 | 提升分箱专业度和交付效率 | optbinning |
| P1 | 规则运营工具 | 支持策略人员做规则覆盖、冲突、跨期追踪 | 策略运营实践 |
| P1 | 报告模板体系 | 提升模型报告、特征报告、策略报告可交付性 | scorecardpipeline |
| P2 | 二维分箱文档、示例和测试 | 捕捉变量交互效应 | optbinning |
| P2 | SHAP 与反事实解释 | 支持监管、可解释性和拒绝原因说明 | optbinning / XAI 实践 |
| P2 | PMML / SQL / Python / Java 导出完善 | 支持模型上线交付闭环 | scorecardpipeline |
| P3 | 快捷入口 API | 降低新用户学习成本 | toad |

## 七、版本规划

### v0.1.x：可信度建设

目标：修正文档、补充测试、统一口径，让现有能力能被稳定理解和复现。

任务：

1. 重写 README，突出真实代码能力和业务场景。
2. 更新 ROADMAP，形成清晰竞品策略和版本规划。
3. 修正 `hscredit.info()` 中过期描述。
4. 补充核心模块示例：EDA、分箱、WOE、筛选、ScoreCard、规则挖掘、Swap、模型报告。
5. 建立 CI：Python 3.8 到当前主流版本的测试矩阵。
6. 补充 `core.metrics`、`core.eda`、`report.mining`、`ScoreCard`、`OptimalBinning2D` 的基础测试。
7. 完善打包校验：`python -m build`、`twine check`、`check-manifest`。

验收标准：

1. README 中示例可运行或标注依赖前提。
2. `make check` 在主开发环境通过。
3. 核心公开 API 有最小测试覆盖。
4. 文档不再包含明显夸大或已过期内容。

### v0.2.0：特征工程与拒绝推断

目标：补齐信贷建模前置处理和样本选择偏差处理能力，形成差异化。

计划新增模块：

```text
hscredit/core/feature_engineering/
├── time_features.py
├── cross_features.py
├── preprocessing.py
└── aggregations.py

hscredit/core/reject_inference/
├── __init__.py
└── reject_inference.py
```

建议能力：

| 类/函数 | 说明 |
|:---|:---|
| `TimeFeatureGenerator` | 从日期字段生成月份、星期、季度、月末、账龄、距参考日期天数等变量 |
| `CrossFeatureGenerator` | 生成比值、差值、乘积、log ratio 等交叉变量 |
| `MissingValueImputer` | 数值/类别缺失填充，保留 DataFrame 列名 |
| `OutlierClipper` | 分位数、IQR、固定边界异常值截断 |
| `FeatureScaler` | 标准化、归一化、RobustScaler 封装 |
| `GroupAggregationTransformer` | 按客户、设备、手机号、商户等实体聚合行为变量 |
| `RejectInference` | hard cutoff、fuzzy augmentation、parceling、twin 方法 |

验收标准：

1. 所有 Transformer 兼容 sklearn Pipeline。
2. 支持 DataFrame 输入输出并保留列名。
3. 拒绝推断提供模拟数据验证和业务说明。
4. 文档说明每种方法适用前提和风险。

### v0.3.0：分箱与规则运营增强

目标：提升策略和模型团队最常用的分箱、规则分析、Excel 交付效率。

计划能力：

| 能力 | 说明 |
|:---|:---|
| `get_quality_report()` | 输出 IV、KS、Gini、HHI、单调性、样本占比、综合评分 |
| `batch_to_excel()` | 批量分箱汇总、单变量分箱表、图表和质量报告输出 |
| `BestPSIBinning` | 在训练/验证集稳定性约束下选择分箱 |
| `auto_select_bins()` | 根据样本量、唯一值数和监督指标推荐箱数 |
| `ruleset_overlap_matrix()` | 规则覆盖重叠矩阵 |
| `rule_effectiveness_tracking()` | 单条规则跨期覆盖率、坏率、Lift 追踪 |
| `detect_rule_conflicts()` | 规则冲突、包含关系、高重叠检测 |
| `rule_strategy_simulation()` | 规则集审批率、坏率、通过人群变化仿真 |

验收标准：

1. 批量分箱 Excel 能直接用于业务评审。
2. 规则分析支持单期和跨期报告。
3. 每个新增函数有至少一个真实风控语义示例。

### v0.4.0：报告、解释与上线交付

目标：将建模结果转换为更完整的解释报告和上线材料。

计划能力：

| 能力 | 说明 |
|:---|:---|
| SHAP 报告 Sheet | 模型解释汇总、Top 特征贡献、依赖图导出 |
| `ScoreCard.score_segment_analysis()` | 分数段客群特征分布和坏率对比 |
| `compare_models()` 增强 | 多模型 KS、AUC、Lift、分数分布、校准曲线对比 |
| `CounterfactualExplainer` | 输出最小特征变化方案和拒绝原因解释 |
| `scorecard2sql()` | 评分卡 SQL 部署代码导出 |
| `scorecard2python()` | Python 评分函数导出 |
| `scorecard2pmml()` | PMML 导出能力完善 |

验收标准：

1. 模型报告可覆盖评分卡和树模型两类主流模型。
2. 导出代码有单元测试验证预测一致性。
3. 解释性输出明确标注假设、限制和不可变特征。

### v0.5.0：易用性和生态完善

目标：降低新用户学习成本，形成稳定 API 和教程体系。

计划能力：

| 能力 | 说明 |
|:---|:---|
| `hscredit.detect()` | 快速数据画像入口 |
| `hscredit.quality()` | 快速变量质量评估入口 |
| `hscredit.select()` | 多条件特征筛选入口 |
| `hscredit.scorecard_pipeline()` | 标准评分卡 Pipeline 快速构建 |
| 示例项目 | 贷前评分卡、贷后 Vintage、策略规则、模型监控完整示例 |
| API 文档 | Sphinx 自动文档和 Notebook 教程 |

## 八、工程质量计划

### 8.1 测试优先级

| 测试目录 | 重点 |
|:---|:---|
| `tests/test_binning/` | 各分箱器 fit/transform、边界值、特殊值、单调性、DataFrame 输入 |
| `tests/test_encoders/` | WOE、Target、Count、类别编码器输出一致性 |
| `tests/test_selectors/` | 每个筛选器选择结果、报告字段、Pipeline 兼容性 |
| `tests/test_models/` | LR、ScoreCard、Boosting 可选依赖、损失函数、校准 |
| `tests/test_metrics/` | KS、AUC、Lift、IV、PSI、CSI 的数值正确性 |
| `tests/test_eda/` | 数据概览、策略分析、Vintage、稳定性报告 |
| `tests/test_report/` | 模型报告、规则报告、Swap、Excel 输出 |
| `tests/test_utils/` | pandas 扩展、IO、随机种子、输入校验 |

### 8.2 CI 与兼容性

建议 CI 矩阵：

| 环境 | 说明 |
|:---|:---|
| Python 3.8 | 最低支持版本 |
| Python 3.9 / 3.10 / 3.11 / 3.12 | 主流生产环境 |
| Python 3.13 / 3.14 | 跟进验证，特别注意可选依赖支持情况 |

CI 阶段：

1. `pip install -e ".[dev]"`
2. `make lint`
3. `make type-check`
4. `pytest tests/ -m "not slow and not integration"`
5. `python -m build`
6. `python -m twine check dist/*`

### 8.3 文档质量要求

1. README 只描述已经实现或明确标注“规划中”的能力。
2. 示例代码必须与当前公开 API 保持一致。
3. 每个核心模块至少提供一个最小可运行示例。
4. 竞品对比避免“全面超越”等绝对化表述。
5. 涉及监管、拒绝原因、反事实解释等内容必须说明假设和限制。

## 九、当前短期任务清单

| 优先级 | 任务 | 产出 |
|:---|:---|:---|
| P0 | 修正 `hscredit.info()` 过期内容 | 顶层信息与当前模块一致 |
| P0 | 校验 README 示例 | 可运行示例或明确依赖条件 |
| P0 | 补充 `OptimalBinning2D` 文档 | 示例、参数说明、测试 |
| P0 | 增加核心 API smoke tests | 导入、fit/transform、报告生成 |
| P1 | 建立 GitHub Actions | lint、type-check、test、build |
| P1 | 完善模型报告示例 | 评分卡、树模型各一个端到端示例 |
| P1 | 增加规则挖掘 notebook | 单特征、多特征、树规则、手工树 |
| P1 | 分箱质量报告设计 | 指标定义、输出格式、Excel 模板 |
| P2 | 拒绝推断设计文档 | 方法说明、适用前提、风险提示 |
| P2 | 特征工程模块接口设计 | Transformer API 和命名规范 |

## 十、长期方向

`hscredit` 的长期方向是成为面向信贷风控团队的场景化建模基础设施，而不是只提供若干算法函数。后续建设重点应落在：

1. 稳定、可测试、可持续发布的工程体系。
2. 从变量到模型到策略到报告的统一数据结构和审计轨迹。
3. 贴近业务术语的中文报告和可解释输出。
4. 面向审批偏差、客群迁移、规则衰减、收益约束的风控专用算法。
5. 支持模型上线、策略迭代和贷后监控的交付闭环。
