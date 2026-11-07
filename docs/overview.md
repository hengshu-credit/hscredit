# 功能总览

hscredit 面向信贷风控的策略分析和模型开发。可以从一批放款订单出发，定义逾期标签、检查变量覆盖率、训练评分卡或规则树，再比较候选策略对通过率、坏账率和放款金额的影响。[功能与示例](showcase.md)提供代码、结果表和图形，[API 参考](api/index.rst)列出各模块的参数、返回值和方法。

## 数据与样本

| 要做的事 | 入口与结果 |
| --- | --- |
| 读取样例数据 | `hscredit_dataset()` 离线读取内置现金贷订单。`columns` 按需读取字段，`split=True` 分开返回订单信息、贷后表现和授信特征。可在线浏览全部授信特征的分布及两个标签下的 IV、KS、AUC。`hscredit.utils.datasets` 还提供 German Credit、Titanic 和 Iris 数据入口。[数据集说明](datasets.md) · [特征摘要报告](dataset-summary.md) |
| 定义好坏样本 | 支持标签列、外部标签、标签函数，以及由 `overdue`、`dpds` 和比较符组合生成的多个逾期标签。比较策略时可统一灰样本处理、观察日期、客群和金额口径。[标签与灰样本](overdue_labels.md) · [分析参数](analysis_parameters.md) |
| 分训练集与测试集 | `time_train_test_split` 按时间顺序切分，`oot_split` 指定时间外验证窗口，`group_train_test_split` 按客户等分组隔离样本。[样本切分 API](api/model_selection.rst) |
| 接入业务数据 | `Database` 支持 MySQL、Oracle、Hive、Impala、StarRocks、ClickHouse、MaxCompute、Redis 和 MongoDB。可按适配器能力使用连接池、参数化 SQL、流式读写、元数据查询或 NoSQL 操作。[数据库接入](database.md) |

例如，用内置数据分析两个授信特征与 `MOB6 > 30` 的关系。下面只读取必要字段，使用逾期天数有记录且非负的订单，同一客户的订单划入同一组。

```python
from hscredit import hscredit_dataset
from hscredit.core.model_selection import group_train_test_split

orders, overdue, X = hscredit_dataset(
    split=True, columns=["用户编码", "MOB6", "age_year", "LZ_SCORE"]
)
days = overdue["MOB6"]
valid = days.notna() & days.ge(0)
X_train, X_test, y_train, y_test = group_train_test_split(
    X.loc[valid], days.loc[valid].gt(30).astype(int),
    groups=orders.loc[valid, "用户编码"], test_size=0.2, random_state=42,
)
summary = X_train.summary(y=y_train, fillna=-99999)
summary[["特征名", "样本数", "缺失率", "IV", "KS", "AUC"]]
```

`fillna` 只决定 KS 和 AUC 如何处理特征缺失值。默认使用特征和标签都有值的样本。IV 和趋势使用原始特征分箱，缺失率及分位数也按原始特征统计。

## 探索与变量工程

| 环节 | 可以得到什么 | 主要入口 |
| --- | --- | --- |
| 数据探索 | 查缺失、异常、稀有类别、目标泄漏和多标签相关性。按月份或客群比较变量有效性，用 Vintage 看账龄表现，用 Roll Rate 看逾期状态迁徙。 | `feature_summary`、`data_quality_report`、`vintage_analysis`、`roll_rate_analysis`。[数据探索 API](api/eda.rst) |
| 特征衍生 | 用命名表达式批量生成数值变量，可嵌入训练和推理的 Pipeline。 | `NumExprDerive`。[特征衍生 API](api/feature_engineering.rst) |
| 一维与二维分箱 | 支持 17 种一维方法，包括等频、等宽、卡方、树、单调、最优 IV/KS/Lift 和约束求解。可指定切点、类别组合或自定义目标，并对两项特征做交叉分箱。 | `OptimalBinning`、`OptimalBinning2D`、`CustomObjectives`。[分箱 API](api/binning.rst) |
| 分箱效果比较 | 比较多种分箱方法，按月、客群或多种逾期标签检查各箱的样本量、坏率和 Lift。二维交叉表用于观察两个变量的组合风险。 | `feature_binning_summary`、`feature_group_binning_summary`、`feature_bin_stats_2d`。[报告 API](api/report.rst) |
| 特征编码 | 将分箱或类别转为 WOE、目标均值、频数、独热、序数或分位数编码，也可用有序目标编码、GBM 叶节点和高基数编码。 | `WOEEncoder`、`TargetEncoder` 等 9 种编码器。[编码 API](api/encoders.rst) |
| 特征筛选 | 先按缺失、单一值、相关性、IV、KS、Lift、PSI 过滤，再用逐步选择、RFE、Boruta 或 Null Importance 筛选。可串联多个筛选器并追踪每个字段的保留或剔除原因。 | `CompositeFeatureSelector`、`ScorecardFeatureSelection`、`SelectionReportCollector`。[筛选 API](api/selectors.rst) |

## 模型训练与调参

- **经典模型与提升树**：逻辑回归支持系数与显著性分析。随机森林、极端随机树、梯度提升、SVM 和决策树用于构建对照模型。Boosting 可选 XGBoost、LightGBM、CatBoost 和 NGBoost。[经典模型](api/classical_models.rst) · [提升模型](api/boosting.rst)
- **表格神经网络**：提供 ResNet、FTTransformer、TabM、TabPFN 和 TabNet 五个模型入口。可按模型能力选择无标签预训练、微调或提取 embedding。已有提升树可按对应后端的续训接口继续训练。[表格模型](api/net.rst) · [预训练与续训](api/training_lifecycle.rst)
- **训练损失与评估指标**：支持分类误差、排序、头部坏账捕获、金额加权、审批率与利润目标。用 `list_losses()`、`list_metrics()` 查看注册项，用 `BaseLoss`、`BaseMetric` 或指标函数扩展训练与评估。[损失与评估](articles/losses-guide.md)
- **超参数搜索**：模型的 `.tune()`、`ModelTuner` 和 `AutoTuner` 支持单次验证、交叉验证、剪枝及多目标搜索。可查看逐折结果、折外预测（OOF）、学习曲线和 Pareto 前沿，通过 `retention` 控制模型与预测的保留范围。[训练工作流](articles/model-workflow.md) · [调参指南](articles/tuning-guide.md)

## 评分、解释与部署

| 任务 | 入口与输出 |
| --- | --- |
| 生成信用分 | `ScoreCard` 将 WOE 和逻辑回归系数转成分值表，`RoundScoreCard` 按分箱分数精度计分，`ProbabilityScoreCard` 将概率模型输出转为评分。可选择标准、线性、分位数或 Box-Cox 转分，并校准评分漂移。[评分卡 API](api/scorecard.rst) |
| 校准预测概率 | `ProbabilityCalibrator` 支持 Platt、Isotonic、Beta 和 Histogram 方法，用独立验证数据拟合概率映射，再比较校准效果。[概率校准 API](api/calibration.rst) |
| 解释模型与单笔决策 | `ModelExplainer` 提供全局重要性和单样本 SHAP 贡献，可生成原因码。`CounterfactualExplainer` 支持反事实分析。[解释工作流](articles/model-interpretability.md) · [解释 API](api/explainability.rst) |
| 保存与部署 | 例如，`RandomForest` 可保存完整模型，或用 `save_inference` 导出推理制品。模型的 `save_deployment` 接口可记录依赖、前处理信息并用验证样本核对预测。`ScoreCard` 还可导出 PMML 或 Python、SQL、Java 计分代码。[部署与资源管理](articles/report-resources-deployment.md) · [评分卡导出](api/scorecard.rst) |

## 规则与策略

- **执行规则**：`Rule` 用表达式生成命中结果，`RuleSet` 组合多条规则，`RuleFlow` 编排执行流程。`RulesClassifier` 将规则组合封装为可调用 `fit`、`predict` 的分类器。[规则引擎](api/rules.rst) · [规则模型](api/model_rules.rst)
- **挖掘规则**：支持单特征、多特征、多标签规则挖掘，以及自动决策树、手工拆分和多叉树。可对非根节点拆分、剪枝，从叶节点取规则，在训练集和测试集比较覆盖率、坏样本率与 Lift。[规则挖掘 API](api/mining.rst) · [手工树](articles/manual_tree_rule_mining.md) · [多叉树](multiway_tree.rst)
- **评估策略变更**：`ruleset_analysis` 评估规则组合，`swap_analysis` 比较新旧策略的置入、置出样本。固定申请母集后，可按标签、月份、客群和金额比较通过率与风险。[规则置换口径](rule_swap_approval.md)

## 评估、监控与报告

按交付物选择报告入口。下列四类报告均有实际结果表和 Excel 样例。{ref}`查看四类报告示例 <gallery-reports>`

| 交付物 | 主要入口 | 报告回答的问题 |
| --- | --- | --- |
| 三方数据分析报告 | `auto_feature_analysis` | 哪些变量有覆盖、有区分度？从数据摘要、分箱明细、KS 曲线和分布图比较变量在多个逾期标签下的表现。 |
| 模型上线报告 | `auto_model_report`、`ModelReport` | 分类模型在各样本集上的 KS、AUC 和分数段坏率如何？回归模型的误差如何？可结合训练过程、重要性、SHAP 和样本明细复核。 |
| 规则挖掘报告 | 已拟合规则树的 `.report()` | 哪些节点值得保留？同一组节点对训练、测试样本及不同风险口径的覆盖与效果如何？ |
| 策略上线报告 | `swap_out_report` | 调整规则会放行或拦截多少订单、多少金额？新旧策略的命中、风险和分群稳定性如何变化？ |

还可以单独调用以下组件：

- **指标与图表**：分类和回归指标用于核对区分度与误差，PSI、CSI 用于跟踪客群或变量分布变化。图表覆盖分箱与跨期趋势、ROC/KS、Lift、校准、审批率与坏率、规则置换和树图。[指标 API](api/metrics.rst) · [可视化 API](api/viz.rst)
- **监控与推算**：`MonitoringBaseline` 保存监控基准，`BinStatsAccumulator` 按块汇总分箱统计。`OverduePredictor` 可用有标签样本的分箱逾期率，对无标签客群推算加权逾期率。[报告 API](api/report.rst) · [流式汇总](articles/report-resources-deployment.md)
- **金融计算**：现值、终值、分期还款额、利息与本金拆分、净现值和内部收益率可由 `pv`、`fv`、`pmt`、`ipmt`、`ppmt`、`npv`、`irr` 等函数计算。[金融计算 API](api/financial.rst)

## Excel、公共工具与 Agent Skills

- **Excel 交付**：`dataframe2excel` 写入数据和多级表头，`ExcelWriter` 编排多工作表、图片、条件格式、筛选与冻结窗格，也能生成原生迷你图和透视表。[Excel API](api/excel.rst)
- **pandas 与并行计算**：DataFrame 和 Series 可用 `.summary()` 描述字段、用 `.save()` 导出。DataFrame 还可用 `.show()` 格式化展示结果表。`.hscredit(n_jobs=...).apply(...)` 支持 DataFrame、Series 及其分组上的并行计算。[并行指南](parallelism.md) · [公共工具 API](api/utils.rst)
- **日常分析工具**：`feature_describe` 与 `groupby_feature_describe` 做单字段和分组描述，`seed_everything` 固定随机种子，`reload` 在 Notebook 中重载模块，`reduce_memory_usage` 压缩 DataFrame 内存，`save_pickle` / `load_pickle` 保存和读取中间结果。{ref}`公共方法示例 <public-utils>`
- **Agent Skills**：可通过 `hscredit` Skill 发现和路由操作，由 `hsbin`、`hsselect`、`hsmodel`、`hsrule`、`hsreport`、`hsviz`、`hsexcel` 分别完成分箱、筛选、建模、规则、报告、绘图和 Excel 导出。[Skills 使用指南](skills.md) · [运行接口](api/skills_runtime.rst)

完整建模流程见{ref}`功能与示例中的风控建模 <gallery-scorecard>`。Boosting、神经网络、调参、PMML 和数据库驱动可按需安装对应的扩展依赖。[安装指南](installation.md)
