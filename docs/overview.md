# 功能总览

hscredit 覆盖数据探索、变量工程、模型训练、规则分析与报告交付。分箱、编码、筛选和学习模型兼容 sklearn；报告、结果容器与工具函数提供独立入口。

## 变量工程

- **数据探索 `hscredit.core.eda`**：58 个公开分析函数，覆盖数据质量、目标分布、变量有效性、稳定性、客群、Vintage、Roll Rate 与报告。
- **分箱 `hscredit.core.binning`**：17 种一维方法与独立的 OptimalBinning2D。支持等频、等宽、卡方、树、CART、最优 IV/KS/Lift、MDLP、OR-Tools、CP-SAT、KMeans、单调、遗传、平滑、核密度与目标坏率。
- **编码 `hscredit.core.encoders`**：9 个编码器：WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、GBM、Cardinality。
- **筛选 `hscredit.core.selectors`**：23 个筛选器与 1 个 CompositeFeatureSelector，覆盖类型、正则、缺失率、众数率、基数、方差、相关性、VIF、IV/KS/Lift/PSI、模型重要性、RFE、逐步回归、Boruta 等。
- **衍生 `hscredit.core.feature_engineering`**：1 个 NumExprDerive 表达式衍生器。

[分箱示例](../examples/01_binning.ipynb) · [编码示例](../examples/02_encoders.ipynb) · [筛选示例](../examples/03_selectors.ipynb) · [EDA 示例](../examples/06_eda.ipynb)

## 模型训练

- **经典模型 `hscredit.core.models.classical`**：6 个模型：LogisticRegression、RandomForest、ExtraTrees、GradientBoosting、SVM、DecisionTreeClassifier。
- **提升模型 `hscredit.core.models.boosting`**：4 个模型：XGBoost、LightGBM、CatBoost、NGBoost。
- **神经网络 `hscredit.core.models.net`**：4 个模型：ResNet、FTTransformer、TabM、TabPFN。
- **评分卡 `hscredit.core.models.scorecard`**：3 个评分卡（ScoreCard、RoundScoreCard、ProbabilityScoreCard）、4 个转换器（Standard、Linear、Quantile、BoxCox）与 1 个 ScoreDriftCalibrator。
- **概率校准 `hscredit.core.models.calibration`**：4 种方法：Platt、Isotonic、Beta、Histogram；另有统一入口与模型包装器。
- **损失指标 `hscredit.core.models.losses`**：16 种注册损失与 12 种注册指标，覆盖区分度、成本、利润、审批率、坏账与排序。
- **参数搜索 `hscredit.core.models.tuning`**：2 个调参器 ModelTuner、AutoTuner，支持交叉验证、剪枝、单/多目标、Pareto 与持久化。
- **模型解释 `hscredit.core.models.explainability`**：14 个公开函数与 3 个组件 ModelExplainer、ExplanationResult、CounterfactualExplainer，覆盖 SHAP、原因码、反事实与解释报告。

[模型流程](../examples/28_model_workflow.ipynb) · [神经网络](../examples/net_models.ipynb) · [迁移续训](../examples/pretraining_and_continuation.ipynb) · [参数搜索](../examples/tuning_workflow.ipynb)

## 评估与策略

- **指标 `hscredit.core.metrics`**：43 个公开函数，以及 MonitoringBaseline、BinStatsAccumulator、MetricSpec 共 3 个组件。
- **可视化 `hscredit.core.viz`**：52 个绘图函数与 1 个 DecisionTreeViz 组件，覆盖分箱、模型、评分、策略、漂移与决策树。
- **规则引擎 `hscredit.core.rules`**：2 个规则执行组件 Rule、RuleFlow；另有规则制品、累加器和表达式工具。
- **规则挖掘 `hscredit.report.mining`**：6 个挖掘器 SingleFeatureRuleMiner、MultiFeatureRuleMiner、MultiLabelRuleMiner、TreeRuleExtractor、ManualTreeExtractor、MultiwayTreeExtractor，以及 1 个 DecisionTreeAnalyzer。
- **金融计算 `hscredit.core.financial`**：10 个函数 fv、pv、pmt、nper、ipmt、ppmt、rate、npv、irr、mirr。

[指标示例](../examples/05_metrics.ipynb) · [可视化示例](../examples/07_viz.ipynb) · [规则置换](../examples/20_rule_swap_analysis.ipynb) · [手工树](../examples/25_manual_tree_rule_mining.ipynb)

## 数据与报告

- **报告 `hscredit.report`**：24 个报告及辅助函数，覆盖特征、模型、规则、策略置换、逾期预测、筛选读写与客群漂移；另有 ModelReport、SwapAnalyzer 等报告类与结果容器。
- **Excel `hscredit.excel`**：1 个 ExcelWriter 与 3 个工具函数 dataframe2excel、resolve_condition_color、register_pivot_aggregation，支持多 Sheet、图片、格式与透视表。
- **数据库 `hscredit.database`**：9 类适配器：ClickHouse、Hive、Impala、MaxCompute、MongoDB、MySQL、Oracle、Redis、StarRocks。支持连接池、流式查询、批量写入与元数据。
- **Pandas 扩展 `hscredit.utils`**：summary()、show()、save() 等入口，供 Notebook 展示与 Excel 交付使用。

[报告示例](../examples/11_report.ipynb) · [Excel 示例](../examples/23_excel_writer.ipynb) · [数据库指南](database.md)

## 统计口径

以上数量于 **2026-10-30** 对照源码核实，不是测试数量或覆盖率。各项单位不同，重导出的能力不跨模块相加。

- EDA、指标与模型解释按各模块 `__all__` 的公开函数与类分别计数。
- 编码、筛选、模型、评分卡与校准按具体实现计数，排除基类、容器与兼容别名；筛选报告收集器不算筛选器。
- 分箱按 `OptimalBinning.VALID_METHODS` 计数，二维分箱单列；工厂与基类不算算法。
- 损失与模型指标按 `list_losses()`、`list_metrics()` 的注册条目计数，包含 logloss、brier 两个函数适配指标。
- 绘图按 `core.viz.__all__` 的绘图区域计数，排除样式、颜色、图例与保存工具，保留独立树图后端入口。
- 报告函数按 `report.__all__` 计数，排除重导出的 dataframe2excel 与 calculate_rule_metrics；报告类不计入函数数量。
- 数据库按 `available_adapters()` 的内置适配器类型计数，不含用户注册的扩展。

## 调用方式

有监督组件支持 sklearn 风格，也支持通过 target 指定标签列：

```python
from hscredit import OptimalBinning

binner = OptimalBinning(method="quantile")
binner.fit(X_train, y_train)

binner = OptimalBinning(method="quantile", target="FPD")
binner.fit(train_df)
```

完整参数、返回值和成员方法见 [API 参考](api/index.rst)。
