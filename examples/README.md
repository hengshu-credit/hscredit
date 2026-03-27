# hscredit Examples

完整示例目录，覆盖所有模块和 API 方法。

## 快速开始

```bash
pip install hscredit
jupyter notebook examples/
```

## 示例目录

| 文件 | 模块 | 主要内容 |
|------|------|----------|
| [01_binning.ipynb](01_binning.ipynb) | `core.binning` | 16种分箱算法、自定义切分点、单调约束、批量分箱、WOE/indices/labels转换 |
| [02_encoders.ipynb](02_encoders.ipynb) | `core.encoders` | WOE / Target / Count / OneHot / Ordinal / Quantile 编码器 |
| [03_selectors.ipynb](03_selectors.ipynb) | `core.selectors` | 缺失率/方差/众数/IV/相关性/VIF/PSI/特征重要性筛选 + CompositeFeatureSelector |
| [04_models.ipynb](04_models.ipynb) | `core.models` | LogisticRegression（含统计量）/ ScoreCard / 自定义损失函数 / 概率转评分 |
| [05_rules.ipynb](05_rules.ipynb) | `core.rules` | Rule对象 / 规则挖掘 / 表达式优化 / ruleset_report |
| [06_viz.ipynb](06_viz.ipynb) | `core.viz` | bin_plot / corr_plot / ks_plot / hist_plot / psi_plot / roc_plot / score_plots / strategy_plots / variable_plots |
| [07_metrics.ipynb](07_metrics.ipynb) | `core.metrics` | KS/AUC/Gini / IV/WOE / PSI/CSI/batch_psi / Lift / 回归指标 / 大写别名兼容 |
| [08_eda.ipynb](08_eda.ipynb) | `core.eda` | 数据概览 / 目标分析 / IV/WOE / 相关性 / 稳定性 / Vintage / eda_summary |
| [09_complete_workflow.ipynb](09_complete_workflow.ipynb) | 全链路 | EDA → 分箱 → WOE → 特征筛选 → LR → 评分卡 → PSI监控 |
| [10_report.ipynb](10_report.ipynb) | `report` | ExcelWriter / FeatureAnalyzer / auto_model_report / compare_models / ruleset_report / SwapAnalyzer |
| [11_financial_utils.ipynb](11_financial_utils.ipynb) | `core.financial` + `utils` | FV/PV/PMT/NPER/IRR/NPV/MIRR / pickle工具 / pandas扩展 / NumExprDerive |
| [12_strategy_analysis.ipynb](12_strategy_analysis.ipynb) | 策略分析 | feature_trend_by_time / feature_drift_comparison / population_drift_monitor / psi_cross_analysis |
| [13_score_drift_calibration.ipynb](13_score_drift_calibration.ipynb) | 评分监控 | ScoreDriftDetector / ModelCalibrator / OverduePredictor |
| [14_pipeline_sklearn.ipynb](14_pipeline_sklearn.ipynb) | Pipeline集成 | Pipeline / ColumnTransformer / VotingClassifier / StackingClassifier / 交叉验证 |

## 模块覆盖矩阵

| 模块 | 示例文件 |
|------|----------|
| `hscredit.core.binning` | 01 |
| `hscredit.core.encoders` | 02, 09, 14 |
| `hscredit.core.selectors` | 03, 09, 14 |
| `hscredit.core.models` | 04, 09, 13 |
| `hscredit.core.rules` | 05, 10 |
| `hscredit.core.viz` | 06, 09 |
| `hscredit.core.metrics` | 07, 09 |
| `hscredit.core.eda` | 08, 09, 12 |
| `hscredit.core.financial` | 11 |
| `hscredit.core.feature_engineering` | 11 |
| `hscredit.report` | 10 |
| `hscredit.utils` | 11 |
| sklearn Pipeline集成 | 14 |
| [15_tree_models.ipynb](15_tree_models.ipynb) | 树模型/集成模型 | RandomForest / ExtraTrees / GradientBoosting / XGBoost / LightGBM + ModelReport + ModelExplainer(SHAP) |
| [16_advanced_selectors_tuning.ipynb](16_advanced_selectors_tuning.ipynb) | 高级筛选+调优 | NullImportanceSelector / RFESelector / SFS / Chi2 / FTest / MutualInfo / BorutaSelector / StepwiseSelector / RulesClassifier / ModelTuner / AutoTuner |
| [17_advanced_binning_viz.ipynb](17_advanced_binning_viz.ipynb) | 高级分箱+可视化 | GeneticBinning / KernelDensityBinning / ORBinning / GBMEncoder / CatBoostEncoder / bin_trend_plot / batch_bin_trend_plot / overdues_bin_plot / risk_plots |
