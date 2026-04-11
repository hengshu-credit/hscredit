"""hscredit - 金融信贷风险策略和模型开发库.

一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。
"""

__version__ = "0.1.0"
__author__ = "hscredit team"
__email__ = "hscredit@hengshucredit.com"

from .exceptions import (
    HSCreditError,
    ValidationError,
    InputValidationError,
    InputTypeError,
    FeatureNotFoundError,
    StateError,
    NotFittedError,
    DependencyError,
    SerializationError,
)


# ========== sklearn Pipeline 和集成学习组件 ==========
# 为了方便用户，直接从hscredit导入sklearn的Pipeline相关组件

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer

# ========== 核心模块导入 (先导入core，避免循环导入) ==========

# 核心分箱模块
from .core.binning import (
    BaseBinning,
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    ORBinning,
    CustomObjectives,
    OptimalBinning,
    CartBinning,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
)

# 核心编码器模块
from .core.encoders import (
    BaseEncoder,
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    GBMEncoder,
    CardinalityEncoder,
)

# 核心特征筛选模块
from .core.selectors import (
    BaseFeatureSelector,
    SelectionReportCollector,
    VarianceSelector,
    NullSelector,
    ModeSelector,
    CorrSelector,
    VIFSelector,
    IVSelector,
    LiftSelector,
    PSISelector,
    CardinalitySelector,
    TypeSelector,
    RegexSelector,
    FeatureImportanceSelector,
    NullImportanceSelector,
    RFESelector,
    SequentialFeatureSelector,
    BorutaSelector,
    MutualInfoSelector,
    Chi2Selector,
    FTestSelector,
    CompositeFeatureSelector,
    StepwiseSelector,
    StabilityAwareSelector,
)

# 核心模型/损失函数模块
from .core.models import (
    BaseLoss,
    BaseMetric,
    FocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    KSMetric,
    GiniMetric,
    PSIMetric,
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
    LogisticRegression,
    ScoreCard,
    # 规则集分类
    RuleSet,
    RulesClassifier,
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules,
)

# 模型类单独导入（避免可选依赖失败影响整体）
try:
    from .core.models import (
        BaseRiskModel,
        XGBoostRiskModel,
        LightGBMRiskModel,
        CatBoostRiskModel,
        RandomForestRiskModel,
        ExtraTreesRiskModel,
        GradientBoostingRiskModel,
        ModelReport,
    )
except ImportError:
    pass

try:
    from .core.models import ModelTuner, AutoTuner, TuningObjective
except ImportError:
    ModelTuner = None
    AutoTuner = None
    TuningObjective = None

# 核心可视化模块
from .core.viz import (
    # 基础图表
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
    plot_weights,
    # 特征趋势分析
    bin_trend_plot,
    batch_bin_trend_plot,
    bin_overdues_plot,
    # 模型评估
    roc_plot,
    pr_plot,
    lift_plot,
    gain_plot,
    confusion_matrix_plot,
    calibration_plot,
    # 评分卡
    score_dist_plot,
    score_bin_plot,
    # 风控策略
    threshold_analysis_plot,
    strategy_compare_plot,
    vintage_plot,
    feature_importance_plot,
    approval_rate_trend_plot,
    bad_rate_trend_plot,
    # 新增：变量分析图表
    variable_iv_plot,
    variable_woe_trend_plot,
    variable_psi_heatmap,
    variable_importance_grouped_plot,
    variable_missing_badrate_plot,
    # 新增：评分分析图表
    score_ks_plot,
    score_distribution_comparison_plot,
    score_badrate_bin_plot,
    score_lift_plot,
    score_approval_badrate_curve,
    # 新增：策略分析图表
    feature_trend_by_time,
    feature_drift_comparison,
    feature_effectiveness_by_segment,
    feature_cross_heatmap,
    population_drift_monitor,
    segment_scorecard_comparison,
)

# 核心特征工程模块
from .core.feature_engineering import NumExprDerive

# 核心规则引擎模块
from .core.rules import (
    Rule,
    get_columns_from_query,
    optimize_expr,
    beautify_expr,
    get_expr_variables,
)

# 核心金融计算模块
from .core.financial import (
    fv, pv, pmt, nper, ipmt, ppmt, rate,
    npv, irr, mirr,
)

# 核心EDA模块（函数式API）
from .core import eda
from .core.eda import (
    # 数据概览
    data_info,
    missing_analysis,
    feature_summary,
    numeric_summary,
    category_summary,
    data_quality_report,
    feature_group_analysis,
    population_stability_monitor,
    # 目标变量分析
    target_distribution,
    bad_rate_overall,
    bad_rate_by_dimension,
    bad_rate_trend,
    bad_rate_by_bins,
    sample_distribution,
    # 特征分析
    numeric_distribution,
    categorical_distribution,
    outlier_detection,
    # 特征标签关系
    iv_analysis,
    batch_iv_analysis,
    woe_analysis,
    binning_bad_rate,
    # 相关性分析
    correlation_matrix,
    high_correlation_pairs,
    vif_analysis,
    # 稳定性分析
    psi_analysis,
    batch_psi_analysis,
    csi_analysis,
    # Vintage分析
    vintage_analysis,
    # 综合报告
    eda_summary,
)

# 核心指标计算模块
from .core.metrics import (
    ks, auc, gini, iv, psi, csi,
    iv_table, psi_table, csi_table, ks_bucket, roc_curve,
    lift, lift_at, lift_table, lift_curve, lift_monotonicity_check, rule_lift,
    badrate, badrate_by_group, badrate_trend, badrate_by_score_bin,
    score_stats, score_stability,
    mse, mae, rmse, r2,
)

# ========== 报告模块导入 (在core之后导入，避免循环导入) ==========

from .report.excel import ExcelWriter, dataframe2excel
from .report.feature_analyzer import feature_bin_stats, FeatureAnalyzer
from .report.ruleset_report import ruleset_report
from .report.swap_analysis_report import (
    ReferenceDataProvider,
    SwapAnalyzer,
    SwapAnalysisResult,
    SwapRiskConfig,
    create_swap_dataset,
    create_swap_dataset_from_rules,
    swap_analysis_report,
    SwapType,
)

from .report.overdue_predictor import OverduePredictor, overdue_prediction_report
from .report.model_report import QuickModelReport, auto_model_report, compare_models
from .report.mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    MultiLabelRuleMiner,
    TreeRuleExtractor,
    RuleMetrics,
    calculate_rule_metrics,
    TreeVisualizer,
    plot_decision_tree,
)

try:
    from .report.feature_report import auto_feature_analysis_report
except ImportError:
    pass

# ========== 工具函数导入 ==========

from .utils import (
    seed_everything,
    load_pickle,
    save_pickle,
    feature_describe,
    groupby_feature_describe,
    germancredit,
    round_float,
    init_setting,
    init_logger,
    get_logger,
)

# ========== Pandas DataFrame 扩展方法 ==========
# 导入 pandas_extensions 模块，自动注册 df.summary(), df.save(), df.show() 等扩展方法
from .utils import pandas_extensions

# 导出的公共API
__all__ = [
    # 版本信息
    "__version__",

    # sklearn Pipeline 和集成学习组件
    "Pipeline",
    "make_pipeline",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "ColumnTransformer",
    "make_column_selector",
    "make_column_transformer",
    "FunctionTransformer",

    # 报告模块
    "ExcelWriter",
    "dataframe2excel",
    "auto_feature_analysis_report",

    # 可视化模块 (viz) - 基础图表
    "bin_plot",
    "corr_plot",
    "ks_plot",
    "hist_plot",
    "psi_plot",
    "dataframe_plot",
    "distribution_plot",
    "plot_weights",
    # 特征趋势分析
    "bin_trend_plot",
    "batch_bin_trend_plot",
    "bin_overdues_plot",
    # 模型评估
    "roc_plot",
    "pr_plot",
    "lift_plot",
    "gain_plot",
    "confusion_matrix_plot",
    "calibration_plot",
    # 评分卡
    "score_dist_plot",
    "score_bin_plot",
    # 风控策略
    "threshold_analysis_plot",
    "strategy_compare_plot",
    "vintage_plot",
    "feature_importance_plot",
    "approval_rate_trend_plot",
    "bad_rate_trend_plot",
    # 变量分析图表
    "variable_iv_plot",
    "variable_woe_trend_plot",
    "variable_psi_heatmap",
    "variable_importance_grouped_plot",
    "variable_missing_badrate_plot",
    # 评分分析图表
    "score_ks_plot",
    "score_distribution_comparison_plot",
    "score_badrate_bin_plot",
    "score_lift_plot",
    "score_approval_badrate_curve",
    # 策略分析图表
    "feature_trend_by_time",
    "feature_drift_comparison",
    "feature_effectiveness_by_segment",
    "feature_cross_heatmap",
    "population_drift_monitor",
    "segment_scorecard_comparison",

    # 分析模块
    "feature_bin_stats",
    "FeatureAnalyzer",

    # 规则置换风险分析模块
    "ReferenceDataProvider",
    "SwapAnalyzer",
    "SwapAnalysisResult",
    "SwapRiskConfig",
    "create_swap_dataset",
    "create_swap_dataset_from_rules",
    "swap_analysis_report",
    "SwapType",

    # 逾期数据预测模块
    "OverduePredictor",
    "overdue_prediction_report",

    # 分箱模块
    "BaseBinning",
    "UniformBinning",
    "QuantileBinning",
    "TreeBinning",
    "ChiMergeBinning",
    "BestKSBinning",
    "BestIVBinning",
    "MDLPBinning",
    "ORBinning",
    "CustomObjectives",
    "OptimalBinning",
    "CartBinning",
    "KMeansBinning",
    "MonotonicBinning",
    "GeneticBinning",
    "SmoothBinning",
    "KernelDensityBinning",
    "BestLiftBinning",
    "TargetBadRateBinning",

    # 编码器模块（core.encoders）
    "BaseEncoder",
    "WOEEncoder",
    "TargetEncoder",
    "CountEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "QuantileEncoder",
    "CatBoostEncoder",
    "GBMEncoder",
    "CardinalityEncoder",

    # 特征筛选模块（core.selection）
    "BaseFeatureSelector",
    "SelectionReportCollector",
    "VarianceSelector",
    "NullSelector",
    "ModeSelector",
    "CorrSelector",
    "VIFSelector",
    "IVSelector",
    "LiftSelector",
    "PSISelector",
    "CardinalitySelector",
    "TypeSelector",
    "RegexSelector",
    "FeatureImportanceSelector",
    "NullImportanceSelector",
    "RFESelector",
    "SequentialFeatureSelector",
    "BorutaSelector",
    "MutualInfoSelector",
    "Chi2Selector",
    "FTestSelector",
    "CompositeFeatureSelector",
    "StepwiseSelector",
    "StabilityAwareSelector",

    # 模型/损失函数模块
    "BaseLoss",
    "BaseMetric",
    "FocalLoss",
    "WeightedBCELoss",
    "CostSensitiveLoss",
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",
    "LogisticRegression",
    "ScoreCard",
    # 规则集分类
    "RuleSet",
    "RulesClassifier",
    "LogicOperator",
    "RuleResult",
    "create_and_ruleset",
    "create_or_ruleset",
    "combine_rules",

    "ks",
    "auc",
    "gini",
    "iv",
    "psi",
    "csi",
    "iv_table",
    "psi_table",
    "csi_table",
    "ks_bucket",
    "roc_curve",
    "lift",
    "lift_at",
    "lift_table",
    "lift_curve",
    "lift_monotonicity_check",
    "rule_lift",
    "badrate",
    "badrate_by_group",
    "badrate_trend",
    "badrate_by_score_bin",
    "score_stats",
    "score_stability",
    "mse",
    "mae",
    "rmse",
    "r2",

    # EDA模块 (函数式API)
    "data_info",
    "missing_analysis",
    "feature_summary",
    "numeric_summary",
    "category_summary",
    "data_quality_report",
    "feature_group_analysis",
    "population_stability_monitor",
    "target_distribution",
    "bad_rate_overall",
    "bad_rate_by_dimension",
    "bad_rate_trend",
    "bad_rate_by_bins",
    "sample_distribution",
    "numeric_distribution",
    "categorical_distribution",
    "outlier_detection",
    "iv_analysis",
    "batch_iv_analysis",
    "woe_analysis",
    "binning_bad_rate",
    "correlation_matrix",
    "high_correlation_pairs",
    "vif_analysis",
    "psi_analysis",
    "batch_psi_analysis",
    "csi_analysis",
    "vintage_analysis",
    "eda_summary",

    # 特征工程模块
    "NumExprDerive",

    # 规则引擎模块
    "Rule",
    "get_columns_from_query",
    "ruleset_report",
    "optimize_expr",
    "beautify_expr",
    "get_expr_variables",

    # 金融计算模块
    "fv",
    "pv",
    "pmt",
    "nper",
    "ipmt",
    "ppmt",
    "rate",
    "npv",
    "irr",
    "mirr",

    # 工具函数
    "seed_everything",
    "load_pickle",
    "save_pickle",
    "feature_describe",
    "groupby_feature_describe",
    "germancredit",
    "round_float",
    "init_setting",
    "init_logger",
    "get_logger",

    # 模型报告
    "QuickModelReport",
    "auto_model_report",
    "compare_models",

    # 规则挖掘
    "SingleFeatureRuleMiner",
    "MultiFeatureRuleMiner",
    "MultiLabelRuleMiner",
    "TreeRuleExtractor",
    "RuleMetrics",
    "calculate_rule_metrics",
    "TreeVisualizer",
    "plot_decision_tree",

    # 异常体系
    "HSCreditError",
    "ValidationError",
    "InputValidationError",
    "InputTypeError",
    "FeatureNotFoundError",
    "StateError",
    "NotFittedError",
    "DependencyError",
    "SerializationError",
]


def get_version():
    """获取版本号."""
    return __version__


def info():
    """打印包信息."""
    print(f"hscredit version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print("一个完整的金融信贷风险建模工具包")
    print()
    print("Pipeline 和集成学习组件 (从sklearn导入):")
    print("  - Pipeline: 管道，串联多个转换器和模型")
    print("  - make_pipeline: 快速创建Pipeline")
    print("  - VotingClassifier/VotingRegressor: 投票分类器/回归器")
    print("  - StackingClassifier/StackingRegressor: 堆叠分类器/回归器")
    print("  - ColumnTransformer: 列转换器，对不同列应用不同转换")
    print("  - FunctionTransformer: 函数转换器，将函数包装成Transformer")
    print()
    print("核心模块 (core):")
    print("  - core.binning: 分箱算法 (Uniform/Quantile/Tree/ChiMerge/BestKS/BestIV/MDLP)")
    print("  - core.selectors: 特征筛选 (Variance/Null/IV/Corr/VIF/Lift/PSI...)")
    print("  - core.encoders: 编码器 (WOE/Target/Count/OneHot...)")
    print("  - core.models: 自定义损失函数和评估指标")
    print("  - core.viz: 可视化 (bin_plot/ks_plot/corr_plot...)")
    print("  - core.feature_engineering: 特征工程 (NumExprDerive)")
    print("  - core.rules: 规则引擎 (Rule)")
    print("  - core.financial: 金融计算 (FV/PV/PMT/NPER/IRR/NPV)")
    print("  - core.eda: 数据探索分析 (EDAReport/DataOverview/TargetAnalysis/IV/PSI...)")
    print()
    print("报告模块 (report):")
    print("  - report.excel: Excel报告生成")
    print("  - report.feature_analyzer: 特征分箱统计分析")
    print("  - report.ruleset_report: 规则集综合评估报告")
    print("  - report.feature_report: 特征分析报告")
    print()
    print("工具模块 (utils):")
    print("  - utils: 工具函数 (随机种子、数据集、pickleIO)")
    print()
    print("待实现模块:")
    print("  - core.encoding: 编码转换")
    print("  - core.metrics: 指标计算 (KS/AUC/PSI/IV/Gini)")
