"""hscredit - 金融信贷风险策略和模型开发库.

一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。
"""

__version__ = "0.1.0"
__author__ = "hscredit team"
__email__ = "hscredit@hengshucredit.com"


# ========== 核心模块导入 (先导入core，避免循环导入) ==========

# 核心分箱模块
from .core.binning import (
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
)

# 核心可视化模块
from .core.viz import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
    plot_weights,
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

# 核心指标计算模块
from .core.metrics import (
    KS, AUC, Gini, PSI, IV,
    KS_bucket, ROC_curve,
    PSI_table, CSI_table,
    IV_table,
    MSE, MAE, RMSE, R2,
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
)

# 导出的公共API
__all__ = [
    # 版本信息
    "__version__",

    # 报告模块
    "ExcelWriter",
    "dataframe2excel",
    "auto_feature_analysis_report",

    # 可视化模块 (viz)
    "bin_plot",
    "corr_plot",
    "ks_plot",
    "hist_plot",
    "psi_plot",
    # "csi_plot",
    "dataframe_plot",
    "distribution_plot",
    "plot_weights",

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

    # 分箱模块
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

    # 指标计算模块
    "KS",
    "AUC",
    "Gini",
    "PSI",
    "IV",
    "KS_bucket",
    "ROC_curve",
    "PSI_table",
    "CSI_table",
    "IV_table",
    "MSE",
    "MAE",
    "RMSE",
    "R2",

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
    print("核心模块 (core):")
    print("  - core.binning: 分箱算法 (Uniform/Quantile/Tree/ChiMerge/BestKS/BestIV/MDLP)")
    print("  - core.selectors: 特征筛选 (Variance/Null/IV/Corr/VIF/Lift/PSI...)")
    print("  - core.encoders: 编码器 (WOE/Target/Count/OneHot...)")
    print("  - core.models: 自定义损失函数和评估指标")
    print("  - core.viz: 可视化 (bin_plot/ks_plot/corr_plot...)")
    print("  - core.feature_engineering: 特征工程 (NumExprDerive)")
    print("  - core.rules: 规则引擎 (Rule)")
    print("  - core.financial: 金融计算 (FV/PV/PMT/NPER/IRR/NPV)")
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
