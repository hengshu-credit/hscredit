"""hscredit - 金融信贷风险策略和模型开发库.

一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。
"""

__version__ = "0.1.0"
__author__ = "hscredit team"
__email__ = "hscredit@hengshucredit.com"

# 注意: 以下模块尚未实现，暂时注释
# 核心模块导入
# from .core.binning import OptimalBinning, TreeBinning, ChiMergeBinning
# from .core.encoding import WOEEncoder, TargetEncoder
# from .core.selection import (
#     FeatureSelector,
#     StepwiseSelector,
#     IVSelector,
#     CorrelationSelector,
#     VIFSelector
# )
# from .core.metrics import KS, AUC, PSI, IV, Gini

# 模型模块导入
# from .model.linear import LogisticRegression
# from .model.scorecard import ScoreCard

# 分析模块导入
# 注意：feature_bin_stats 和 FeatureAnalyzer 已从 analysis 迁移到 report 模块

# 报告模块导入
from .report.excel import ExcelWriter, dataframe2excel

try:
    from .report.feature_report import auto_feature_analysis_report
except ImportError:
    pass

# 可视化模块导入 (viz - 简洁命名)
from .core.viz import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
)

# 分析模块导入
from .report import feature_bin_stats, FeatureAnalyzer

# 核心模块导入
from .core.binning import (
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    OptimalKSBinning,
    OptimalIVBinning,
    MDLPBinning,
    OptimalBinning,
)

# 核心模块编码器导入
from .core.encoders import (
    BaseEncoder,
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
)

# 核心模块特征筛选导入
from .core.selectors import (
    BaseFeatureSelector,
    SelectionReportCollector,
    VarianceSelector,
    NullSelector,
    ModeSelector,
    CorrSelector,
    VIFSelector,
    InformationValueSelector,
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

# 模型/损失函数模块导入
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
)

# 特征工程模块导入
from .core.feature_engineering import NumExprDerive

# 规则引擎模块导入
from .core.rules import (
    Rule,
    get_columns_from_query,
    optimize_expr,
    beautify_expr,
    get_expr_variables,
)
from .report import ruleset_report

# 金融计算模块导入
from .core.financial import (
    fv, pv, pmt, nper, ipmt, ppmt, rate,
    npv, irr, mirr,
)

# 工具函数导入
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

    # 分析模块
    "feature_bin_stats",
    "FeatureAnalyzer",

    # 分箱模块
    "UniformBinning",
    "QuantileBinning",
    "TreeBinning",
    "ChiMergeBinning",
    "OptimalKSBinning",
    "OptimalIVBinning",
    "MDLPBinning",
    "OptimalBinning",

    # 编码器模块（core.encoders）
    "BaseEncoder",
    "WOEEncoder",
    "TargetEncoder",
    "CountEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "QuantileEncoder",
    "CatBoostEncoder",

    # 特征筛选模块（core.selection）
    "BaseFeatureSelector",
    "SelectionReportCollector",
    "VarianceSelector",
    "NullSelector",
    "ModeSelector",
    "CorrSelector",
    "VIFSelector",
    "InformationValueSelector",
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
    print("已实现模块:")
    print("  - report.excel: Excel报告生成")
    print("  - analysis: 特征分析 (feature_bin_stats, FeatureAnalyzer)")
    print("  - core.binning: 分箱算法 (Uniform/Quantile/Tree/ChiMerge/OptimalKS/OptimalIV/MDLP)")
    print("  - core.selectors: 特征筛选 (Variance/Null/IV/Corr/VIF/Lift/PSI...)")
    print("  - core.encoders: 编码器 (WOE/Target/Count/OneHot...)")
    print("  - core.models: 自定义损失函数和评估指标")
    print("  - feature_engineering: 特征工程 (表达式衍生)")
    print("  - rules: 规则引擎")
    print("  - financial: 金融计算 (FV/PV/PMT/NPER/IRR/NPV)")
    print("  - utils: 工具函数 (随机种子、数据集、pickleIO)")
    print()
    print("待实现模块:")
    print("  - core.encoding: 编码转换")
    print("  - core.metrics: 指标计算 (KS/AUC/PSI/IV/Gini)")
    print("  - analysis.strategy: 策略分析")
