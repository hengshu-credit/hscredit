"""核心算法模块.

包含分箱、编码、特征筛选、指标计算、可视化、金融计算、特征工程等核心功能。
"""

# 分箱模块
from .binning import (
    BaseBinning,
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    CartBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    OptimalBinning,
    MDLPBinning,
    ORBinning,
    CustomObjectives,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
)

# 指标计算模块
from .metrics import (
    ks, auc, gini,
    accuracy, precision, recall, f1,
    ks_bucket, roc_curve,
    confusion_matrix, classification_report,
    psi, csi,
    psi_table, csi_table,
    iv, iv_table,
    feature_importance,
    mse, mae, rmse, r2,
)

# 编码器模块
from .encoders import (
    BaseEncoder,
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
)

# 特征筛选模块
from .selectors import (
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
    StepwiseSelector,
    BorutaSelector,
    MutualInfoSelector,
    Chi2Selector,
    FTestSelector,
    CompositeFeatureSelector,
    ScorecardFeatureSelection,
)

# 模型模块
from .models import (
    BaseLoss,
    BaseMetric,
    FocalLoss,
    AsymmetricFocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    KSMetric,
    GiniMetric,
    PSIMetric,
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
)

# 可视化模块
from .viz import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
    plot_weights,
)

# 金融计算模块
from .financial import (
    fv,
    pv,
    pmt,
    nper,
    ipmt,
    ppmt,
    rate,
    npv,
    irr,
    mirr,
)

# 特征工程模块
from .feature_engineering import NumExprDerive

# 规则引擎模块
from .rules import (
    Rule,
    get_columns_from_query,
    optimize_expr,
    beautify_expr,
    get_expr_variables,
)

__all__ = [
    # 分箱
    'BaseBinning',
    'UniformBinning',
    'QuantileBinning',
    'TreeBinning',
    'CartBinning',
    'ChiMergeBinning',
    'BestKSBinning',
    'BestIVBinning',
    'OptimalBinning',
    'MDLPBinning',
    'ORBinning',
    'CustomObjectives',
    'KMeansBinning',
    'MonotonicBinning',
    'GeneticBinning',
    'SmoothBinning',
    'KernelDensityBinning',
    'BestLiftBinning',
    'TargetBadRateBinning',

    'ks', 'auc', 'gini',
    'accuracy', 'precision', 'recall', 'f1',
    'ks_bucket', 'roc_curve',
    'confusion_matrix', 'classification_report',
    'psi', 'csi',
    'psi_table', 'csi_table',
    'iv', 'iv_table',
    'feature_importance',
    'mse', 'mae', 'rmse', 'r2',

    # 编码器
    'BaseEncoder',
    'WOEEncoder',
    'TargetEncoder',
    'CountEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'QuantileEncoder',
    'CatBoostEncoder',

    # 特征筛选
    'BaseFeatureSelector',
    'SelectionReportCollector',
    'VarianceSelector',
    'NullSelector',
    'ModeSelector',
    'CorrSelector',
    'VIFSelector',
    'IVSelector',
    'LiftSelector',
    'PSISelector',
    'CardinalitySelector',
    'TypeSelector',
    'RegexSelector',
    'FeatureImportanceSelector',
    'NullImportanceSelector',
    'RFESelector',
    'SequentialFeatureSelector',
    'StepwiseSelector',
    'BorutaSelector',
    'MutualInfoSelector',
    'Chi2Selector',
    'FTestSelector',
    'CompositeFeatureSelector',
    'ScorecardFeatureSelection',

    # 模型/损失函数
    'BaseLoss',
    'BaseMetric',
    'FocalLoss',
    'AsymmetricFocalLoss',
    'WeightedBCELoss',
    'CostSensitiveLoss',
    'BadDebtLoss',
    'ApprovalRateLoss',
    'ProfitMaxLoss',
    'OrdinalRankLoss',
    'LiftFocusedLoss',
    'KSMetric',
    'GiniMetric',
    'PSIMetric',
    'XGBoostLossAdapter',
    'LightGBMLossAdapter',
    'CatBoostLossAdapter',
    'TabNetLossAdapter',

    # 可视化
    'bin_plot',
    'corr_plot',
    'ks_plot',
    'hist_plot',
    'psi_plot',
    'dataframe_plot',
    'distribution_plot',
    'plot_weights',

    # 金融计算
    'fv',
    'pv',
    'pmt',
    'nper',
    'ipmt',
    'ppmt',
    'rate',
    'npv',
    'irr',
    'mirr',

    # 特征工程
    'NumExprDerive',

    # 规则引擎
    'Rule',
    'get_columns_from_query',
    'optimize_expr',
    'beautify_expr',
    'get_expr_variables',
]
