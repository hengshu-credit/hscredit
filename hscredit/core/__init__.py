"""核心算法模块.

包含分箱、编码、特征筛选、指标计算、可视化、金融计算、特征工程等核心功能。
"""

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
    StepwiseSelector,
    StepwiseFeatureSelector,
    BorutaSelector,
    MutualInfoSelector,
    Chi2Selector,
    FTestSelector,
    CompositeFeatureSelector,
)

# 模型模块
from .models import (
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

# 可视化模块
from .viz import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
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
    'InformationValueSelector',
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
    'StepwiseFeatureSelector',
    'BorutaSelector',
    'MutualInfoSelector',
    'Chi2Selector',
    'FTestSelector',
    'CompositeFeatureSelector',

    # 模型/损失函数
    'BaseLoss',
    'BaseMetric',
    'FocalLoss',
    'WeightedBCELoss',
    'CostSensitiveLoss',
    'BadDebtLoss',
    'ApprovalRateLoss',
    'ProfitMaxLoss',
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
