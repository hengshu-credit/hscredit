"""核心算法模块.

包含分箱、编码、特征筛选、指标计算等核心功能。
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
]
