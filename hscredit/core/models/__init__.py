"""模型模块 (core/models).

提供评分卡建模相关的模型和工具。

核心功能:
- 逻辑回归模型 (扩展sklearn) - 待实现
- 评分卡转换 - 待实现
- PMML导出 - 待实现
- 模型验证 - 待实现
- 自定义损失函数和评估指标 - 已实现
"""

# 导入已实现的模块
from .losses import (
    # 基类
    BaseLoss,
    BaseMetric,
    # 不平衡数据处理
    FocalLoss,
    WeightedBCELoss,
    # 成本敏感
    CostSensitiveLoss,
    # 风控业务损失
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    # 自定义评估指标
    KSMetric,
    GiniMetric,
    PSIMetric,
    # 框架适配器
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
)

# 待实现的模块（暂时注释）
# from .linear import LogisticRegression, StatsLogisticRegression
# from .scorecard import ScoreCard, ScoreCardValidator

__all__ = [
    # 损失函数基类
    "BaseLoss",
    "BaseMetric",
    # 不平衡数据处理
    "FocalLoss",
    "WeightedBCELoss",
    # 成本敏感
    "CostSensitiveLoss",
    # 风控业务损失
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    # 自定义评估指标
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    # 框架适配器
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",

    # 待实现的模型（暂时注释）
    # "LogisticRegression",
    # "StatsLogisticRegression",
    # "ScoreCard",
    # "ScoreCardValidator",
]
