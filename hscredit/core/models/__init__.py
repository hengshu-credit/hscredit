"""模型模块 (core/models).

提供评分卡建模相关的模型和工具。

核心功能:
- 逻辑回归模型 (扩展sklearn) - 已实现
- 评分卡转换 - 待实现
- PMML导出 - 待实现
- 模型验证 - 待实现
- 自定义损失函数和评估指标 - 已实现

示例:
    >>> from hscredit.core.models import LogisticRegression
    >>> model = LogisticRegression(calculate_stats=True)
    >>> model.fit(X_train, y_train)
    >>> summary = model.summary()
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

# 导入逻辑回归模型
from .logistic_regression import LogisticRegression

# 导入评分卡模型
from .scorecard import ScoreCard

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
    # 逻辑回归模型
    "LogisticRegression",
    # 评分卡模型
    "ScoreCard",
]
