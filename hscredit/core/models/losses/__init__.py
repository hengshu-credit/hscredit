"""自定义损失函数模块.

为各种机器学习框架提供自定义损失函数和评估指标，特别针对金融风控场景优化。

支持框架:
- XGBoost
- LightGBM
- CatBoost
- TabNet
- PyTorch (可选)

核心功能:
- 不平衡数据处理 (Focal Loss, Class Weighted Loss)
- 成本敏感学习 (Cost-sensitive Loss)
- 风控业务指标 (Bad Debt Loss, Approval Rate Loss)
- 自定义评估指标 (KS, Gini, PSI等)
"""

from .base import BaseLoss, BaseMetric
from .focal_loss import FocalLoss
from .weighted_loss import WeightedBCELoss, CostSensitiveLoss
from .risk_loss import BadDebtLoss, ApprovalRateLoss, ProfitMaxLoss
from .custom_metrics import KSMetric, GiniMetric, PSIMetric
from .adapters import (
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
)

__all__ = [
    # 基类
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
]
