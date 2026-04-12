"""自定义损失函数模块.

为各种机器学习框架提供自定义损失函数和评估指标，特别针对金融风控场景优化。

支持框架:
- XGBoost
- LightGBM
- CatBoost
- NGBoost
- TabNet
- PyTorch (可选)

核心功能:
- 不平衡数据处理 (Focal Loss, Balanced Focal Loss, Class Weighted Loss)
- 成本敏感学习 (Cost-sensitive Loss)
- 风控业务指标 (Bad Debt Loss, Approval Rate Loss, Expected Profit Loss)
- 排序与 AUC 优化 (OrdinalRankLoss, RankingAUCProxyLoss, LiftFocusedLoss)
- KS 分布分离 (KSFocusedLoss)
- 头部捕获优化 (TopKBadCaptureLoss)
- 金额/敞口加权 (AmountWeightedLoss, ExpectedValueLoss)
- 自定义评估指标 (KS, Gini, PSI等)
"""

from .base import BaseLoss, BaseMetric
from .focal_loss import FocalLoss
from .asymmetric_focal_loss import AsymmetricFocalLoss
from .balanced_focal_loss import BalancedFocalLoss
from .weighted_loss import WeightedBCELoss, CostSensitiveLoss
from .risk_loss import BadDebtLoss, ApprovalRateLoss, ProfitMaxLoss
from .expected_profit_loss import ExpectedProfitLoss
from .ranking_loss import OrdinalRankLoss, LiftFocusedLoss
from .ranking_auc_proxy_loss import RankingAUCProxyLoss
from .ks_focused_loss import KSFocusedLoss
from .topk_bad_capture_loss import TopKBadCaptureLoss
from .amount_weighted_loss import AmountWeightedLoss, ExpectedValueLoss
from .custom_metrics import KSMetric, GiniMetric, PSIMetric
from .adapters import (
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
    NGBoostLossAdapter,
)

__all__ = [
    # 基类
    "BaseLoss",
    "BaseMetric",
    # 不平衡数据处理
    "FocalLoss",
    "AsymmetricFocalLoss",
    "BalancedFocalLoss",
    "WeightedBCELoss",
    # 成本敏感
    "CostSensitiveLoss",
    # 风控业务损失
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    "ExpectedProfitLoss",
    # 排序与 AUC 优化
    "OrdinalRankLoss",
    "LiftFocusedLoss",
    "RankingAUCProxyLoss",
    # KS 分布分离
    "KSFocusedLoss",
    # 头部捕获优化
    "TopKBadCaptureLoss",
    # 金额/敞口加权
    "AmountWeightedLoss",
    "ExpectedValueLoss",
    # 自定义评估指标
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    # 框架适配器
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",
    "NGBoostLossAdapter",
]
