"""特征编码器模块.

提供各类特征编码功能，包括：
- WOEEncoder: 证据权重编码
- TargetEncoder: 目标编码
- CountEncoder: 计数编码
- OneHotEncoder: 独热编码
- OrdinalEncoder: 序数编码
- QuantileEncoder: 分位数编码
- CatBoostEncoder: CatBoost编码
- GBMEncoder: 梯度提升树编码器（支持XGBoost/LightGBM/CatBoost+LR）

所有编码器均遵循sklearn Transformer接口规范。
"""

from .base import BaseEncoder
from .woe_encoder import WOEEncoder
from .target_encoder import TargetEncoder
from .count_encoder import CountEncoder
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .quantile_encoder import QuantileEncoder
from .catboost_encoder import CatBoostEncoder
from .gbm_encoder import GBMEncoder
from .cardinality_encoder import CardinalityEncoder

__all__ = [
    'BaseEncoder',
    'WOEEncoder',
    'TargetEncoder',
    'CountEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'QuantileEncoder',
    'CatBoostEncoder',
    'GBMEncoder',
    'CardinalityEncoder',
]
