"""评分卡子包.

包含评分卡相关的模型和工具:
- ScoreCard: 评分卡模型（逻辑回归 → WOE 线性评分卡）
- RoundScoreCard: 按分箱分数精度一致计分的评分卡
- ProbabilityScoreCard: 通用模型评分卡（任意 predict_proba 模型 → 评分）
- StandardScoreTransformer: 标准评分转换器
- LinearScoreTransformer: 线性评分转换器
- QuantileScoreTransformer: 分位数评分转换器
- BoxCoxScoreTransformer: Box-Cox幂变换评分转换器
- ScoreTransformer: 统一评分转换接口
- transform_probability_to_score: 便捷概率转评分函数
- ScoreDriftCalibrator: 评分漂移校准
"""

import os as _os

if _os.environ.get("HSCREDIT_LIGHT_IMPORT") == "1":
    from importlib import import_module as _import_module
    from hscredit._public_api import EXPORTS

    _exports = EXPORTS[__name__]
    __all__ = [
        "ScoreCard",
        "RoundScoreCard",
        "ProbabilityScoreCard",
        "BaseScoreTransformer",
        "StandardScoreTransformer",
        "LinearScoreTransformer",
        "QuantileScoreTransformer",
        "BoxCoxScoreTransformer",
        "ScoreTransformer",
        "transform_probability_to_score",
        "ScoreDriftCalibrator",
    ]

    def __getattr__(name):
        if name not in _exports:
            raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
        value = getattr(_import_module(_exports[name]), name)
        globals()[name] = value
        return value

    def __dir__():
        return sorted(set(globals()) | set(__all__))

else:
    from .scorecard import ScoreCard, RoundScoreCard

    from .score_transformer import (
        BaseScoreTransformer,
        StandardScoreTransformer,
        LinearScoreTransformer,
        QuantileScoreTransformer,
        BoxCoxScoreTransformer,
        ScoreTransformer,
        transform_probability_to_score,
    )
    from .model_scorecard import ProbabilityScoreCard
    from .score_drift import ScoreDriftCalibrator

    __all__ = [
        "ScoreCard",
        "RoundScoreCard",
        "ProbabilityScoreCard",
        "BaseScoreTransformer",
        "StandardScoreTransformer",
        "LinearScoreTransformer",
        "QuantileScoreTransformer",
        "BoxCoxScoreTransformer",
        "ScoreTransformer",
        "transform_probability_to_score",
        "ScoreDriftCalibrator",
    ]
