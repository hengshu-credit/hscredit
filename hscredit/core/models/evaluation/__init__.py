"""模型评估子包.

包含模型评估、校准、解释相关的工具:
- ModelReport: 模型评估报告
- ProbabilityCalibrator / CalibratedModel: 概率校准
- ModelExplainer: SHAP模型解释
"""

from .report import ModelReport
from .calibration import (
    ProbabilityCalibrator,
    CalibratedModel,
    PlattCalibrator,
    IsotonicCalibrator,
    BetaCalibrator,
    HistogramCalibrator,
)
from .interpretability import ModelExplainer

__all__ = [
    "ModelReport",
    "ProbabilityCalibrator",
    "CalibratedModel",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "HistogramCalibrator",
    "ModelExplainer",
]
