"""模型评估子包.

包含模型评估、校准、解释相关的工具:
- ModelReport: 模型评估报告
- ProbabilityCalibrator / CalibratedModel: 概率校准
- ModelExplainer: SHAP模型解释（懒加载，首次访问时才导入 shap）
"""

import importlib

from .report import ModelReport
from .calibration import (
    ProbabilityCalibrator,
    CalibratedModel,
    PlattCalibrator,
    IsotonicCalibrator,
    BetaCalibrator,
    HistogramCalibrator,
)

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


def __getattr__(name):
    if name != "ModelExplainer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(".interpretability", __name__)
    value = module.ModelExplainer
    globals()[name] = value
    return value
