"""模型评估子包.

包含模型评估、校准、解释相关的工具:
- ModelReport: 模型评估报告（已统一为 :class:`hscredit.report.ModelReport`，
  此处仅作为兼容别名按需懒加载，避免与 ``hscredit.report.model_report`` 重复实现）
- ProbabilityCalibrator / CalibratedModel: 概率校准
- ModelExplainer: SHAP模型解释（懒加载，首次访问时才导入 shap）
- model_explain_report: 不依赖 SHAP 的基础模型解释报告
"""

import importlib

from .interpretability import model_explain_report
from .explanation import ExplanationResult
from .counterfactual import CounterfactualExplainer
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
    "ExplanationResult",
    "CounterfactualExplainer",
    "model_explain_report",
]


def __getattr__(name):
    if name == "ModelExplainer":
        module = importlib.import_module(".explainer", __name__)
        value = module.ModelExplainer
    elif name == "ModelReport":
        # 模型报告已统一由 hscredit.report.model_report 生成，这里仅做兼容别名，
        # 懒加载以避免在 import hscredit 期间触发 hscredit.report 的循环导入。
        module = importlib.import_module("hscredit.report.model_report")
        value = module.ModelReport
    else:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    globals()[name] = value
    return value
