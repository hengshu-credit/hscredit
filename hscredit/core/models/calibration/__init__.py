"""模型概率校准子包。"""

import os as _os

if _os.environ.get("HSCREDIT_LIGHT_IMPORT") == "1":
    from importlib import import_module as _import_module
    from hscredit._public_api import EXPORTS

    _exports = EXPORTS[__name__]
    __all__ = [
        "BaseCalibrator",
        "PlattCalibrator",
        "IsotonicCalibrator",
        "BetaCalibrator",
        "HistogramCalibrator",
        "ProbabilityCalibrator",
        "CalibratedModel",
        "calibrate_model",
        "plot_calibration_comparison",
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
    from .base import BaseCalibrator
    from .methods import BetaCalibrator, HistogramCalibrator, IsotonicCalibrator, PlattCalibrator
    from .model import CalibratedModel, ProbabilityCalibrator, calibrate_model
    from .plots import plot_calibration_comparison

    __all__ = [
        "BaseCalibrator",
        "PlattCalibrator",
        "IsotonicCalibrator",
        "BetaCalibrator",
        "HistogramCalibrator",
        "ProbabilityCalibrator",
        "CalibratedModel",
        "calibrate_model",
        "plot_calibration_comparison",
    ]
