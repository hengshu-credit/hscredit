"""模型校准与可解释性子包的不兼容迁移契约。"""

import importlib

import pytest


def test_new_model_subpackages_own_canonical_public_api():
    """新公开对象必须由拆分后的子包定义，而非旧包转发。"""
    from hscredit.core.models.calibration import ProbabilityCalibrator, calibrate_model
    from hscredit.core.models.explainability import ModelExplainer, build_reason_codes

    assert ProbabilityCalibrator.__module__.startswith("hscredit.core.models.calibration")
    assert ModelExplainer.__module__.startswith("hscredit.core.models.explainability")
    assert callable(calibrate_model)
    assert callable(build_reason_codes)


def test_old_evaluation_package_is_removed():
    """迁移完成后旧 evaluation 包不能继续被导入。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hscredit.core.models.evaluation")
