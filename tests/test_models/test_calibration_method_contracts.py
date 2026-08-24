"""拆分后基础校准器与具体算法的独立契约。"""

import numpy as np
import pytest

from hscredit.core.models.calibration import (
    BaseCalibrator,
    BetaCalibrator,
    HistogramCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)


def test_calibration_base_and_methods_are_owned_by_focused_modules():
    """防止拆包后仍把全部实现堆在 model.py。"""
    assert BaseCalibrator.__module__ == "hscredit.core.models.calibration.base"
    for calibrator in (PlattCalibrator, IsotonicCalibrator, BetaCalibrator, HistogramCalibrator):
        assert calibrator.__module__ == "hscredit.core.models.calibration.methods"


@pytest.mark.parametrize("factory", [PlattCalibrator, IsotonicCalibrator, BetaCalibrator, HistogramCalibrator])
def test_calibration_methods_reject_nonfinite_probabilities(factory):
    """任何算法都不能把 NaN 概率带入拟合状态。"""
    with pytest.raises(ValueError, match="概率"):
        factory().fit(np.array([0.2, np.nan]), np.array([0, 1]))


def test_set_params_invalid_strategy_is_rejected_before_metric_calculation():
    """set_params 不得绕过构造期的分箱策略校验。"""
    calibrator = PlattCalibrator().set_params(strategy="bad")
    with pytest.raises(ValueError, match="strategy"):
        calibrator.compute_calibration_metrics(np.array([0, 1]), np.array([0.2, 0.8]))


def test_histogram_quantile_duplicates_return_finite_probabilities():
    """重复分位点仍应映射到真实箱内频率。"""
    calibrator = HistogramCalibrator(n_bins=5).fit(np.full(10, 0.2), np.array([0, 1] * 5))
    result = calibrator.calibrate(np.array([0.2]))
    assert result.tolist() == pytest.approx([0.5])


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: PlattCalibrator(C=0), "C"),
        (lambda: IsotonicCalibrator(out_of_bounds="bad"), "out_of_bounds"),
    ],
)
def test_method_specific_constructor_parameters_are_validated(factory, message):
    """子类参数不能因基类过早校验而漏过构造期检查。"""
    with pytest.raises(ValueError, match=message):
        factory()


def test_probability_calibrator_rejects_invalid_bin_count_at_construction():
    """统一包装器应在构造时拒绝明显非法的分箱数。"""
    from hscredit.core.models.calibration import ProbabilityCalibrator

    with pytest.raises(ValueError, match="n_bins"):
        ProbabilityCalibrator(n_bins=0)
