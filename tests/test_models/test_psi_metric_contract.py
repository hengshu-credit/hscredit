"""PSIMetric 与公共稳定性统计的口径和输入边界。"""

import numpy as np
import pytest

from hscredit.core.metrics import psi, psi_table
from hscredit.core.models.losses import PSIMetric


def test_psi_metric_counts_predictions_outside_baseline_range():
    expected = np.linspace(0.2, 0.8, 100)
    actual = np.linspace(0.9, 1.0, 100)
    table = psi_table(expected, actual, method="quantile", max_n_bins=5)
    value = PSIMetric(expected, n_bins=5)(np.ones(100), actual)
    assert table["实际样本数"].sum() == len(actual)
    assert value == pytest.approx(table["PSI贡献"].sum())
    assert value == pytest.approx(psi(expected, actual, method="quantile", max_n_bins=5))
    assert value > 0


def test_psi_metric_uses_public_constant_baseline_fallback():
    expected = np.full(100, 0.5)
    actual = np.full(100, 0.9)
    metric = PSIMetric(expected, n_bins=5)
    assert metric(None, actual) == pytest.approx(psi(expected, actual, max_n_bins=5))
    assert metric(None, actual) > 0
    assert metric(None, expected) == pytest.approx(0)


@pytest.mark.parametrize("n_bins", [True, 0, 1, -1, 2.5, "5", np.nan])
def test_psi_metric_rejects_invalid_bin_counts(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        PSIMetric(n_bins=n_bins)


@pytest.mark.parametrize("values", [[], [[0.1, 0.2]], [np.nan], [np.inf], ["无效"]])
def test_psi_metric_validates_both_distributions(values):
    with pytest.raises(ValueError, match="基准分布"):
        PSIMetric(expected=values)
    metric = PSIMetric(expected=[0.1, 0.2], n_bins=2)
    with pytest.raises(ValueError, match="当前分布"):
        metric(None, values)
    with pytest.raises(ValueError, match="基准分布"):
        metric(None, [0.1, 0.2], expected=values)


def test_psi_metric_keeps_snapshot_and_override_local():
    expected = np.linspace(0.1, 0.9, 100)
    original = expected.copy()
    metric = PSIMetric(expected, n_bins=5)
    expected[:] = 0
    assert metric(None, original) == pytest.approx(0)
    override = np.linspace(0.01, 0.05, 100)
    assert metric(None, original, expected=override) > 0
    assert metric(None, original) == pytest.approx(0)


def test_psi_metric_has_clear_missing_reference_and_weight_errors():
    with pytest.raises(ValueError, match="expected"):
        PSIMetric()(None, [0.1, 0.9])
    with pytest.raises(ValueError, match="不支持样本权重"):
        PSIMetric(expected=[0.1, 0.9]).evaluate([0, 1], [0.1, 0.9], sample_weight=[1, 1])
