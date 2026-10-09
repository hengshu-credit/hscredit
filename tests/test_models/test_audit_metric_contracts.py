"""审计发现的统计守恒与监控基准回归。"""
import numpy as np
import pandas as pd
import pytest

from hscredit.core.metrics import compute_bin_stats, psi, psi_table


@pytest.mark.parametrize("y", [[0, 2, 0, 2], [0, np.nan, 0, 1], [0, np.inf, 0, 1], [[0, 1], [0, 1]]])
def test_binary_statistics_reject_invalid_target(y):
    with pytest.raises(ValueError):
        compute_bin_stats([0, 0, 1, 1], y)


@pytest.mark.parametrize("bins,y", [([0], [0, 1]), ([0, np.nan], [0, 1]), ([0, .5], [0, 1])])
def test_statistics_reject_invalid_bins(bins, y):
    with pytest.raises(ValueError):
        compute_bin_stats(bins, y)


def test_large_bin_codes_do_not_cross_reserved_bins():
    table = compute_bin_stats([20000, -1, -2, 0], [1, 0, 1, 0])
    assert table["分箱"].tolist() == [0, 20000, -1, -2]
    assert table["样本总数"].sum() == 4


def test_missing_population_drift_is_not_hidden():
    expected = np.array([0., 1.] * 50)
    actual = np.array([0., 1.] + [np.nan] * 98)
    assert psi(expected, actual, n_jobs=1) > 1
    table = psi_table(expected, actual, n_jobs=1)
    assert table["期望样本数"].sum() == table["实际样本数"].sum() == 100


def test_monitoring_baseline_freezes_categories_and_roundtrips(tmp_path):
    from hscredit.core.metrics.monitoring import MonitoringBaseline
    baseline = MonitoringBaseline().fit(pd.Series(["A"] * 90 + ["B"] * 10))
    drift = baseline.evaluate(pd.Series(["A"] * 10 + ["B"] * 80 + ["C"] * 10))
    assert drift["PSI"] > 1
    assert drift["未知类别率"] == pytest.approx(.1)
    saved = tmp_path / "monitor.joblib"
    baseline.save_artifact(saved)
    loaded = MonitoringBaseline.load_artifact(saved)
    assert loaded.evaluate(pd.Series(["A"] * 90 + ["B"] * 10))["PSI"] == 0


def test_monitoring_baseline_never_refits_when_evaluating():
    from hscredit.core.metrics.monitoring import MonitoringBaseline
    baseline = MonitoringBaseline(max_n_bins=4).fit(pd.Series(np.arange(100, dtype=float)))
    before = baseline.binner_.splits_["value"].copy()
    baseline.evaluate(pd.Series([1000., 10000., np.nan]))
    np.testing.assert_array_equal(before, baseline.binner_.splits_["value"])


def test_streaming_binary_counts_equal_whole_table():
    from hscredit.core.metrics.aggregation import BinStatsAccumulator
    rng = np.random.RandomState(42)
    bins = rng.randint(-2, 20, 1000)
    y = rng.randint(0, 2, 1000)
    left = BinStatsAccumulator().update(bins[:600], y[:600])
    right = BinStatsAccumulator().update(bins[600:], y[600:])
    actual = left.merge(right).finalize(round_digits=False)
    expected = compute_bin_stats(bins, y, round_digits=False)
    pd.testing.assert_frame_equal(actual, expected)


def test_numeric_text_codes_are_normalized_before_aggregation():
    table = compute_bin_stats(['0', '1'], [0, 1])
    assert table['样本总数'].sum() == 2
    assert table['分箱'].tolist() == [0, 1]


def test_baseline_smoothing_is_frozen_and_part_of_version():
    from hscredit.core.metrics import MonitoringBaseline
    base = ['a'] * 90 + ['b'] * 10
    first = MonitoringBaseline(epsilon=1e-10).fit(base)
    second = MonitoringBaseline(epsilon=.1).fit(base)
    assert first.version_ != second.version_
    value = first.evaluate(['a'] * 100)['PSI']
    first.set_params(epsilon=.1)
    assert first.evaluate(['a'] * 100)['PSI'] == value


def test_baseline_accepts_array_user_splits():
    from hscredit.core.metrics import MonitoringBaseline
    baseline = MonitoringBaseline(binning_params={'user_splits': np.array([.5, 1.5])}).fit([0., 1., 2., 3.])
    assert baseline.evaluate([0., 1., 2., 3.])['PSI'] == 0


@pytest.mark.parametrize('method', ['quantile', 'uniform', 'tree', 'best_iv'])
def test_empty_window_and_all_missing_population_have_explicit_status(method):
    table = psi_table([0., 1., 2.], [], method=method)
    assert table.attrs['状态'] == '数据不足'
    assert table['PSI贡献'].isna().all()
    missing = psi_table([np.nan] * 3, [1., 2., 3.], method=method)
    assert missing['PSI贡献'].sum() > 1


def test_monitoring_honors_explicit_categorical_groups():
    from hscredit.core.metrics import MonitoringBaseline
    base = MonitoringBaseline(binning_params={'user_splits': {'value': [['a', 'b'], ['c']]}}).fit(['a', 'b', 'c', 'a'])
    codes = base.transform(['a', 'b', 'c', 'z'])
    assert codes[0] == codes[1] != codes[2]
    assert codes[3] == -3
    with pytest.raises(TypeError):
        MonitoringBaseline(binning_params={'unknown_option': 42}).fit(['a', 'b'])


def test_cross_psi_uses_each_row_group_as_reference_instead_of_mirroring():
    from hscredit.core.eda.stability import _psi_cross_feature_worker
    first = np.arange(100, dtype=float)
    second = first ** 2
    frame = pd.DataFrame({'x': np.r_[first, second], 'group': ['A'] * 100 + ['B'] * 100})
    _, matrix = _psi_cross_feature_worker((frame, 'x', 'group', ['A', 'B'], 5, True))
    assert matrix.loc['A', 'B'] == pytest.approx(psi(first, second, max_n_bins=5))
    assert matrix.loc['B', 'A'] == pytest.approx(psi(second, first, max_n_bins=5))
    assert matrix.loc['A', 'B'] != pytest.approx(matrix.loc['B', 'A'])
