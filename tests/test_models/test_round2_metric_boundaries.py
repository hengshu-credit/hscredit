"""第二轮监控版本、缺失政策及稳定数值统计的回归。"""
import numpy as np
import pandas as pd
import pytest
from hscredit import MonitoringBaseline, compute_bin_stats, psi_table
from hscredit.core.binning import OptimalBinning, QuantileBinning, TreeBinning
from hscredit.core.metrics.aggregation import BinStatsAccumulator


def test_baseline_version_includes_transform_policy_and_raw_unknown_rate():
    config = {"user_splits": [["A"], ["B"]]}
    a = MonitoringBaseline(binning_params={**config, "handle_unknown": -3}).fit(["A", "B"])
    b = MonitoringBaseline(binning_params={**config, "handle_unknown": 0}).fit(["A", "B"])
    assert a.version_ != b.version_
    assert a.evaluate(["C"])["未知类别率"] == b.evaluate(["C"])["未知类别率"] == 1.0
    assert b.evaluate_batches([["C"], ["A"]])["未知类别率"] == .5


@pytest.mark.parametrize('policy', [-3, -1, 0])
def test_unknown_merged_into_missing_does_not_change_raw_missing_count(policy):
    baseline = MonitoringBaseline(binning_params={'user_splits': [['A'], ['B']], 'handle_unknown': policy}).fit(['A', 'B', None])
    result = baseline.evaluate(['C', None, 'A'], include_missing=False)
    assert result['未知类别率'] == result['实际缺失率'] == pytest.approx(1 / 3)
    assert result['有效样本数'] == result['分箱明细']['实际样本数'].sum() == 2


@pytest.mark.parametrize("method", ["quantile", "uniform", "tree", "best_iv"])
def test_excluded_missing_values_never_reappear_in_fallback(method):
    table = psi_table([0., 1.], [np.nan, np.nan], method=method, include_missing=False)
    assert table["实际样本数"].sum() == 0
    assert table["PSI贡献"].isna().all()
    assert table.attrs["状态"] == "数据不足"
    assert table.attrs["缺失策略"] == "排除"


def test_supplied_baseline_respects_missing_policy():
    baseline = MonitoringBaseline().fit([0., 1., np.nan])
    table = psi_table(None, [0., 1., np.nan], baseline=baseline, include_missing=False)
    assert table["期望样本数"].sum() == table["实际样本数"].sum() == 2
    assert table["PSI贡献"].sum() == 0


@pytest.mark.parametrize("offset", [0, 1e10, 1e12])
def test_continuous_dispersion_is_translation_invariant(offset):
    y = np.array([0., 1., 2., 4., 6., 8.]) + offset
    bins = np.array([0, 0, 0, -1, -1, -1])
    table = compute_bin_stats(bins, y, target_type="continuous", round_digits=False)
    np.testing.assert_allclose(table["目标值标准差"], [np.std(y[:3]), np.std(y[3:])], rtol=1e-12)
    np.testing.assert_array_equal(table["目标值最小值"], [y[0], y[3]])
    np.testing.assert_array_equal(table["目标值最大值"], [y[2], y[5]])


@pytest.mark.parametrize("factory", [OptimalBinning, QuantileBinning, TreeBinning])
def test_unsupported_binning_weights_are_not_silently_discarded(factory):
    x = pd.DataFrame({"x": np.arange(20)})
    y = np.arange(20) % 2
    for weights in ([-1.], np.ones(20)):
        with pytest.raises(ValueError, match="sample_weight"):
            factory(n_jobs=1).fit(x, y, sample_weight=weights)


def test_accumulator_rejects_different_bin_identity():
    a = BinStatsAccumulator(feature="x", target="bad", binning_version="v1", strict_identity=True).update([0], [0])
    b = BinStatsAccumulator(feature="x", target="bad", binning_version="v2", strict_identity=True).update([0], [1])
    with pytest.raises(ValueError, match="分箱"):
        a.merge(b)
    assert a.rows_seen_ == 1


def test_refitting_ambiguous_legacy_baseline_clears_migration_block():
    baseline = MonitoringBaseline(binning_params={'user_splits': [['A'], ['B']], 'handle_unknown': -1}).fit(['A', 'B', None])
    old_state = dict(baseline.__dict__)
    old_state.pop('schema_version_')
    migrated = MonitoringBaseline()
    migrated.__setstate__(old_state)
    assert migrated._legacy_missing_ambiguous_ is True
    with pytest.raises(ValueError, match='重新拟合'):
        migrated.evaluate(['A', 'B'])
    migrated.fit(['A', 'B', None])
    assert migrated._legacy_missing_ambiguous_ is False
    result = migrated.evaluate(['A', 'C', None], include_missing=False)
    assert result['状态'] == '成功'
    assert result['实际缺失率'] == result['未知类别率'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('reference', [['A', 'B'], ['A']])
def test_special_unknown_bin_never_uses_negative_category_index(reference):
    baseline = MonitoringBaseline(binning_params={'handle_unknown': -2}).fit(reference)
    result = baseline.evaluate(['C'])
    table = result['分箱明细']
    current = table.loc[table['实际样本数'] == 1]
    assert current['分箱'].tolist() == ['特殊值及并入类别']
    assert result['未知类别率'] == 1


@pytest.mark.parametrize('invalid', [-4, -100])
def test_monitoring_rejects_unsupported_negative_unknown_bins(invalid):
    with pytest.raises(ValueError, match='未知类别'):
        MonitoringBaseline(binning_params={'handle_unknown': invalid}).fit(['A', 'B'])


@pytest.mark.parametrize('expected,actual', [([np.nan], [1.]), ([1.], [np.nan]), ([np.nan], [np.nan]), ([1.], [])])
def test_empty_psi_fallback_still_validates_configuration(expected, actual):
    with pytest.raises(ValueError, match='method'):
        psi_table(expected, actual, method='not-a-method')
    with pytest.raises(ValueError, match='not_a_parameter'):
        psi_table(expected, actual, method='uniform', not_a_parameter=1)
    with pytest.raises(ValueError, match='min_bin_size'):
        psi_table(expected, actual, method='uniform', min_bin_size=-1)
