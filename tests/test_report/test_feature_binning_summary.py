"""feature_binning_summary 的回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning
from hscredit.report import feature_bin_stats, feature_binning_summary
from hscredit.report import feature_analyzer


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    size = 400
    return pd.DataFrame(
        {
            '特征1': rng.normal(size=size),
            '特征2': rng.uniform(size=size),
            'MOB1': rng.choice([0, 1, 2, 4, 8], size=size),
        }
    )


def test_feature_binning_summary_returns_tables_and_multi_target_summary(sample_data):
    tables, summary = feature_binning_summary(
        sample_data,
        feature=['特征1', '特征2'],
        methods=['quantile', 'mdlp'],
        overdue='MOB1',
        dpds=[3, 1, 0],
        max_n_bins=4,
        margins=True,
    )

    assert list(tables) == ['特征1', '特征2']
    assert list(tables['特征1']) == ['quantile', 'mdlp']
    assert tables['特征1']['quantile'].iloc[-1]['分箱详情', '分箱标签'] == '合计'
    assert list(summary[('分箱详情', '分箱方法')]) == ['quantile', 'quantile', 'mdlp', 'mdlp']
    assert list(summary[('分箱详情', '指标名称')]) == ['特征1', '特征2', '特征1', '特征2']
    assert list(dict.fromkeys(summary.columns.get_level_values(0))) == [
        '分箱详情', '分档KS值', 'LIFT值', '指标IV值', '坏样本数', '坏样本率'
    ]
    assert ('坏样本率', 'MOB1@3') in summary.columns

    table = tables['特征1']['quantile']
    valid = table[table[('分箱详情', '分箱标签')] != '合计']
    expected_bad = valid[('MOB1_3+', '坏样本数')].sum()
    expected_total = valid[('分箱详情', '样本总数')].sum()
    row = summary[
        (summary[('分箱详情', '分箱方法')] == 'quantile')
        & (summary[('分箱详情', '指标名称')] == '特征1')
    ].iloc[0]
    assert row[('坏样本数', 'MOB1@3')] == expected_bad
    assert row[('坏样本率', 'MOB1@3')] == pytest.approx(expected_bad / expected_total)
    assert row[('分档KS值', 'MOB1@3')] == valid[('MOB1_3+', '分档KS值')].max()


def test_feature_binning_summary_bin_params_override_common_params(sample_data):
    tables, _ = feature_binning_summary(
        sample_data,
        feature='特征1',
        methods=['quantile', 'mdlp'],
        overdue='MOB1',
        dpds=3,
        max_n_bins=6,
        bin_params={
            'quantile': {'max_n_bins': 3},
            'mdlp': {'max_n_bins': 4},
        },
    )

    assert len(tables['特征1']['quantile']) <= 3
    assert len(tables['特征1']['mdlp']) <= 4


def test_feature_binning_summary_single_level_bin_params_apply_to_all(sample_data):
    tables, summary = feature_binning_summary(
        sample_data,
        feature='特征1',
        methods=['quantile', 'uniform'],
        overdue='MOB1',
        dpds=3,
        bin_params={'max_n_bins': 3},
    )

    assert all(len(table) <= 3 for table in tables['特征1'].values())
    assert ('坏样本率', 'MOB1@3') in summary.columns


def test_feature_binning_summary_uses_target_name_for_single_target(sample_data):
    sample_data['是否坏样本'] = (sample_data['MOB1'] > 3).astype(int)
    _, summary = feature_binning_summary(
        sample_data,
        feature='特征1',
        methods='quantile',
        target='是否坏样本',
        max_n_bins=3,
    )

    assert ('分档KS值', '是否坏样本') in summary.columns
    assert ('坏样本率', '是否坏样本') in summary.columns


def test_feature_binning_summary_rejects_invalid_method(sample_data):
    with pytest.raises(ValueError, match='不支持的methods'):
        feature_binning_summary(sample_data, '特征1', methods='not-exists', target='MOB1')


def test_feature_binning_summary_parameter_priority(monkeypatch, sample_data):
    calls = []

    def fake_feature_bin_stats(data, feature, method, **params):
        calls.append((feature, method, params))
        return pd.DataFrame(
            {
                '分箱标签': ['低', '高'],
                '样本总数': [200, 200],
                '坏样本数': [20, 40],
                '坏样本率': [0.1, 0.2],
                '分档KS值': [0.1, 0.2],
                'LIFT值': [0.8, 1.2],
                '指标IV值': [0.3, 0.3],
            }
        )

    monkeypatch.setattr(feature_analyzer, 'feature_bin_stats', fake_feature_bin_stats)
    feature_binning_summary(
        sample_data,
        feature='特征1',
        methods=['quantile', 'mdlp'],
        target='MOB1',
        max_n_bins=8,
        prebinning_params={'max_n_bins': 50},
        bin_params={'mdlp': {'max_n_bins': 3}},
        lift_refine=False,
    )

    quantile_params = calls[0][2]
    mdlp_params = calls[1][2]
    assert quantile_params['max_n_bins'] == 8
    assert mdlp_params['max_n_bins'] == 3
    assert quantile_params['prebinning_params'] == {'max_n_bins': 50}
    assert mdlp_params['lift_refine'] is False


@pytest.mark.parametrize("method", OptimalBinning.VALID_METHODS)
def test_feature_bin_stats_supports_categorical_feature_for_all_methods(method):
    rng = np.random.default_rng(42)
    size = 300
    data = pd.DataFrame(
        {
            '商品类别': rng.choice(['礼包', '珠宝首饰', '家用电器', '智能设备', '电脑数码'], size=size),
            'FPD': rng.choice([0, 1], p=[0.82, 0.18], size=size),
        }
    )

    table = feature_bin_stats(
        data,
        feature='商品类别',
        target='FPD',
        method=method,
        max_n_bins=5,
        min_bin_size=0.05,
    )

    assert not table.empty
    assert '分箱标签' in table.columns
    assert table['样本总数'].sum() == size
    assert not table['分箱标签'].astype(str).str.startswith('bin_').all()
