"""Notebook 中单逾期标签分箱统计的回归测试。"""

import numpy as np
import pandas as pd

from hscredit.report import feature_bin_stats


def _make_overdue_data():
    rng = np.random.RandomState(42)
    sample_count = 1000
    return pd.DataFrame(
        {
            "score": rng.randn(sample_count) * 50 + 600,
            "MOB1": rng.choice([0, 1, 2, 3, 5, 7, 10, 15, 30, 60], sample_count),
        }
    )


def test_single_overdue_feature_bin_stats_uses_flat_columns():
    table = feature_bin_stats(
        data=_make_overdue_data(),
        feature="score",
        overdue="MOB1",
        dpds=7,
        del_grey=False,
        method="quantile",
        max_n_bins=5,
        verbose=0,
    )

    assert not isinstance(table.columns, pd.MultiIndex)
    assert {"分箱标签", "样本总数", "坏样本数", "坏样本率", "分档WOE值"}.issubset(table.columns)


def test_single_overdue_feature_bin_stats_supports_grey_customer_filtering():
    data = _make_overdue_data()
    table = feature_bin_stats(
        data=data,
        feature="score",
        overdue="MOB1",
        dpds=7,
        del_grey=True,
        method="quantile",
        max_n_bins=5,
        verbose=0,
    )

    assert 0 < table["样本总数"].sum() < len(data)
