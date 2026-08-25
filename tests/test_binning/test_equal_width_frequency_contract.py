"""等距与等频分箱的切点、箱数及公开报表契约回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning, QuantileBinning, UniformBinning
from hscredit.report import feature_bin_stats


def _make_skewed_dpd_data() -> pd.DataFrame:
    """构造零值集中、正值稀疏的逾期天数型特征。"""
    return pd.DataFrame(
        {
            "特征": np.r_[np.zeros(80), np.arange(1, 21, dtype=float)],
            "DPD": np.resize([0, 3, 8, 10], 100),
        }
    )


def test_uniform_wrapper_does_not_merge_equal_width_intervals():
    """通用包装层不能按最小箱占比二次合并等距切点。"""
    data = _make_skewed_dpd_data()
    target = (data["DPD"] > 7).astype(int)
    params = {"max_n_bins": 5, "min_bin_size": None, "n_jobs": 1}

    direct = UniformBinning(**params).fit(data[["特征"]], target)
    wrapped = OptimalBinning(method="uniform", **params).fit(data[["特征"]], target)

    expected_splits = np.array([4.0, 8.0, 12.0, 16.0])
    np.testing.assert_array_equal(direct.splits_["特征"], expected_splits)
    np.testing.assert_array_equal(wrapped.splits_["特征"], expected_splits)
    assert wrapped.n_bins_["特征"] == 5
    assert wrapped.transform(data[["特征"]], metric="indices")["特征"].value_counts(sort=False).to_dict() == {
        0: 83,
        1: 4,
        2: 4,
        3: 4,
        4: 5,
    }


def test_uniform_min_bin_size_merges_after_initial_equal_width_split():
    """等距切分完成后再应用最小箱约束，允许合并为宽度不等的区间。"""
    data = _make_skewed_dpd_data()
    target = (data["DPD"] > 7).astype(int)

    direct = UniformBinning(
        max_n_bins=5,
        min_bin_size=0.05,
        n_jobs=1,
    ).fit(data[["特征"]], target)
    wrapped = OptimalBinning(
        method="uniform",
        max_n_bins=5,
        min_bin_size=0.05,
        n_jobs=1,
    ).fit(data[["特征"]], target)

    np.testing.assert_array_equal(direct.splits_["特征"], [4.0, 8.0, 12.0, 16.0])
    np.testing.assert_array_equal(wrapped.splits_["特征"], [4.0, 16.0])
    assert wrapped.transform(data[["特征"]], metric="indices")["特征"].value_counts(sort=False).to_dict() == {
        0: 83,
        1: 12,
        2: 5,
    }


def test_quantile_discrete_values_do_not_create_an_empty_first_bin():
    """离散分位点必须位于相邻取值之间，不能把最小值本身作为左闭切点。"""
    data = pd.DataFrame({"特征": np.repeat(np.arange(5, dtype=float), 20)})
    target = pd.Series([0, 1] * 50)

    binner = QuantileBinning(max_n_bins=5, min_bin_size=0.05, n_jobs=1).fit(data, target)

    np.testing.assert_allclose(binner.splits_["特征"], [0.5, 1.5, 2.5, 3.5])
    assert binner.n_bins_["特征"] == 5
    assert binner.transform(data, metric="indices")["特征"].value_counts(sort=False).to_dict() == {
        0: 20,
        1: 20,
        2: 20,
        3: 20,
        4: 20,
    }


def test_uniform_supports_more_than_ten_bins_when_data_allows():
    """max_n_bins 大于10时不能被未公开的固定上限截断。"""
    data = pd.DataFrame({"特征": np.arange(1000, dtype=float)})
    target = pd.Series([0, 1] * 500)

    binner = UniformBinning(max_n_bins=20, min_bin_size=None, n_jobs=1).fit(data, target)

    assert binner.n_bins_["特征"] == 20
    assert len(binner.splits_["特征"]) == 19
    assert binner.transform(data, metric="indices")["特征"].nunique() == 20


def test_uniform_none_keeps_empty_intervals_in_the_learned_rule_space():
    """不限制最小样本数时应保留等距空箱对应的切点。"""
    data = pd.DataFrame({"特征": [0.0] * 50 + [10.0] * 50})
    target = pd.Series([0, 1] * 50)

    binner = OptimalBinning(
        method="uniform",
        max_n_bins=5,
        min_bin_size=None,
        n_jobs=1,
    ).fit(data, target)

    np.testing.assert_array_equal(binner.splits_["特征"], [2.0, 4.0, 6.0, 8.0])
    assert binner.n_bins_["特征"] == 5
    assert binner.transform(data, metric="indices")["特征"].value_counts(sort=False).to_dict() == {0: 50, 4: 50}


def test_quantile_none_disables_minimum_bin_size_constraint():
    """显式传入 None 时应保留原始等频目标箱数。"""
    data = pd.DataFrame({"特征": np.arange(1000, dtype=float)})
    target = pd.Series([0, 1] * 500)

    binner = QuantileBinning(max_n_bins=200, min_bin_size=None, n_jobs=1).fit(data, target)

    assert binner.n_bins_["特征"] == 200
    assert len(binner.splits_["特征"]) == 199
    assert binner.transform(data, metric="indices")["特征"].value_counts().sort_index().tolist() == [5] * 200


@pytest.mark.parametrize(
    "method",
    [
        "uniform",
        "quantile",
        "tree",
        "chi",
        "best_ks",
        "best_iv",
        "mdlp",
        "cart",
        "kmeans",
        "monotonic",
        "genetic",
        "smooth",
        "kernel_density",
        "best_lift",
        "target_bad_rate",
    ],
)
def test_public_binning_methods_accept_no_minimum_sample_constraint(method):
    """统一入口显式传入 min_bin_size=None 时，各内置方法都应可完成拟合。"""
    values = np.linspace(0, 1, 120)
    data = pd.DataFrame({"特征": values})
    target = pd.Series((values > 0.55).astype(int))

    binner = OptimalBinning(
        method=method,
        max_n_bins=3,
        min_bin_size=None,
        lift_refine=False,
        random_state=1,
        n_jobs=1,
    ).fit(data, target)

    assert binner._is_fitted
    assert "特征" in binner.splits_


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_unsupervised_wrapper_preserves_woe_clip_through_transactional_fit(method):
    """事务候选必须保留无监督底层分箱器的 WOE 截断参数。"""
    data = pd.DataFrame({"特征": np.arange(100, dtype=float)})
    target = pd.Series(np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)])

    binner = OptimalBinning(
        method=method,
        min_n_bins=2,
        max_n_bins=2,
        min_bin_size=0.01,
        woe_clip=0.1,
        n_jobs=1,
    ).fit(data, target)

    ordinary_woe = binner.bin_tables_["特征"].loc[lambda frame: frame["分箱"] >= 0, "分档WOE值"]
    assert ordinary_woe.abs().le(0.1).all()


@pytest.mark.parametrize(
    ("method", "expected_rules", "expected_counts"),
    [
        pytest.param("uniform", [4.0, 8.0, 12.0, 16.0], [83, 4, 4, 4, 5], id="等距"),
        pytest.param("quantile", [0.2], [80, 20], id="等频重复值"),
    ],
)
def test_feature_bin_stats_preserves_unsupervised_binning_contract(method, expected_rules, expected_counts):
    """多 DPD 长表和合计行不能改变无监督分箱器的切点或实际箱数。"""
    data = _make_skewed_dpd_data()

    table, rules = feature_bin_stats(
        data,
        "特征",
        overdue="DPD",
        dpds=[7, 3, 0],
        method=method,
        max_n_bins=5,
        margins=True,
        long_format=True,
        return_rules=True,
        n_jobs=1,
    )

    assert rules["特征"] == pytest.approx(expected_rules)
    assert table["逾期标签"].drop_duplicates().tolist() == ["DPD_7+", "DPD_3+", "DPD_0+"]
    for _, target_table in table.groupby("逾期标签", sort=False):
        ordinary = target_table[target_table["分箱标签"] != "合计"]
        assert ordinary["样本总数"].tolist() == expected_counts
        assert ordinary["样本总数"].gt(0).all()
        assert ordinary["样本总数"].sum() == len(data)


def test_feature_bin_stats_defaults_to_one_percent_minimum_bin_size():
    """公开报表默认1%最小箱占比时，50个等频箱不应按旧5%默认值继续合并。"""
    data = pd.DataFrame(
        {
            "特征": np.arange(1000, dtype=float),
            "目标": [0, 1] * 500,
        }
    )

    table, rules = feature_bin_stats(
        data,
        "特征",
        target="目标",
        method="quantile",
        max_n_bins=50,
        return_rules=True,
        n_jobs=1,
    )

    assert len(rules["特征"]) == 49
    assert len(table) == 50
    assert table["样本总数"].tolist() == [20] * 50
