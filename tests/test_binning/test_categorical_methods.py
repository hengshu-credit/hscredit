"""全部直接分箱器对类别有序编码的行为测试。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import (
    BestIVBinning,
    BestKSBinning,
    BestLiftBinning,
    CPSATBinning,
    CartBinning,
    ChiMergeBinning,
    GeneticBinning,
    KMeansBinning,
    KernelDensityBinning,
    MDLPBinning,
    MonotonicBinning,
    ORBinning,
    OptimalBinning,
    QuantileBinning,
    SmoothBinning,
    TargetBadRateBinning,
    TreeBinning,
    UniformBinning,
)


DIRECT_BINNER_CLASSES = [
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    ORBinning,
    CPSATBinning,
    CartBinning,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
]

METHOD_NAMES = [
    "uniform",
    "quantile",
    "tree",
    "chi",
    "best_ks",
    "best_iv",
    "mdlp",
    "or_tools",
    "cp_sat",
    "cart",
    "kmeans",
    "monotonic",
    "genetic",
    "smooth",
    "kernel_density",
    "best_lift",
    "target_bad_rate",
]


def _make_category_data():
    specifications = [
        ("A", 40, 2),
        ("B", 30, 6),
        ("C", 50, 15),
        ("D", 20, 10),
        ("E", 60, 42),
        ("F", 40, 36),
    ]
    categories = []
    target = []
    for category, count, bad_count in specifications:
        categories.extend([category] * count)
        target.extend([1] * bad_count + [0] * (count - bad_count))
    return pd.DataFrame({"category": categories}), pd.Series(target, name="target")


def _make_binner(binner_cls, category_order=None):
    kwargs = {
        "max_n_bins": 3,
        "min_n_bins": 2,
        "min_bin_size": 0.01,
        "random_state": 7,
    }
    if category_order is not None:
        kwargs["category_order"] = {"category": category_order}
    if binner_cls is GeneticBinning:
        kwargs.update(population_size=12, generations=4)
    elif binner_cls is ORBinning:
        kwargs.update(time_limit=2, n_prebins=8, max_candidates=20)
    elif binner_cls is CPSATBinning:
        kwargs.update(time_limit=2, n_prebins=8, max_candidates=20)
    elif binner_cls is KernelDensityBinning:
        kwargs.update(n_grid_points=128)
    elif binner_cls is SmoothBinning:
        kwargs.update(n_prebins=12)
    elif binner_cls is BestLiftBinning:
        kwargs.update(n_prebins=12, max_bin_size=None)
    return binner_cls(**kwargs)


def _groups_from_numeric_transform(binner, ordered_categories):
    codes = pd.DataFrame({"category": np.arange(len(ordered_categories), dtype=float)})
    indices = binner.transform(codes, metric="indices")["category"].to_numpy(dtype=int)
    return [
        [category for category, bin_index in zip(ordered_categories, indices) if bin_index == expected_index]
        for expected_index in sorted(set(indices.tolist()))
        if expected_index >= 0
    ]


def _make_optimal_binner(method, order):
    kwargs = {
        "method": method,
        "max_n_bins": 3,
        "min_n_bins": 2,
        "min_bin_size": 0.01,
        "random_state": 7,
        "category_order": {"category": order},
        "lift_refine": False,
    }
    if method == "genetic":
        kwargs.update(population_size=12, generations=4)
    elif method == "or_tools":
        kwargs.update(or_time_limit=2, n_prebins=8, max_candidates=20)
    elif method == "cp_sat":
        kwargs.update(cp_sat_time_limit=2, cp_sat_n_prebins=8, max_candidates=20)
    elif method == "kernel_density":
        kwargs.update(n_grid_points=128)
    elif method == "smooth":
        kwargs.update(n_prebins=12)
    elif method == "best_lift":
        kwargs.update(n_prebins=12, max_bin_size=None)
    return OptimalBinning(**kwargs)


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES, ids=lambda cls: cls.__name__)
def test_direct_binner_uses_its_numeric_algorithm_after_category_ordering(binner_cls):
    """防止直接分箱器绕过自身数值算法，退化为统一类别合并。"""
    X, y = _make_category_data()
    order = ["A", "B", "C", "D", "E", "F"]

    numeric = _make_binner(binner_cls)
    numeric.fit(pd.DataFrame({"category": X["category"].map(dict(zip(order, range(len(order)))))}), y)
    expected_groups = _groups_from_numeric_transform(numeric, order)

    categorical = _make_binner(binner_cls, category_order=order)
    categorical.fit(X, y)
    actual_groups = categorical.export_rules()["category"]

    assert actual_groups == expected_groups
    assert categorical.feature_types_["category"] == "categorical"
    assert categorical.n_bins_["category"] == len(actual_groups)
    assert categorical.get_bin_table("category")["样本总数"].sum() == len(X)


def test_different_methods_keep_different_category_boundaries():
    """防止 OptimalBinning 或共享层再次把不同方法覆盖成同一套类别组。"""
    X, y = _make_category_data()
    order = ["A", "B", "C", "D", "E", "F"]
    methods = [UniformBinning, QuantileBinning, TreeBinning, ChiMergeBinning, BestIVBinning, MDLPBinning]

    rules = []
    for binner_cls in methods:
        binner = _make_binner(binner_cls, category_order=order).fit(X, y)
        rules.append(binner.export_rules()["category"])

    normalized = {repr(rule) for rule in rules}
    assert len(normalized) >= 3


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES, ids=lambda cls: cls.__name__)
def test_direct_binner_transform_uses_fitted_category_groups(binner_cls):
    """防止直接分箱器在 transform 时重新按局部 Categorical codes 编码。"""
    X, y = _make_category_data()
    order = ["A", "B", "C", "D", "E", "F"]
    binner = _make_binner(binner_cls, category_order=order).fit(X, y)

    for expected_index, group in enumerate(binner.export_rules()["category"]):
        transformed = binner.transform(pd.DataFrame({"category": group}), metric="indices")["category"]
        assert transformed.tolist() == [expected_index] * len(group)


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES, ids=lambda cls: cls.__name__)
def test_direct_binner_transform_reserves_unknown_category_index(binner_cls):
    """防止任一直接分箱器把预测期未知类别映射为普通箱。"""
    X, y = _make_category_data()
    order = ["A", "B", "C", "D", "E", "F"]
    binner = _make_binner(binner_cls, category_order=order).fit(X, y)

    transformed = binner.transform(pd.DataFrame({"category": ["UNSEEN"]}), metric="indices")

    assert transformed.iloc[0, 0] == -3


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_optimal_binning_forwards_explicit_order_to_every_method(method):
    """防止统一入口只对部分 method 传递 category_order。"""
    X, y = _make_category_data()
    order = ["F", "E", "D", "C", "B", "A"]

    binner = _make_optimal_binner(method, order).fit(X, y)
    flattened = [category for group in binner.export_rules()["category"] for category in group]

    assert flattened == order


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_optimal_binning_forwards_category_detection_and_missing_policy(method):
    """防止统一入口只对部分 method 传递 cat_cutoff 或 missing_separate。"""
    X, y = _make_category_data()
    encoded_order = [0, 1, 2, 3, 4, 5]
    encoded = X.assign(category=X["category"].map(dict(zip(["A", "B", "C", "D", "E", "F"], encoded_order))))
    kwargs = {
        "method": method,
        "max_n_bins": 3,
        "min_n_bins": 2,
        "min_bin_size": 0.01,
        "random_state": 7,
        "cat_cutoff": 10,
        "missing_separate": False,
        "category_order": {"category": encoded_order},
        "lift_refine": False,
    }
    if method == "genetic":
        kwargs.update(population_size=12, generations=4)
    elif method == "or_tools":
        kwargs.update(or_time_limit=2, n_prebins=8, max_candidates=20)
    elif method == "cp_sat":
        kwargs.update(cp_sat_time_limit=2, cp_sat_n_prebins=8, max_candidates=20)
    elif method == "kernel_density":
        kwargs.update(n_grid_points=128)
    elif method == "smooth":
        kwargs.update(n_prebins=12)
    elif method == "best_lift":
        kwargs.update(n_prebins=12, max_bin_size=None)

    binner = OptimalBinning(**kwargs).fit(encoded, y)

    assert binner.feature_types_["category"] == "categorical"
    assert binner._binner.cat_cutoff == 10
    assert binner._binner.missing_separate is False


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES, ids=lambda cls: cls.__name__)
def test_direct_binner_exposes_all_common_category_parameters(binner_cls):
    """所有直接分箱器都应通过 sklearn get_params 暴露公共类别参数。"""
    params = binner_cls().get_params(deep=False)

    assert {
        "cat_cutoff",
        "category_order",
        "handle_unknown",
        "missing_separate",
        "max_bin_size",
        "min_bad_rate",
        "monotonic",
    }.issubset(params)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx").exists(),
    reason="缺少 examples/hscredit_yyp.xlsx",
)
def test_yyp_category_explicit_order_and_custom_missing_groups():
    """在约定的商品类别字段上固化显式顺序及两种缺失值自定义分组。"""
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    df = pd.read_excel(path)
    X = df[["商品类别"]]
    y = df["FPD"]
    category_order = X["商品类别"].dropna().drop_duplicates().tolist()

    explicit = OptimalBinning(
        method="uniform",
        min_n_bins=1,
        max_n_bins=5,
        min_bin_size=1,
        category_order={"商品类别": category_order},
    ).fit(X, y)
    assert explicit._category_orders_["商品类别"] == category_order

    first_cut = max(1, len(category_order) // 3)
    second_cut = max(first_cut + 1, 2 * len(category_order) // 3)
    chunks = [category_order[:first_cut], category_order[first_cut:second_cut], category_order[second_cut:]]
    assert all(chunks)
    missing_alone = [*chunks, [np.nan]]
    alone = OptimalBinning(
        user_splits={"商品类别": missing_alone},
        strict_user_splits=True,
        min_n_bins=1,
        max_n_bins=4,
        min_bin_size=1,
    ).fit(X, y)
    assert alone.transform(pd.DataFrame({"商品类别": [np.nan]}), metric="indices").iloc[0, 0] == 3

    missing_mixed = [chunks[0], chunks[1], [*chunks[2], np.nan]]
    mixed = OptimalBinning(
        user_splits={"商品类别": missing_mixed},
        strict_user_splits=True,
        missing_separate=False,
        min_n_bins=1,
        max_n_bins=3,
        min_bin_size=1,
    ).fit(X, y)
    assert mixed.transform(pd.DataFrame({"商品类别": [np.nan]}), metric="indices").iloc[0, 0] == 2
