"""全部直接分箱器对类别有序编码的行为测试。"""

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
