"""类别变量有序编码适配器的回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning, TreeBinning, UniformBinning


def test_default_category_order_uses_bad_rate_and_first_seen_ties():
    """防止默认排序忽略坏样本率相同时的首次出现顺序。"""
    X = pd.DataFrame({"grade": ["B", "A", "C", "B", "A", "C"]})
    y = pd.Series([0, 0, 1, 1, 1, 1], name="target")

    binner = OptimalBinning(method="uniform", min_n_bins=1, max_n_bins=3).fit(X, y)

    assert binner._category_orders_["grade"] == ["B", "A", "C"]


def test_explicit_category_order_is_saved_without_string_coercion():
    """防止用户顺序被忽略，或数值 1 与字符串 '1' 被合并。"""
    X = pd.DataFrame({"mixed": pd.Series([1, "1", 2, 1, "1", 2], dtype=object)})
    y = pd.Series([0, 1, 1, 0, 1, 0], name="target")
    expected = [2, "1", 1]

    binner = OptimalBinning(
        method="uniform",
        min_n_bins=1,
        max_n_bins=3,
        category_order={"mixed": expected},
    ).fit(X, y)

    actual = binner._category_orders_["mixed"]
    assert actual == expected
    assert type(actual[1]) is str
    assert type(actual[2]) is int


def test_nullable_boolean_dtype_is_categorical():
    """防止 pandas BooleanDtype 被误判为数值变量。"""
    values = pd.Series([True, False, pd.NA], dtype="boolean")

    assert OptimalBinning()._detect_feature_type(values) == "categorical"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_n_bins": 0},
        {"min_n_bins": 3, "max_n_bins": 2},
        {"min_bin_size": 0},
        {"max_bin_size": -0.1},
        {"min_bad_rate": -0.1},
        {"min_bad_rate": 1.1},
        {"cat_cutoff": 0},
        {"woe_clip": -1},
        {"handle_unknown": "ignore"},
    ],
)
def test_invalid_common_parameters_raise_value_error(kwargs):
    """防止非法公共参数被静默接受并在拟合后产生无效规则。"""
    with pytest.raises(ValueError):
        OptimalBinning(**kwargs)


def test_bad_rate_order_excludes_missing_and_special_codes():
    """防止缺失值和特殊值被错误加入普通类别排序。"""
    X = pd.DataFrame({"grade": ["A", "B", np.nan, "SPECIAL", "A", "B"]})
    y = pd.Series([0, 1, 1, 0, 0, 1], name="target")

    binner = OptimalBinning(
        method="uniform",
        min_n_bins=1,
        max_n_bins=2,
        special_codes=["SPECIAL"],
    ).fit(X, y)

    assert binner._category_orders_["grade"] == ["A", "B"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OptimalBinning(method="uniform", **kwargs),
        lambda **kwargs: UniformBinning(**kwargs),
    ],
    ids=["OptimalBinning", "UniformBinning"],
)
def test_unknown_category_has_reserved_index_label_and_neutral_woe(factory):
    """防止预测期未知类别静默落入第 0 箱或产生 NaN WOE。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "B"]})
    y = pd.Series([0, 1, 0, 1], name="target")
    binner = factory(min_n_bins=2, max_n_bins=2).fit(X, y)
    unseen = pd.DataFrame({"city": ["C"]})

    assert binner.transform(unseen, metric="indices").iloc[0, 0] == -3
    assert binner.transform(unseen, metric="bins").iloc[0, 0] == "unknown"
    assert binner.transform(unseen, metric="woe").iloc[0, 0] == 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OptimalBinning(method="uniform", **kwargs),
        lambda **kwargs: UniformBinning(**kwargs),
    ],
    ids=["OptimalBinning", "UniformBinning"],
)
def test_handle_unknown_error_reports_feature_and_value(factory):
    """防止 handle_unknown='error' 在转换时被忽略。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "B"]})
    y = pd.Series([0, 1, 0, 1], name="target")
    binner = factory(min_n_bins=2, max_n_bins=2, handle_unknown="error").fit(X, y)

    with pytest.raises(ValueError, match="city.*C"):
        binner.transform(pd.DataFrame({"city": ["C"]}), metric="indices")


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OptimalBinning(method="uniform", **kwargs),
        lambda **kwargs: UniformBinning(**kwargs),
    ],
    ids=["OptimalBinning", "UniformBinning"],
)
def test_native_int_and_string_categories_transform_to_different_bins(factory):
    """防止 transform 阶段通过字符串转换混淆原生数值和字符串类别。"""
    X = pd.DataFrame({"value": pd.Series([1, "1", 1, "1"], dtype=object)})
    y = pd.Series([0, 1, 0, 1], name="target")
    binner = factory(
        min_n_bins=2,
        max_n_bins=2,
        category_order={"value": [1, "1"]},
    ).fit(X, y)

    transformed = binner.transform(
        pd.DataFrame({"value": pd.Series([1, "1"], dtype=object)}),
        metric="indices",
    )["value"]

    assert transformed.tolist() == [0, 1]


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OptimalBinning(method="uniform", **kwargs),
        lambda **kwargs: UniformBinning(**kwargs),
    ],
    ids=["OptimalBinning", "UniformBinning"],
)
def test_missing_and_special_codes_keep_reserved_indices(factory):
    """防止共享类别匹配覆盖缺失箱和特殊值箱索引。"""
    X = pd.DataFrame({"city": ["A", "B", np.nan, "SPECIAL", "A", "B"]})
    y = pd.Series([0, 1, 1, 0, 0, 1], name="target")
    binner = factory(
        min_n_bins=2,
        max_n_bins=2,
        special_codes=["SPECIAL"],
    ).fit(X, y)

    indices = binner.transform(pd.DataFrame({"city": [np.nan, "SPECIAL"]}), metric="indices")["city"]

    assert indices.tolist() == [-1, -2]


def _make_category_constraint_data():
    categories = []
    targets = []
    for category, count, bad_count in [
        ("A", 10, 1),
        ("B", 15, 3),
        ("C", 20, 8),
        ("D", 25, 15),
        ("E", 30, 24),
    ]:
        categories.extend([category] * count)
        targets.extend([1] * bad_count + [0] * (count - bad_count))
    return pd.DataFrame({"category": categories}), pd.Series(targets, name="target")


def test_feasible_category_max_bin_size_is_enforced():
    """防止类别恢复后丢失数值算法已经执行的 max_bin_size 约束。"""
    X, y = _make_category_constraint_data()

    binner = OptimalBinning(
        method="best_iv",
        min_n_bins=2,
        max_n_bins=4,
        max_bin_size=0.30,
    ).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")

    assert ordinary["样本占比"].max() <= 0.30 + 1e-12


def test_infeasible_single_category_max_size_raises_clear_error():
    """单个原子类别已超限时必须明确报错，不能伪装成约束已生效。"""
    X = pd.DataFrame({"category": ["A"] * 40 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20})
    y = pd.Series(([0, 1] * 20) + ([0, 1] * 30), name="target")

    with pytest.raises(ValueError, match="category.*max_bin_size"):
        OptimalBinning(
            method="uniform",
            min_n_bins=2,
            max_n_bins=4,
            max_bin_size=0.30,
        ).fit(X, y)


def test_explicit_descending_category_order_keeps_descending_bad_rates():
    """防止恢复类别规则后又被旧的升序合并逻辑改写。"""
    X, y = _make_category_constraint_data()
    order = ["E", "D", "C", "B", "A"]

    binner = OptimalBinning(
        method="tree",
        min_n_bins=2,
        max_n_bins=4,
        monotonic="descending",
        category_order={"category": order},
    ).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")

    assert np.all(np.diff(ordinary["坏样本率"].to_numpy(dtype=float)) <= 1e-12)


def test_category_min_bin_size_and_min_bad_rate_are_enforced():
    """防止类别还原后丢失最小箱样本量和最小坏样本率约束。"""
    X, y = _make_category_constraint_data()

    binner = OptimalBinning(
        method="best_iv",
        min_n_bins=2,
        max_n_bins=4,
        min_bin_size=0.15,
        min_bad_rate=0.15,
    ).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")

    assert ordinary["样本总数"].min() >= 15
    assert ordinary["坏样本率"].min() >= 0.15 - 1e-12


def test_tree_binning_sparse_category_never_uses_zero_min_samples_leaf():
    """防止极稀疏类别把比例型 min_samples_leaf 向下取整为 0。"""
    X = pd.DataFrame({"category": ["A", np.nan, np.nan, np.nan]})
    y = pd.Series([1, 0, 1, 0], name="target")

    binner = TreeBinning(min_n_bins=1, max_n_bins=2, min_samples_leaf=0.05).fit(X, y)

    assert binner.export_rules()["category"] == [["A"]]
