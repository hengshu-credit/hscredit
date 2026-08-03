"""类别变量有序编码适配器的回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning


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
