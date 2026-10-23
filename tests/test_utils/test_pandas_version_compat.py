"""pandas apply 的显式版本边界与原生语义回归。"""

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

import hscredit  # noqa: F401
from hscredit._compat import pandas_apply_api


@pytest.mark.parametrize(
    ("version", "api"),
    [
        ("2.0.0", "2.0"),
        ("2.0.3", "2.0"),
        ("2.1.0", "2.1"),
        ("2.1.4", "2.1"),
        ("2.2.0", "2.2"),
        ("2.3.3", "2.2"),
        ("3.0.0", "3.0"),
        ("3.0.6", "3.0"),
        ("3.1.0", "3.0"),
    ],
)
def test_apply_version_boundaries(version, api):
    assert pandas_apply_api(Version(version)) == api


def test_apply_rejects_unsupported_pandas_version():
    with pytest.raises(ValueError, match="pandas>=2.0"):
        pandas_apply_api(Version("1.5.3"))


def test_convert_dtype_matches_native_control_or_udf_keyword():
    """2.x 的控制参数在 3.x 成为 UDF 参数，不得继续吞掉或访问已删除属性。"""
    series = pd.Series([1, 2], name="金额")
    calls = []

    def transform(value, **kwargs):
        calls.append((value, kwargs))
        return value + 1

    expected = series.apply(transform, convert_dtype=False)
    expected_calls = calls.copy()
    calls.clear()
    actual = series.hscredit(n_jobs=1, bar=False).apply(transform, convert_dtype=False)
    pd.testing.assert_series_equal(actual, expected)
    assert calls == expected_calls
    if Version(pd.__version__) >= Version("3.0"):
        assert calls == [(1, {"convert_dtype": False}), (2, {"convert_dtype": False})]
    else:
        assert actual.dtype == object


def test_include_groups_version_semantics_and_single_execution():
    frame = pd.DataFrame({"组": ["a", "a", "b"], "金额": [1, 2, 3]})
    grouped = frame.groupby("组", sort=False)
    calls = []

    def total(group):
        calls.append(tuple(group.columns))
        return group["金额"].sum()

    if Version(pd.__version__) >= Version("3.0"):
        with pytest.raises(ValueError, match="include_groups"):
            grouped.hscredit(n_jobs=1, bar=False).apply(total, include_groups=True)
        assert calls == []
    expected = grouped.apply(total)
    expected_calls = calls.copy()
    calls.clear()
    actual = grouped.hscredit(n_jobs=1, bar=False).apply(total)
    pd.testing.assert_series_equal(actual, expected)
    assert calls == expected_calls


@pytest.mark.parametrize("dtype", ["Int64", "string", "datetime64[ns]"])
def test_nullable_boxing_preserves_native_missing_scalar(dtype):
    values = ["2026-01-01", None] if dtype == "datetime64[ns]" else [1, None]
    series = pd.Series(values, dtype=dtype, name="特征")
    expected = series.apply(lambda value: (type(value).__name__, bool(pd.isna(value))))
    actual = series.hscredit(n_jobs=1, bar=False).apply(lambda value: (type(value).__name__, bool(pd.isna(value))))
    pd.testing.assert_series_equal(actual, expected)


def test_groupby_missing_key_and_output_columns_match_native():
    frame = pd.DataFrame({"组": ["a", None, "a"], "金额": [2, 5, 7]})
    grouped = frame.groupby("组", dropna=False, sort=False, as_index=False)
    kwargs = {"include_groups": False} if Version(pd.__version__) >= Version("2.2") else {}
    expected = grouped.apply(lambda group: group["金额"].sum(), **kwargs)
    actual = grouped.hscredit(n_jobs=1, bar=False).apply(lambda group: group["金额"].sum(), **kwargs)
    pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_array_equal(actual.iloc[:, -1].to_numpy(), [9, 5])
