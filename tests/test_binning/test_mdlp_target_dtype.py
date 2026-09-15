"""MDLP 分箱对二分类标签存储类型的兼容性回归测试。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hscredit.core.binning import MDLPBinning, OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.selectors import CompositeFeatureSelector, NullSelector, TypeSelector


@pytest.fixture
def binary_data():
    """提供有不同坏样本率的数值特征，并保留非默认索引和特征缺失值。"""
    X = pd.DataFrame(
        {"评分": np.arange(80, dtype=float), "机构数": np.arange(80, dtype=float)[::-1]},
        index=np.arange(100, 180),
    )
    X.loc[105, "评分"] = np.nan
    y = pd.Series([0, 0, 0, 1] * 10 + [0, 1, 1, 1] * 10, index=X.index, name="目标")
    return X, y


@pytest.mark.parametrize("dtype", ["float64", "float32", "bool", "Float64"])
@pytest.mark.parametrize("factory", [False, True], ids=["direct", "optimal"])
@pytest.mark.parametrize("n_jobs,backend", [(1, None), (2, "threading"), (2, "loky")])
@pytest.mark.parametrize("target_column", [False, True], ids=["explicit-y", "target-column"])
def test_mdlp_binary_target_dtypes_match_integer(binary_data, dtype, factory, n_jobs, backend, target_column):
    X, y = binary_data
    params = dict(target="目标", max_n_bins=5, n_jobs=n_jobs, parallel_backend=backend)
    cls = OptimalBinning if factory else MDLPBinning
    if factory:
        params["method"] = "mdlp"
    expected = cls(**params).fit(X, y)
    target = y.astype(dtype)
    original = target.copy()
    actual = cls(**params)
    if target_column:
        actual.fit(X.assign(目标=target))
    else:
        actual.fit(X, target)

    pd.testing.assert_series_equal(target, original)
    for feature in X:
        np.testing.assert_array_equal(actual.splits_[feature], expected.splits_[feature])
        pd.testing.assert_frame_equal(actual.bin_tables_[feature], expected.bin_tables_[feature])
    for metric in ("indices", "bins", "woe"):
        pd.testing.assert_frame_equal(actual.transform(X, metric=metric), expected.transform(X, metric=metric))


@pytest.mark.parametrize("invalid", [0.5, np.nan, np.inf])
def test_mdlp_rejects_invalid_targets_before_fitting(binary_data, invalid):
    X, y = binary_data
    y = y.astype(float)
    y.iloc[0] = invalid
    with pytest.raises(ValueError, match="目标变量"):
        MDLPBinning(n_jobs=1).fit(X, y)


@pytest.mark.integration
def test_real_data_mdlp_woe_pipeline_accepts_float_targets():
    workbook = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not workbook.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    data = pd.read_excel(workbook)
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    X = data[features].astype(float)
    y = data["FPD"]

    def make_pipeline(n_jobs):
        return Pipeline(
            [
                (
                    "pre_selector",
                    CompositeFeatureSelector(
                        [
                            TypeSelector(dtype_include="number", target="FPD", n_jobs=n_jobs),
                            NullSelector(threshold=0.95, target="FPD", n_jobs=n_jobs),
                        ]
                    ),
                ),
                (
                    "binner",
                    OptimalBinning(
                        target="FPD",
                        method="mdlp",
                        max_n_bins=5,
                        min_bin_size=0.01,
                        monotonic="auto_asc_desc",
                        user_splits={},
                        user_splits_fixed=True,
                        n_jobs=n_jobs,
                        parallel_backend="threading",
                    ),
                ),
                ("woe", WOEEncoder(target="FPD", n_jobs=n_jobs)),
            ]
        )

    expected = make_pipeline(1).fit_transform(X, y.astype(int))
    actual = make_pipeline(2).fit_transform(X, y.astype(float))
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.shape == X.shape
    assert np.isfinite(actual.to_numpy()).all()
