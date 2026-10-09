"""分箱输入按位置对齐且不得隐式丢弃样本。"""
import numpy as np
import pandas as pd
import pytest
from hscredit.core.binning import OptimalBinning


@pytest.mark.parametrize("index", [[10, 20, 30, 40], [1, 1, 2, 2], ["a", "b", "c", "d"]])
def test_equal_length_labels_use_positions_even_with_duplicate_indices(index):
    X = pd.DataFrame({"x": [1, 2, 3, 4]}, index=index)
    y = pd.Series([0, 1, 0, 1], index=[4, 3, 2, 1])
    left = OptimalBinning(method="quantile", max_n_bins=2, n_jobs=1).fit(X, y)
    right = OptimalBinning(method="quantile", max_n_bins=2, n_jobs=1).fit(X, y.to_numpy())
    pd.testing.assert_frame_equal(left.get_bin_table("x"), right.get_bin_table("x"))
    assert left.get_bin_table("x")["样本总数"].sum() == 4


def test_failed_unequal_refit_keeps_previous_model():
    X = pd.DataFrame({"x": range(6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    binner = OptimalBinning(method="quantile", n_jobs=1).fit(X, y)
    before = binner.get_bin_table("x")
    with pytest.raises(ValueError, match="样本数量不一致"):
        binner.fit(X, y.iloc[:4])
    pd.testing.assert_frame_equal(before, binner.get_bin_table("x"))
