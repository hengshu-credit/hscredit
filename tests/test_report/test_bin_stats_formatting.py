"""用断言验证分箱格式化，而非打印失败文字后继续通过。"""

import numpy as np
import pytest

from hscredit.core.metrics import compute_bin_stats


@pytest.mark.parametrize("weighted", [False, True])
def test_formatting(weighted):
    bins = np.array([0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    options = (
        dict(target_type="amount_weighted", amount=[1.1234, 2.2345, 3.3456, 4.4567, 5.5678, 6.6789, 7.7891])
        if weighted
        else {}
    )
    raw = compute_bin_stats(bins, y, round_digits=False, **options)
    shown = compute_bin_stats(bins, y, round_digits=True, **options)
    for column, digits in {"坏样本率": 6, "分档WOE值": 6, "指标IV值": 6, "LIFT值": 4, "坏账改善": 4}.items():
        np.testing.assert_array_equal(shown[column], np.round(raw[column], digits))
    if weighted:
        np.testing.assert_allclose(raw["样本总数"], [6.7035, 24.4925], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(shown["样本总数"], [6.7, 24.49])
    else:
        np.testing.assert_array_equal(shown["样本总数"], [3, 4])
        assert raw.iloc[0]["坏样本率"] == 1 / 3
    assert raw.attrs["统计口径"]["展示舍入"] is False
    assert shown.attrs["统计口径"]["展示舍入"] is True


def test_continuous_formatting_keeps_unrounded_source():
    raw = compute_bin_stats([0, 0, 1], [1.1111, 2.2222, 4.4444], target_type="continuous", round_digits=False)
    shown = compute_bin_stats([0, 0, 1], [1.1111, 2.2222, 4.4444], target_type="continuous")
    np.testing.assert_allclose(raw["目标值均值"], [1.66665, 4.4444])
    np.testing.assert_array_equal(shown["目标值均值"], [1.67, 4.44])
    np.testing.assert_allclose(shown["目标值总和"].sum(), 7.7777)
