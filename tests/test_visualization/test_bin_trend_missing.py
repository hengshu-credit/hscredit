"""分箱趋势图在全缺失、全特殊值及字符串扩展类型下的回归验证。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

from hscredit import bin_trend_plot
from hscredit.core.viz.binning_plots import _detect_bad_rate_trend


@pytest.mark.parametrize("label_dtype", ["object", "string[python]", "string[pyarrow]"])
@pytest.mark.parametrize(
    "bins, labels, rates, expected",
    [
        ([], [], [], "未知"),
        ([-1], ["缺失值"], [0.25], "未知"),
        ([-2], ["特殊值"], [0.25], "未知"),
        ([-1, -2], ["缺失值", "特殊值"], [0.25, 0.5], "未知"),
        (["缺失值", "特殊值"], ["缺失值", "特殊值"], [0.25, 0.5], "未知"),
        ([0, -1, 1, -2], ["低分", "缺失值", "高分", "特殊值"], [0.1, 0.9, 0.2, 0.0], "上升"),
    ],
)
def test_bad_rate_trend_handles_empty_string_labels(label_dtype, bins, labels, rates, expected):
    """过滤后空标签列必须按布尔掩码处理，且不能改变正常分箱的趋势。"""
    table = pd.DataFrame({"分箱": bins, "分箱标签": pd.Series(labels, dtype=label_dtype), "坏样本率": rates})
    original = table.copy(deep=True)

    assert _detect_bad_rate_trend(table) == expected
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("special_value, label", [(np.nan, "缺失值"), (-999.0, "特殊值")])
def test_bin_trend_renders_month_without_regular_bins(special_value, label):
    """全缺失/全特殊值月份必须保留真实样本和坏率，不能在图内吞掉异常。"""
    data = pd.DataFrame(
        {
            "特征": [special_value] * 8 + list(range(8)),
            "目标": [0, 0, 0, 1] * 4,
            "放款时间": pd.to_datetime(["2025-11-01"] * 8 + ["2025-12-01"] * 8),
        }
    )
    options = {"special_codes": [-999.0]} if label == "特殊值" else {}
    figure = bin_trend_plot(
        data,
        "特征",
        target="目标",
        date="放款时间",
        freq="M",
        max_n_bins=5,
        figsize=(20, 8),
        n_jobs=1,
        **options,
    )
    try:
        figure.canvas.draw()
        texts = [artist.get_text() for artist in figure.findobj(Text)]
        assert not any("Error:" in text or "错误" in text for text in texts)
        panel = next(axis for axis in figure.axes if axis.get_title().startswith("2025-11"))
        assert panel.get_title() == "2025-11\n(2/8, 25.0%)"
        assert [tick.get_text() for tick in panel.get_xticklabels()] == [label]
        assert [container.datavalues.tolist() for container in panel.containers] == [[6], [2]]
        assert "25.00%" in texts
        assert any("IV 0.0000" in text and "LIFT 1.00~1.00" in text for text in texts)
    finally:
        plt.close(figure)
