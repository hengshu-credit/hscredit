"""测试分箱图对合计行的处理。"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from hscredit.core.viz.binning_plots import bin_plot


def _feature_table_with_total():
    return pd.DataFrame(
        {
            "分箱": [0, 1, "合计"],
            "分箱标签": ["[0, 100)", "[100, 200)", "合计"],
            "好样本数": [80, 60, 140],
            "坏样本数": [20, 40, 60],
            "样本总数": [100, 100, 200],
            "坏样本率": [0.2, 0.4, 0.3],
        }
    )


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_bin_plot_excludes_total_row(orientation):
    fig, plotted_table = bin_plot(
        _feature_table_with_total(),
        orientation=orientation,
        return_frame=True,
    )

    assert plotted_table["分箱标签"].tolist() == ["[0, 100)", "[100, 200)"]
    assert len(fig.axes[0].patches) == 4


def test_bin_plot_rejects_table_with_only_total_row():
    total_row = _feature_table_with_total().tail(1)

    with pytest.raises(ValueError, match="排除合计行后没有可绘制"):
        bin_plot(total_row)
