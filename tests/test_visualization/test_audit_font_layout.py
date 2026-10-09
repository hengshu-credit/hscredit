"""主题字体变大后，摘要仍可读且不会侵入图例或容器边界。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from tests.test_visualization import test_bin_plot_layout as standalone
from tests.test_visualization import test_embedded_bin_plot_layout as embedded


@pytest.mark.parametrize("case", ["compact", "trend", "batch", "overdue_raw", "overdue_table"])
def test_twelve_point_theme_preserves_metric_summary_bounds(case):
    # 原全量测试中的其他主题设置将轴标题字号切到12；显式固定该条件，
    # 不再依赖测试运行顺序才能触发五个布局回归。
    try:
        with plt.rc_context({"axes.labelsize": 12.0}):
            if case == "compact":
                standalone.test_metric_summary_stays_clear_of_legend((6, 4))
                figure = plt.gcf()
                figure.canvas.draw()
                summary = standalone._summary_text(figure)
                assert summary.get_bbox_patch().get_window_extent(figure.canvas.get_renderer()).y1 < figure.bbox.y1
            elif case == "trend":
                embedded.test_bin_trend_plot_reserves_space_for_every_metric_summary("vertical")
            elif case == "batch":
                embedded.test_batch_bin_trend_plot_preserves_embedded_header_layout()
            elif case == "overdue_raw":
                embedded.test_bin_overdues_plot_raw_mode_preserves_embedded_header_layout()
            else:
                embedded.test_bin_overdues_plot_table_mode_preserves_embedded_header_layout()
    finally:
        plt.close("all")
