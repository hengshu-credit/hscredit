"""主题字体变大后，摘要仍可读且不会侵入图例或容器边界。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from tests.test_visualization import test_bin_plot_layout as standalone
from tests.test_visualization import test_embedded_bin_plot_layout as embedded
from hscredit.core.viz.binning_plots import _FullWidthBinMetricSummary


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


def test_full_width_summary_compacts_metric_gaps_after_resize_and_restores_them():
    """窄面板压缩到单空格后仍保留文字边界，放宽时恢复默认间距。"""
    text = "IV 0.8789    KS 0.4000    LIFT 0.50~1.50    趋势 倒U型"
    compact = "IV0.8789 KS0.4000 LIFT0.50~1.50 趋势倒U型"
    fig, ax = plt.subplots(figsize=(9, 3))
    try:
        summary = _FullWidthBinMetricSummary(ax, text, fontsize=12.0, color="blue")
        ax.add_artist(summary)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        original_position = ax.get_position().frozen()
        assert summary.metric_text.get_text() == text

        # 按实际字体度量构造只有单空格版本可容纳的面板，兼容不同渲染器。
        summary.metric_text.set_text(compact)
        compact_width = summary.metric_text.get_window_extent(renderer).width
        padding = summary.pad * renderer.points_to_pixels(summary.prop.get_size_in_points())
        axes_width = compact_width + 2.0 * padding + 2.5
        ax.set_position([0.1, 0.15, axes_width / fig.bbox.width, 0.5])
        fig.canvas.draw()
        summary_bbox = summary.get_window_extent(renderer)
        text_bbox = summary.metric_text.get_window_extent(renderer)
        assert summary.metric_text.get_fontsize() == 12.0
        assert text_bbox.x0 >= summary_bbox.x0 + padding + 1.0
        assert text_bbox.x1 <= summary_bbox.x1 - padding - 1.0
        assert summary.metric_text.get_text() == compact

        ax.set_position(original_position)
        fig.canvas.draw()
        assert summary.metric_text.get_text() == text
        assert summary.metric_text.get_fontsize() == 12.0
    finally:
        plt.close(fig)
