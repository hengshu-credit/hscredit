"""可视化统计口径、保存和无副作用契约的回归测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hscredit.core.metrics import iv, ks
from hscredit.core.viz import (
    approval_rate_trend_plot,
    bad_rate_trend_plot,
    calibration_plot,
    confusion_matrix_plot,
    get_palette,
    psi_plot,
    score_bin_plot,
    score_distribution_comparison_plot,
)
from hscredit.core.viz.binning_plots import _compute_feature_bin_stats, bin_trend_plot
from hscredit.core.viz.strategy_plots import _quick_psi
from hscredit.core.viz.style import get_current_theme, reset_style, set_style
from hscredit.core.viz.utils import save_figure
from hscredit.utils import init as init_module


def _metric_data():
    rng = np.random.RandomState(42)
    target = np.tile([0, 1], 100)
    feature = target + rng.normal(scale=0.5, size=len(target))
    return pd.DataFrame({"特征": feature, "目标": target})


def test_bin_trend_stats_use_target_first_metric_contract():
    """趋势摘要不能因 IV/KS 参数传反而静默显示为零。"""
    data = _metric_data()

    stats = _compute_feature_bin_stats(data, "特征", "目标", max_n_bins=5)

    assert stats["iv_bin"].sum() == pytest.approx(iv(data["目标"], data["特征"]))
    assert stats["ks_bin"].max() == pytest.approx(ks(data["目标"], data["特征"]))


@pytest.mark.parametrize(
    "base,target",
    [
        (np.array([0.0, 0.5, 1.0]), np.array([10.0, 10.5, 11.0])),
        (np.array(["A", "A", "B", "B"]), np.array(["A", "C", "C", "C"])),
    ],
)
def test_quick_psi_detects_outside_range_and_categorical_drift(base, target):
    """范围外和类别漂移都必须得到有限的正 PSI，不能返回 NaN/零。"""
    value = _quick_psi(base, target, n_bins=3)

    assert np.isfinite(value)
    assert value > 0


def test_quick_psi_returns_nan_when_either_side_has_no_observations():
    """全缺失期间应标记为不可计算，不能抛错或伪装成稳定的零。"""
    assert np.isnan(_quick_psi(np.array([np.nan]), np.array([1.0, 2.0])))
    assert np.isnan(_quick_psi(np.array([1.0, 2.0]), np.array([np.nan])))


def test_calibration_plot_reuses_supplied_axes_with_histogram():
    """默认带直方图时也不能丢弃调用方传入的 Axes。"""
    fig, ax = plt.subplots()

    returned = calibration_plot(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        ax=ax,
    )

    assert returned is fig
    assert ax.lines
    plt.close(fig)


def test_calibration_plot_uses_observed_mean_probability_and_keeps_zero():
    """校准曲线横坐标应为箱内实际均值，且概率 0 必须计入第一箱。"""
    fig = calibration_plot(
        np.array([0, 1, 0, 1]),
        np.array([0.0, 0.49, 0.99, 1.0]),
        n_bins=2,
        show_histogram=False,
    )

    model_line = fig.axes[0].lines[1]
    np.testing.assert_allclose(model_line.get_xdata(), [0.245, 0.995])
    plt.close(fig)


def test_score_bin_plot_honors_ax_and_n_bins():
    """评分分箱辅助入口必须复用推荐 bin_plot 的 Axes 和分箱数。"""
    data = pd.DataFrame(
        {
            "评分": np.arange(120, dtype=float),
            "目标": np.tile([0, 1], 60),
        }
    )
    fig, ax = plt.subplots()

    returned = score_bin_plot(
        data,
        "评分",
        "目标",
        n_bins=3,
        bin_type="uniform",
        ax=ax,
        show_table=False,
    )

    assert returned is fig
    assert len(ax.get_yticklabels()) == 3
    plt.close(fig)


def test_save_figure_infers_svg_format_from_suffix(tmp_path):
    """SVG 路径必须保存真实 SVG，不能写入伪装成 SVG 的 PNG。"""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    output = tmp_path / "figure.svg"

    save_figure(fig, output)

    assert output.read_bytes().lstrip().startswith(b"<?xml")
    plt.close(fig)


def test_save_figure_preserves_extensionless_path(tmp_path):
    """无后缀路径仍应写入用户给出的精确文件名，并使用 PNG 内容。"""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    output = tmp_path / "figure"

    save_figure(fig, output)

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "figure.png").exists()
    plt.close(fig)


@pytest.mark.parametrize(
    "plotter,kwargs",
    [
        (
            approval_rate_trend_plot,
            {"date_col": "日期", "decision_col": "通过", "target_col": "目标"},
        ),
        (
            bad_rate_trend_plot,
            {"date_col": "日期", "target": "目标", "show_sample_count": False},
        ),
    ],
)
def test_trend_plots_do_not_mutate_input_dataframe(plotter, kwargs):
    """绘图不得永久修改日期类型或向调用方 DataFrame 注入临时列。"""
    data = pd.DataFrame(
        {
            "日期": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
            "通过": [1, 1, 0, 1],
            "目标": [0, 1, 1, 0],
        }
    )
    original = data.copy(deep=True)

    fig = plotter(data, **kwargs)

    pd.testing.assert_frame_equal(data, original)
    plt.close(fig)


def test_bin_trend_plot_does_not_mutate_dimension_list():
    """日期分组不能向用户复用的 dimension_cols 列表追加内部字段。"""
    data = pd.DataFrame(
        {
            "日期": pd.date_range("2026-01-01", periods=40),
            "渠道": np.repeat(["A", "B"], 20),
            "特征": np.arange(40, dtype=float),
            "目标": np.tile([0, 1], 20),
        }
    )
    dimensions = ["渠道"]

    fig = bin_trend_plot(
        data,
        "特征",
        "目标",
        dimension_cols=dimensions,
        date_col="日期",
        max_n_bins=3,
        show_stats=False,
    )

    assert dimensions == ["渠道"]
    plt.close(fig)


def test_set_and_reset_style_build_on_init_setting(monkeypatch):
    """主题切换与重置都应从 canonical init_setting 基线开始。"""
    calls = []

    def fake_init_setting():
        calls.append(True)
        plt.rcParams["font.family"] = ["Canonical Font"]

    monkeypatch.setattr(init_module, "init_setting", fake_init_setting)

    with plt.rc_context():
        set_style("minimal")
        assert calls == [True]
        assert plt.rcParams["font.family"] == ["Canonical Font"]
        assert get_current_theme() == "minimal"

        reset_style()
        assert calls == [True, True]
        assert plt.rcParams["font.family"] == ["Canonical Font"]
        assert get_current_theme() is None


def test_set_style_without_font_detection_still_uses_init_setting(monkeypatch):
    """关闭额外字体检测时也不能绕过 canonical init_setting 样式基线。"""
    calls = []

    def fake_init_setting():
        calls.append(True)
        plt.rcParams["font.family"] = ["Canonical Font"]

    monkeypatch.setattr(init_module, "init_setting", fake_init_setting)

    with plt.rc_context():
        set_style("minimal", chinese_font=False)
        assert calls == [True]
        assert plt.rcParams["font.family"] == ["Canonical Font"]


def test_get_palette_returns_isolated_mutable_values():
    """调用方修改返回色板时不能污染模块内的全局配色。"""
    colors = get_palette("default")
    semantic = get_palette("semantic")
    original_color = colors[0]
    original_bad_rate = semantic["bad_rate"]

    colors[0] = "#000000"
    semantic["bad_rate"] = "#000000"

    assert get_palette("default")[0] == original_color
    assert get_palette("semantic")["bad_rate"] == original_bad_rate


def test_score_distribution_supports_constant_scores_without_kde_failure():
    """常量评分仍应绘制直方图，KDE 不可用时不能抛奇异矩阵异常。"""
    fig = score_distribution_comparison_plot({"常量评分": np.ones(10)})

    assert fig.axes[0].patches
    plt.close(fig)


def test_confusion_matrix_plot_supports_multiclass_labels():
    """混淆矩阵不能把所有输入都硬编码成二分类 2×2。"""
    fig = confusion_matrix_plot(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([0, 2, 2, 0, 1, 1]),
    )

    heatmap = np.asarray(fig.axes[0].collections[0].get_array())
    assert heatmap.size == 9
    assert "准确率" in fig.axes[0].get_title()
    plt.close(fig)


def test_bad_rate_trend_accepts_single_custom_color():
    """单色主题也应能绘制样本数面板，不能固定访问 colors[2]。"""
    data = pd.DataFrame(
        {
            "日期": pd.date_range("2026-01-01", periods=8, freq="MS"),
            "目标": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    fig = bad_rate_trend_plot(data, "日期", target="目标", colors=["#2639E9"])

    assert len(fig.axes) == 2
    assert fig.axes[1].patches
    plt.close(fig)


def _psi_feature_table(counts, bad_rates):
    total = sum(counts)
    return pd.DataFrame(
        {
            "分箱": ["[0, 1)", "[1, 2)"],
            "样本总数": counts,
            "样本占比": [count / total for count in counts],
            "坏样本率": bad_rates,
        }
    )


@pytest.mark.parametrize(
    "expected,actual",
    [
        (
            pd.Series([0.1, 0.2, 0.8, 0.9], name="评分"),
            pd.Series([0.15, 0.3, 0.7, 0.95], name="评分"),
        ),
        (
            _psi_feature_table([100, 200], [0.10, 0.20]),
            _psi_feature_table([120, 180], [0.15, 0.25]),
        ),
    ],
)
def test_psi_plot_without_y_hides_bad_rate_axis_and_legend(expected, actual):
    """未传 y 时，即使分箱表含坏率列，也只能展示样本占比。"""
    fig = psi_plot(expected, actual)

    assert len(fig.axes) == 1
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == ["预期样本占比", "实际样本占比"]
    assert not any("坏样本率" in line.get_label() for line in fig.axes[0].lines)
    plt.close(fig)


def test_psi_plot_with_y_keeps_bad_rate_axis_and_legend():
    """显式传入 expected+actual 对应标签时继续展示两组坏率曲线。"""
    expected = pd.Series([0.1, 0.2, 0.8, 0.9], name="评分")
    actual = pd.Series([0.15, 0.3, 0.7, 0.95], name="评分")
    target = np.array([0, 1, 0, 1, 0, 1, 1, 1])

    fig = psi_plot(expected, actual, y=target)

    assert len(fig.axes) == 2
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "预期坏样本率" in legend_labels
    assert "实际坏样本率" in legend_labels
    plt.close(fig)


def test_psi_plot_without_y_preserves_result_table_contract():
    """隐藏坏率曲线不应删除 result=True 的兼容列或影响 plot=False。"""
    expected = _psi_feature_table([100, 200], [0.10, 0.20])
    actual = _psi_feature_table([120, 180], [0.15, 0.25])

    result = psi_plot(expected, actual, y=None, result=True, plot=False)

    assert isinstance(result, pd.DataFrame)
    assert "预期坏样本率" in result.columns
    assert "实际坏样本率" in result.columns
    assert result["总体PSI值"].nunique() == 1
