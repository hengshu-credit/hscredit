"""单样本 SHAP 力图与瀑布图组合图测试。"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from hscredit.core import viz
from hscredit.core.viz import plot_model_sample_shap
from hscredit.core.viz.style import PRIMARY_COLORS


class _ProbabilityModel:
    """为绘图测试提供稳定的二维概率输出。"""

    def __init__(self, feature_names):
        self.feature_names_in_ = np.asarray(feature_names)

    def predict_proba(self, values):
        positive = np.full(len(values), 0.8, dtype=float)
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def sample_shap_inputs(monkeypatch):
    """仅替换昂贵的归因计算，保留真实 SHAP 力图和瀑布图渲染。"""
    shap = pytest.importorskip("shap")
    names = ["年龄", "收入", "负债率", "查询次数"]
    sample = pd.DataFrame([[1.123456, 2.234567, 3.345678, 4.456789]], columns=names, index=["申请A"])
    background = pd.DataFrame(
        [[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]],
        columns=names,
    )

    class _DeterministicExplainer:
        def __init__(self, predictor, background_values):
            self.base_value = float(np.asarray(background_values)[0, 0]) / 10.0

        def __call__(self, sample_values):
            class_zero = np.array([0.01, -0.02, 0.03, -0.04])
            class_one = np.array([0.4, -0.3, 0.2, -0.1])
            values = np.stack([class_zero, class_one], axis=-1)[None, :, :]
            return SimpleNamespace(
                values=values,
                base_values=np.array([[0.1, self.base_value]]),
                data=np.asarray(sample_values),
                feature_names=names,
            )

    monkeypatch.setattr(shap, "Explainer", _DeterministicExplainer)
    return _ProbabilityModel(names), sample, background


def test_plot_model_sample_shap_is_public():
    """删除公开导出会让业务代码无法使用新绘图入口。"""
    assert callable(getattr(viz, "plot_model_sample_shap", None))


def test_plot_model_sample_shap_stacks_force_and_full_waterfall(sample_shap_inputs):
    """上下图颠倒、类别选择错误或默认遗漏特征都会误导单样本解释。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(
        model,
        sample,
        background,
        background_size=None,
        max_display=None,
        show=False,
    )
    fig.canvas.draw()

    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    assert force_ax.get_position().y0 > waterfall_ax.get_position().y0
    assert force_ax.patches
    assert force_ax.get_title() == ""
    assert waterfall_ax.get_title() == "SHAP 瀑布图"
    assert fig._suptitle.get_text() == "单样本 SHAP 预测归因：_ProbabilityModel（predict_proba）"

    ytick_text = [tick.get_text() for tick in waterfall_ax.get_yticklabels()]
    assert len(ytick_text) == 2 * sample.shape[1]
    assert all(any(name in text for text in ytick_text) for name in sample.columns)
    assert {text.get_text() for text in waterfall_ax.texts} >= {"+0.4000", "+0.2000", "−0.3000", "−0.1000"}
    assert any("1.1235 = 年龄" in text for text in ytick_text)
    assert any("0.6" in tick.get_text() for ax in fig.axes for tick in ax.get_xticklabels())
    auxiliary_tick_text = [tick.get_text() for axis in fig.axes if axis not in {force_ax, waterfall_ax} for tick in axis.get_xticklabels()]
    assert any("0.6000" in text for text in auxiliary_tick_text)
    assert any("0.8000" in text for text in auxiliary_tick_text)
    np.testing.assert_allclose(fig.get_size_inches(), [14.0, 10.0])
    plt.close(fig)


def test_force_plot_remains_native_artists_without_full_canvas_raster(sample_shap_inputs):
    """把原生力图整体栅格化会导致组合图缩放时条带被压扁。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)

    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    native_labels = {text.get_text() for text in force_ax.texts}
    assert not any(np.asarray(image.get_array()).ndim == 3 for image in force_ax.images)
    assert force_ax.patches
    assert {"base value", "higher", "lower", "output value"} <= native_labels
    plt.close(fig)


def test_force_and_waterfall_bars_have_similar_physical_height(sample_shap_inputs):
    """上方力图条带不能明显细于或粗于下方单条瀑布柱。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")

    force_height = float(np.median([patch.get_window_extent(renderer).height for patch in force_ax.patches]))
    waterfall_height = float(np.median([patch.get_window_extent(renderer).height for patch in waterfall_ax.patches]))
    assert 0.8 <= force_height / waterfall_height <= 1.25
    plt.close(fig)


def test_native_force_labels_do_not_overlap_figure_title(sample_shap_inputs):
    """SHAP 原生 higher/lower 等顶部标注不能与组合图总标题重叠。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    title_box = fig._suptitle.get_window_extent(renderer)
    native_top_texts = [text for text in force_ax.texts if text.get_text() in {"base value", "higher", "lower", "output value"}]
    assert not any(title_box.overlaps(text.get_window_extent(renderer)) for text in native_top_texts)
    plt.close(fig)


def test_native_force_labels_keep_padding_above_axis_ticks(monkeypatch):
    """预测值等原生顶部标注与横轴刻度文本之间应保留清晰留白。"""
    shap = pytest.importorskip("shap")

    class OutputAtTickExplainer:
        def __init__(self, predictor, background_values):
            self.n_features = np.asarray(background_values).shape[1]

        def __call__(self, sample_values):
            class_one = np.linspace(0.4, -0.25, self.n_features)
            values = np.stack([-class_one, class_one], axis=-1)[None, :, :]
            return SimpleNamespace(
                values=values,
                base_values=np.array([[0.4, 0.6]]),
                data=np.asarray(sample_values),
            )

    monkeypatch.setattr(shap, "Explainer", OutputAtTickExplainer)
    long_names = [
        "最近十二个月非银行金融机构查询次数",
        "最近六个月平均账户使用额度占比",
        "历史最长连续逾期月份数量",
        "当前申请人与关联方综合风险等级",
        "近三个月账户平均余额变化率",
        "最近一年内贷款审批查询总次数",
        "过去半年信用卡最高使用率",
        "当前所有未结清贷款余额合计",
    ]
    model = _ProbabilityModel(long_names)
    sample = pd.DataFrame([np.linspace(0.1, 0.9, len(long_names))], columns=long_names)
    background = pd.DataFrame(np.arange(64, dtype=float).reshape(8, 8), columns=long_names)

    fig = plot_model_sample_shap(model, sample, background, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    tick_top = max(tick.get_window_extent(renderer).y1 for tick in force_ax.get_xticklabels() if tick.get_visible())
    native_top_texts = [text for text in force_ax.texts if " = " not in text.get_text()]
    annotation_bottom = min(text.get_window_extent(renderer).y0 for text in native_top_texts)
    assert annotation_bottom - tick_top >= 9.5
    plt.close(fig)


def test_explicit_max_display_limits_waterfall_and_groups_remaining_features(sample_shap_inputs):
    """显式 max_display 未限制瀑布行数时，大模型的局部解释会失去可读性。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, max_display=2, show=False)

    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    ytick_text = [tick.get_text() for tick in waterfall_ax.get_yticklabels()]
    assert len(ytick_text) == 4
    assert any("其他 3 个特征" in text for text in ytick_text)
    plt.close(fig)


def test_signed_contributions_use_hscredit_theme_colors(sample_shap_inputs):
    """回退到 SHAP 亮红亮蓝会破坏项目统一的正负贡献颜色语义。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)

    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    force_colors = {to_hex(patch.get_facecolor(), keep_alpha=False).lower() for patch in force_ax.patches}
    assert {PRIMARY_COLORS[0].lower(), PRIMARY_COLORS[1].lower()} <= force_colors

    waterfall_colors = {to_hex(patch.get_facecolor(), keep_alpha=False).lower() for patch in waterfall_ax.patches}
    assert {PRIMARY_COLORS[0].lower(), PRIMARY_COLORS[1].lower()} <= waterfall_colors
    plt.close(fig)


def test_force_plot_is_centered_independently_from_waterfall_axis(sample_shap_inputs):
    """力图不应被瀑布图长纵轴标签向右推移，应独立居中展示。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)
    fig.canvas.draw()

    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    force_center = (force_ax.get_position().x0 + force_ax.get_position().x1) / 2
    waterfall_center = (waterfall_ax.get_position().x0 + waterfall_ax.get_position().x1) / 2
    assert force_center == pytest.approx(0.5, abs=1e-3)
    assert force_center != pytest.approx(waterfall_center, abs=1e-3)

    assert to_hex(waterfall_ax.spines["bottom"].get_edgecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()
    assert to_hex(waterfall_ax.spines["left"].get_edgecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()
    assert all(to_hex(tick.get_color()).lower() == PRIMARY_COLORS[0].lower() for tick in waterfall_ax.get_xticklabels())
    plt.close(fig)


def test_force_plot_is_centered_in_tight_output_with_long_waterfall_labels(sample_shap_inputs, monkeypatch):
    """紧边界包含瀑布图长标签时，力图仍应位于最终输出的正中间。"""
    shap = pytest.importorskip("shap")

    class DynamicExplainer:
        def __init__(self, predictor, background_values):
            self.n_features = np.asarray(background_values).shape[1]

        def __call__(self, sample_values):
            class_one = np.linspace(0.4, -0.25, self.n_features)
            values = np.stack([-class_one, class_one], axis=-1)[None, :, :]
            return SimpleNamespace(
                values=values,
                base_values=np.array([[0.4, 0.6]]),
                data=np.asarray(sample_values),
            )

    monkeypatch.setattr(shap, "Explainer", DynamicExplainer)
    long_names = [
        "最近十二个月非银行金融机构查询次数",
        "最近六个月平均账户使用额度占比",
        "历史最长连续逾期月份数量",
        "当前申请人与关联方综合风险等级",
        "近三个月账户平均余额变化率",
        "最近一年内贷款审批查询总次数",
        "过去半年信用卡最高使用率",
        "当前所有未结清贷款余额合计",
    ]
    sample = pd.DataFrame([np.linspace(0.1, 0.9, len(long_names))], columns=long_names)
    background = pd.DataFrame(np.arange(64, dtype=float).reshape(8, 8), columns=long_names)

    fig = plot_model_sample_shap(_ProbabilityModel(long_names), sample, background, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    force_box = force_ax.get_window_extent(renderer)
    tight_box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    force_center = (force_box.x0 + force_box.x1) / 2
    tight_center = (tight_box.x0 + tight_box.x1) / 2
    assert force_center == pytest.approx(tight_center, abs=5.0)
    plt.close(fig)


def test_force_labels_do_not_overlap_when_long_feature_count_grows(sample_shap_inputs, monkeypatch):
    """长字段增加到八个时，水平分行标签之间仍不能发生重叠。"""
    shap = pytest.importorskip("shap")

    class DynamicExplainer:
        def __init__(self, predictor, background_values):
            self.n_features = np.asarray(background_values).shape[1]

        def __call__(self, sample_values):
            class_one = np.linspace(0.4, -0.25, self.n_features)
            values = np.stack([-class_one, class_one], axis=-1)[None, :, :]
            return SimpleNamespace(
                values=values,
                base_values=np.array([[0.4, 0.6]]),
                data=np.asarray(sample_values),
            )

    monkeypatch.setattr(shap, "Explainer", DynamicExplainer)
    long_names = [
        "最近十二个月非银行金融机构查询次数",
        "最近六个月平均账户使用额度占比",
        "历史最长连续逾期月份数量",
        "当前申请人与关联方综合风险等级",
        "近三个月账户平均余额变化率",
        "最近一年内贷款审批查询总次数",
        "过去半年信用卡最高使用率",
        "当前所有未结清贷款余额合计",
    ]
    sample = pd.DataFrame([np.linspace(0.1, 0.9, len(long_names))], columns=long_names)
    background = pd.DataFrame(np.arange(64, dtype=float).reshape(8, 8), columns=long_names)

    fig = plot_model_sample_shap(_ProbabilityModel(long_names), sample, background, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    feature_boxes = [text.get_window_extent(renderer) for text in force_ax.texts if " = " in text.get_text()]
    solid_bar_bottom = max(patch.get_window_extent(renderer).y0 for patch in force_ax.patches)
    assert max(box.y1 for box in feature_boxes) <= solid_bar_bottom
    assert not any(first.overlaps(second) for index, first in enumerate(feature_boxes) for second in feature_boxes[index + 1 :])
    plt.close(fig)


def test_force_and_waterfall_bar_height_stays_balanced_for_different_feature_counts(sample_shap_inputs, monkeypatch):
    """字段数量变化时应动态分配面板高度，并保持上下柱条大小接近。"""
    shap = pytest.importorskip("shap")

    class DynamicExplainer:
        def __init__(self, predictor, background_values):
            self.base_value = 0.6

        def __call__(self, sample_values):
            n_features = np.asarray(sample_values).shape[1]
            class_one = np.linspace(0.2, -0.1, n_features)
            values = np.stack([-class_one, class_one], axis=-1)[None, :, :]
            return SimpleNamespace(
                values=values,
                base_values=np.array([[0.4, self.base_value]]),
                data=np.asarray(sample_values),
                feature_names=[f"字段{i}" for i in range(n_features)],
            )

    monkeypatch.setattr(shap, "Explainer", DynamicExplainer)
    model, sample, background = sample_shap_inputs
    short_fig = plot_model_sample_shap(model, sample, background, show=False)

    many_names = [f"字段{i}" for i in range(12)]
    many_sample = pd.DataFrame([np.linspace(0.1, 1.2, 12)], columns=many_names)
    many_background = pd.DataFrame(np.arange(60, dtype=float).reshape(5, 12), columns=many_names)
    many_fig = plot_model_sample_shap(_ProbabilityModel(many_names), many_sample, many_background, show=False)
    short_fig.canvas.draw()
    many_fig.canvas.draw()

    def bar_height_ratio(figure):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        force_ax = next(ax for ax in figure.axes if ax.get_label() == "SHAP力图")
        waterfall_ax = next(ax for ax in figure.axes if ax.get_label() == "SHAP瀑布图")
        force_height = float(np.median([patch.get_window_extent(renderer).height for patch in force_ax.patches]))
        waterfall_height = float(np.median([patch.get_window_extent(renderer).height for patch in waterfall_ax.patches]))
        return force_height / waterfall_height

    assert 0.8 <= bar_height_ratio(short_fig) <= 1.25
    assert 0.8 <= bar_height_ratio(many_fig) <= 1.25
    short_force_ax = next(ax for ax in short_fig.axes if ax.get_label() == "SHAP力图")
    many_force_ax = next(ax for ax in many_fig.axes if ax.get_label() == "SHAP力图")
    assert many_force_ax.get_position().height < short_force_ax.get_position().height
    plt.close(short_fig)
    plt.close(many_fig)


def test_force_feature_labels_stay_horizontal_when_short(sample_shap_inputs, monkeypatch):
    """字段少且标签短时不应旋转，避免为了防极端情况牺牲普通图的可读性。"""
    shap = pytest.importorskip("shap")
    original_force_plot = shap.force_plot
    observed = {}

    def recording_force_plot(*args, **kwargs):
        observed["text_rotation"] = kwargs.get("text_rotation")
        return original_force_plot(*args, **kwargs)

    monkeypatch.setattr(shap, "force_plot", recording_force_plot)
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, show=False)

    assert observed["text_rotation"] == 0
    plt.close(fig)


def test_force_long_feature_labels_wrap_without_rotation_or_overlap(sample_shap_inputs, monkeypatch):
    """超长字段应完整换行并保持水平，且原生顶部标注不能被裁掉。"""
    shap = pytest.importorskip("shap")
    original_force_plot = shap.force_plot
    observed = {}

    def recording_force_plot(*args, **kwargs):
        observed["text_rotation"] = kwargs.get("text_rotation")
        observed["feature_names"] = kwargs.get("feature_names")
        observed["out_names"] = kwargs.get("out_names")
        return original_force_plot(*args, **kwargs)

    monkeypatch.setattr(shap, "force_plot", recording_force_plot)
    _, sample, background = sample_shap_inputs
    long_names = [
        "最近十二个月非银行金融机构查询次数",
        "最近六个月平均账户使用额度占比",
        "历史最长连续逾期月份数量",
        "当前申请人与关联方综合风险等级",
    ]
    sample.columns = long_names
    background.columns = long_names
    model = _ProbabilityModel(long_names)

    fig = plot_model_sample_shap(model, sample, background, show=False)

    assert observed["text_rotation"] == 0
    assert any("\n" in name for name in observed["feature_names"])
    assert all(len(line) <= 8 for name in observed["feature_names"] for line in name.splitlines())
    assert not any("…" in name for name in observed["feature_names"])
    assert observed["out_names"] == "output value"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    force_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP力图")
    native_labels = {text.get_text() for text in force_ax.texts}
    assert {"base value", "higher", "lower", "output value"} <= native_labels
    native_top_texts = [text for text in force_ax.texts if text.get_text() in {"base value", "higher", "lower", "output value"}]
    assert all(0 <= text.get_window_extent(renderer).x0 and text.get_window_extent(renderer).x1 <= canvas_width and 0 <= text.get_window_extent(renderer).y0 and text.get_window_extent(renderer).y1 <= canvas_height for text in native_top_texts)
    feature_texts = [text for text in force_ax.texts if " = " in text.get_text()]
    rendered_feature_names = {text.get_text().split(" = ", maxsplit=1)[0].replace("\n", "") for text in feature_texts}
    assert set(long_names) <= rendered_feature_names
    feature_boxes = [text.get_window_extent(renderer) for text in feature_texts]
    assert not any(first.overlaps(second) for index, first in enumerate(feature_boxes) for second in feature_boxes[index + 1 :])
    assert not any(np.asarray(image.get_array()).ndim == 3 for image in force_ax.images)
    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    waterfall_text = " ".join(tick.get_text() for tick in waterfall_ax.get_yticklabels())
    assert all(name in waterfall_text for name in long_names)
    plt.close(fig)


def test_value_precision_controls_visible_feature_and_contribution_values(sample_shap_inputs):
    """自定义精度必须只改变显示文本，并同时作用于特征值和 SHAP 贡献。"""
    model, sample, background = sample_shap_inputs

    fig = plot_model_sample_shap(model, sample, background, value_precision=3, show=False)

    waterfall_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP瀑布图")
    ytick_text = [tick.get_text() for tick in waterfall_ax.get_yticklabels()]
    assert any("1.123 = 年龄" in text for text in ytick_text)
    assert {text.get_text() for text in waterfall_ax.texts} >= {"+0.400", "+0.200", "−0.300", "−0.100"}
    plt.close(fig)


def test_sample_must_contain_exactly_one_row(sample_shap_inputs):
    """多行样本被静默取首行会让解释对象与业务申请错位。"""
    model, sample, background = sample_shap_inputs

    with pytest.raises(ValueError, match="单个样本"):
        plot_model_sample_shap(model, pd.concat([sample, sample]), background, show=False)


def test_sample_features_must_match_background(sample_shap_inputs):
    """样本与背景字段错位时，SHAP 贡献会绑定到错误的业务字段。"""
    model, sample, background = sample_shap_inputs

    with pytest.raises(ValueError, match="特征名称或顺序"):
        plot_model_sample_shap(model, sample.rename(columns={"年龄": "年龄_错误"}), background, show=False)


def test_ndarray_sample_feature_count_error_is_chinese(sample_shap_inputs):
    """ndarray 字段数量错误不应泄露 pandas 的英文构造异常。"""
    model, _, background = sample_shap_inputs

    with pytest.raises(ValueError, match="特征数量"):
        plot_model_sample_shap(model, np.array([1.0, 2.0, 3.0]), background, show=False)


@pytest.mark.parametrize("max_display", [0, -1, True, 1.5])
def test_max_display_rejects_invalid_limits_in_chinese(sample_shap_inputs, max_display):
    """无效展示上限必须在进入 SHAP 计算前给出明确参数错误。"""
    model, sample, background = sample_shap_inputs

    with pytest.raises((TypeError, ValueError), match="max_display"):
        plot_model_sample_shap(model, sample, background, max_display=max_display, show=False)


@pytest.mark.parametrize("value_precision", [-1, True, 1.5])
def test_value_precision_rejects_invalid_values(sample_shap_inputs, value_precision):
    """显示精度必须是非负整数，且不能把布尔值当作整数接受。"""
    model, sample, background = sample_shap_inputs

    with pytest.raises((TypeError, ValueError), match="value_precision"):
        plot_model_sample_shap(model, sample, background, value_precision=value_precision, show=False)
