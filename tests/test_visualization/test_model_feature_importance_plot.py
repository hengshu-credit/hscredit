"""统一模型特征重要性组合图测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hscredit.core.viz import plot_model_feature_importance
from hscredit.core.viz.style import PRIMARY_COLORS


@pytest.fixture
def fitted_forest():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=23,
    )
    frame = pd.DataFrame(X, columns=["年龄", "收入", "负债率", "近六个月非银多头机构数"])
    model = RandomForestClassifier(n_estimators=12, max_depth=3, random_state=23).fit(frame, y)
    return model, frame, y


def test_combined_plot_uses_independent_top_n_and_brand_styles(fitted_forest):
    """左右 Top N 串用或主题色遗漏会破坏综合看板的核心布局。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    fig = plot_model_feature_importance(
        model,
        X.head(30),
        y[:30],
        prediction_method="predict_proba",
        left_top_n=3,
        right_top_n=4,
        background_size=16,
        random_state=23,
        show=False,
    )

    main_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP分布")
    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "特征重要性")
    dependence_axes = [ax for ax in fig.axes if ax.get_label() == "SHAP依赖"]
    right_title_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP依赖总标题")
    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]

    assert len(main_ax.get_yticklabels()) == 3
    assert len(dependence_axes) == 4
    assert len(colorbar_axes) == 2
    assert all(not ax.spines["outline"].get_visible() for ax in colorbar_axes)
    assert main_ax.get_title() == "模型原生特征重要性与 SHAP 分布"
    assert [text.get_text() for text in right_title_ax.texts] == ["SHAP 特征依赖关系"]
    assert [ax.get_title().split(":", 1)[0] for ax in dependence_axes] == ["Top 1", "Top 2", "Top 3", "Top 4"]
    expected_ranking = np.argsort(-model.feature_importances_, kind="stable")
    assert [ax.get_title().split(": ", 1)[1] for ax in dependence_axes] == [X.columns[index] for index in expected_ranking[:4]]
    assert [label.get_text() for label in main_ax.get_yticklabels()] == [X.columns[index] for index in expected_ranking[:3][::-1]]
    assert importance_ax.patches
    assert all(patch.get_hatch() == "/" for patch in importance_ax.patches)
    np.testing.assert_allclose(
        [patch.get_width() for patch in importance_ax.patches],
        np.sort(model.feature_importances_)[-3:],
    )
    assert importance_ax.get_xlabel() == "模型原生特征重要性"
    assert to_hex(importance_ax.patches[0].get_facecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()
    assert importance_ax.patches[0].get_alpha() < 0.5
    assert to_hex(main_ax.spines["bottom"].get_edgecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()

    shap_scatter = main_ax.collections[0]
    assert to_hex(shap_scatter.cmap(0.0), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()
    assert shap_scatter.cmap(0.0)[2] > shap_scatter.cmap(0.0)[0]
    assert shap_scatter.cmap(1.0)[0] > shap_scatter.cmap(1.0)[2]

    width, height = fig.get_size_inches()
    assert width >= 18
    assert height > 6
    fig.canvas.draw()
    main_box = main_ax.get_position()
    shap_colorbar_box = colorbar_axes[0].get_position()
    assert shap_colorbar_box.x0 == pytest.approx(main_box.x0, abs=1e-6)
    assert shap_colorbar_box.x1 == pytest.approx(main_box.x1, abs=1e-6)
    first_box = dependence_axes[0].get_position()
    physical_aspect = first_box.width * width / (first_box.height * height)
    assert physical_aspect >= 1.2
    plt.close(fig)


def test_dependence_panels_use_compact_spacing(fitted_forest):
    """2×2 依赖图的行列间隔不应接近单个子图本身的尺寸。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    fig = plot_model_feature_importance(
        model,
        X.head(30),
        y[:30],
        left_top_n=4,
        right_top_n=4,
        background_size=16,
        figsize=(18, 8),
        random_state=23,
        show=False,
    )

    fig.canvas.draw()
    dependence_axes = [ax for ax in fig.axes if ax.get_label() == "SHAP依赖"]
    horizontal_gap = dependence_axes[1].get_position().x0 - dependence_axes[0].get_position().x1
    vertical_gap = dependence_axes[0].get_position().y0 - dependence_axes[2].get_position().y1
    mean_width = np.mean([ax.get_position().width for ax in dependence_axes])
    mean_height = np.mean([ax.get_position().height for ax in dependence_axes])

    assert 0 < horizontal_gap / mean_width < 0.55
    assert 0 < vertical_gap / mean_height < 0.55
    plt.close(fig)


def test_feature_importance_panels_hide_all_gridlines(fitted_forest):
    """原生重要性、左侧 SHAP 和右侧依赖图都不应显示网格线。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    native_fig = plot_model_feature_importance(model, X, y=None, left_top_n=2, show=False)
    combined_fig = plot_model_feature_importance(
        model,
        X.head(12),
        y[:12],
        left_top_n=2,
        right_top_n=2,
        background_size=8,
        show=False,
    )

    native_ax = next(ax for ax in native_fig.axes if ax.get_label() == "模型特征重要性")
    main_ax = next(ax for ax in combined_fig.axes if ax.get_label() == "SHAP分布")
    dependence_axes = [ax for ax in combined_fig.axes if ax.get_label() == "SHAP依赖"]
    for axis in [native_ax, main_ax, *dependence_axes]:
        gridlines = [*axis.get_xgridlines(), *axis.get_ygridlines()]
        assert not any(line.get_visible() for line in gridlines)

    plt.close(native_fig)
    plt.close(combined_fig)


def test_left_zero_reference_line_uses_secondary_theme_color(fitted_forest):
    """左侧 SHAP 图的零值虚线必须使用副主题色。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    fig = plot_model_feature_importance(
        model,
        X.head(12),
        y[:12],
        left_top_n=2,
        show_dependence=False,
        background_size=8,
        show=False,
    )

    main_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP分布")
    zero_lines = [line for line in main_ax.lines if np.allclose(line.get_xdata(), [0.0, 0.0])]
    assert len(zero_lines) == 1
    assert to_hex(zero_lines[0].get_color(), keep_alpha=False).lower() == PRIMARY_COLORS[1].lower()
    assert zero_lines[0].get_linestyle() == "--"
    plt.close(fig)


def test_missing_y_falls_back_to_native_importance_bar_only(fitted_forest):
    """缺少 y 时误算 SHAP 或保留右侧面板会违背降级契约。"""
    model, X, _ = fitted_forest

    fig = plot_model_feature_importance(model, X, y=None, left_top_n=2, show=False)

    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "模型特征重要性")
    assert len(fig.axes) == 1
    assert len(importance_ax.patches) == 2
    assert all(patch.get_hatch() == "/" for patch in importance_ax.patches)
    assert len(importance_ax.get_yticklabels()) == 2
    assert fig.get_size_inches()[0] < 18
    plt.close(fig)


def test_callable_prediction_method_controls_shap_output(fitted_forest):
    """忽略 callable 而偷用模型默认预测方法会生成不同的 SHAP 归因。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    fig = plot_model_feature_importance(
        model,
        X.head(12),
        y[:12],
        prediction_method=lambda values: np.zeros(len(values), dtype=float),
        left_top_n=2,
        show_dependence=False,
        background_size=8,
        show=False,
        importance_source="shap",
        shap_by_label=False,
    )

    main_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP分布")
    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "特征重要性")
    shap_x = np.concatenate([collection.get_offsets()[:, 0] for collection in main_ax.collections])
    np.testing.assert_allclose(shap_x, 0.0, atol=1e-12)
    np.testing.assert_allclose([patch.get_width() for patch in importance_ax.patches], 0.0, atol=1e-12)
    assert importance_ax.get_xlabel() == "平均 |SHAP 值|"
    assert main_ax.get_title() == ""
    plt.close(fig)


def test_shap_importance_can_compare_binary_labels_or_show_aggregate(fitted_forest, monkeypatch):
    """分标签柱条应使用 0/1 主题色，也应支持退化为单组全样本柱条。"""
    shap = pytest.importorskip("shap")
    model, X, y = fitted_forest
    X_sample = X.head(12)
    y_sample = y[:12]
    assert set(y_sample) == {0, 1}

    label_zero_values = np.array([1.0, 2.0, 3.0, 4.0])
    label_one_values = np.array([4.0, 3.0, 2.0, 1.0])
    fake_shap_values = np.where(
        np.asarray(y_sample)[:, None] == 0,
        label_zero_values,
        label_one_values,
    )

    class FakeExplainer:
        def __call__(self, values):
            class Explanation:
                pass

            explanation = Explanation()
            explanation.values = fake_shap_values
            return explanation

    monkeypatch.setattr(shap, "Explainer", lambda *args, **kwargs: FakeExplainer())

    common_kwargs = {
        "model": model,
        "X": X_sample,
        "y": y_sample,
        "importance_source": "shap",
        "left_top_n": 4,
        "show_dependence": False,
        "background_size": 8,
        "show": False,
    }
    split_fig = plot_model_feature_importance(**common_kwargs)
    split_ax = next(ax for ax in split_fig.axes if ax.get_label() == "特征重要性")

    aggregate = np.mean(np.abs(fake_shap_values), axis=0)
    ranking = np.argsort(-aggregate, kind="stable")
    reversed_indices = ranking[::-1]
    label_zero_mean = np.mean(np.abs(fake_shap_values[y_sample == 0]), axis=0)
    label_one_mean = np.mean(np.abs(fake_shap_values[y_sample == 1]), axis=0)
    assert len(split_ax.patches) == 8
    np.testing.assert_allclose(
        [patch.get_width() for patch in split_ax.patches[:4]],
        label_zero_mean[reversed_indices],
    )
    np.testing.assert_allclose(
        [patch.get_width() for patch in split_ax.patches[4:]],
        label_one_mean[reversed_indices],
    )
    assert all(to_hex(patch.get_facecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower() for patch in split_ax.patches[:4])
    assert all(to_hex(patch.get_facecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[1].lower() for patch in split_ax.patches[4:])
    assert all(patch.get_alpha() < 0.5 for patch in split_ax.patches)
    assert [patch.get_hatch() for patch in split_ax.patches[:4]] == ["/"] * 4
    assert [patch.get_hatch() for patch in split_ax.patches[4:]] == ["\\"] * 4
    assert split_ax.get_legend() is None
    assert split_ax.get_xlabel() == "各标签平均 |SHAP 值|"

    aggregate_fig = plot_model_feature_importance(**common_kwargs, shap_by_label=False)
    aggregate_ax = next(ax for ax in aggregate_fig.axes if ax.get_label() == "特征重要性")
    assert len(aggregate_ax.patches) == 4
    np.testing.assert_allclose(
        [patch.get_width() for patch in aggregate_ax.patches],
        aggregate[reversed_indices],
    )
    assert aggregate_ax.get_legend() is None
    assert aggregate_ax.get_xlabel() == "平均 |SHAP 值|"
    assert all(patch.get_hatch() == "/" for patch in aggregate_ax.patches)
    plt.close(split_fig)
    plt.close(aggregate_fig)


def test_hatch_false_hides_native_and_shap_importance_patterns(fitted_forest):
    """关闭 hatch 后，原生与分标签 SHAP 重要性柱都不应残留纹理。"""
    pytest.importorskip("shap")
    model, X, y = fitted_forest

    native_fig = plot_model_feature_importance(model, X, y=None, left_top_n=2, hatch=False, show=False)
    native_ax = next(ax for ax in native_fig.axes if ax.get_label() == "模型特征重要性")
    assert all(patch.get_hatch() is None for patch in native_ax.patches)

    shap_fig = plot_model_feature_importance(
        model,
        X.head(12),
        y[:12],
        importance_source="shap",
        left_top_n=2,
        show_dependence=False,
        background_size=8,
        hatch=False,
        show=False,
    )
    shap_ax = next(ax for ax in shap_fig.axes if ax.get_label() == "特征重要性")
    assert all(patch.get_hatch() is None for patch in shap_ax.patches)
    plt.close(native_fig)
    plt.close(shap_fig)


def test_logistic_regression_model_importance_uses_absolute_coefficients(fitted_forest):
    """逻辑回归的模型重要性必须来自 coef，而不是额外计算 SHAP。"""
    _, X, y = fitted_forest
    model = LogisticRegression(max_iter=500).fit(X, y)

    fig = plot_model_feature_importance(model, X, y=None, left_top_n=3, show=False)

    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "模型特征重要性")
    np.testing.assert_allclose(
        [patch.get_width() for patch in importance_ax.patches],
        np.sort(np.abs(model.coef_[0]))[-3:],
    )
    plt.close(fig)


@pytest.mark.parametrize("importance_source", ["model", "permutation"])
def test_unknown_importance_source_is_rejected_in_chinese(fitted_forest, importance_source):
    """重要性来源拼错时应直接给出中文参数提示。"""
    model, X, _ = fitted_forest

    with pytest.raises(ValueError, match="importance_source"):
        plot_model_feature_importance(model, X, importance_source=importance_source)


def test_unknown_prediction_method_is_rejected_in_chinese(fitted_forest):
    """拼错预测入口时不能静默回退到另一种模型输出。"""
    model, X, y = fitted_forest

    with pytest.raises(ValueError, match="预测方法"):
        plot_model_feature_importance(model, X.head(10), y[:10], prediction_method="decision_function")
