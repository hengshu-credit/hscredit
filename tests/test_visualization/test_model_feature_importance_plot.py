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
    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP重要性")
    dependence_axes = [ax for ax in fig.axes if ax.get_label() == "SHAP依赖"]
    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]

    assert len(main_ax.get_yticklabels()) == 3
    assert len(dependence_axes) == 4
    assert len(colorbar_axes) == 2
    assert all(not ax.spines["outline"].get_visible() for ax in colorbar_axes)
    assert [ax.get_title().split(":", 1)[0] for ax in dependence_axes] == ["Top 1", "Top 2", "Top 3", "Top 4"]
    assert importance_ax.patches
    assert to_hex(importance_ax.patches[0].get_facecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()
    assert importance_ax.patches[0].get_alpha() < 0.5
    assert to_hex(main_ax.spines["bottom"].get_edgecolor(), keep_alpha=False).lower() == PRIMARY_COLORS[0].lower()

    shap_scatter = main_ax.collections[0]
    assert shap_scatter.cmap(0.0)[2] > shap_scatter.cmap(0.0)[0]
    assert shap_scatter.cmap(1.0)[0] > shap_scatter.cmap(1.0)[2]

    width, height = fig.get_size_inches()
    assert width >= 18
    assert height > 6
    fig.canvas.draw()
    first_box = dependence_axes[0].get_position()
    physical_aspect = first_box.width * width / (first_box.height * height)
    assert physical_aspect >= 1.2
    plt.close(fig)


def test_missing_y_falls_back_to_native_importance_bar_only(fitted_forest):
    """缺少 y 时误算 SHAP 或保留右侧面板会违背降级契约。"""
    model, X, _ = fitted_forest

    fig = plot_model_feature_importance(model, X, y=None, left_top_n=2, show=False)

    importance_ax = next(ax for ax in fig.axes if ax.get_label() == "模型特征重要性")
    assert len(fig.axes) == 1
    assert len(importance_ax.patches) == 2
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
    )

    main_ax = next(ax for ax in fig.axes if ax.get_label() == "SHAP分布")
    shap_x = np.concatenate([collection.get_offsets()[:, 0] for collection in main_ax.collections])
    np.testing.assert_allclose(shap_x, 0.0, atol=1e-12)
    plt.close(fig)


def test_unknown_prediction_method_is_rejected_in_chinese(fitted_forest):
    """拼错预测入口时不能静默回退到另一种模型输出。"""
    model, X, y = fitted_forest

    with pytest.raises(ValueError, match="预测方法"):
        plot_model_feature_importance(model, X.head(10), y[:10], prediction_method="decision_function")
