"""中文模型解释图形 smoke tests。"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from hscredit import ValidationError
from hscredit.core.models.explainability import (
    ModelExplainer,
    plot_feature_importance,
    plot_importance_comparison,
    plot_shap_importance,
)


@pytest.fixture()
def plotted():
    values, y = make_classification(n_samples=55, n_features=4, random_state=17)
    X = pd.DataFrame(values, columns=["甲", "乙", "丙", "丁"])
    model = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=17).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(15), random_state=17)
    return explainer, explainer.explain(X.tail(12))


@pytest.mark.parametrize(
    "method_name",
    ["plot_decision", "plot_heatmap", "plot_correlation", "plot_feature_clustering", "plot_importance_overview", "plot_explanation_overview"],
)
def test_new_plots_return_figures_without_showing(method_name, plotted, monkeypatch):
    explainer, result = plotted
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    figure = getattr(explainer, method_name)(result, show=False)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert shown == []


def test_distribution_and_interaction_plots(plotted):
    explainer, result = plotted
    assert isinstance(explainer.plot_distribution(result, feature="甲", show=False), matplotlib.figure.Figure)
    assert isinstance(explainer.plot_interaction_heatmap(result, show=False), matplotlib.figure.Figure)
    assert isinstance(explainer.plot_interaction_bubble(result, show=False), matplotlib.figure.Figure)
    figure = explainer.plot_importance_overview(result, show=False)
    assert any("SHAP" in axis.get_title() and "重要性" in axis.get_title() for axis in figure.axes)


def test_importance_comparison_has_two_truthful_panels_and_honors_overrides(plotted):
    """传统与 SHAP 对比入口必须可调用且不能丢失画布和标题。"""
    explainer, result = plotted

    figure = plot_importance_comparison(
        explainer.model,
        result.data,
        top_n=3,
        figsize=(8, 4),
        title="重要性比较",
        show=False,
    )

    assert tuple(figure.get_size_inches()) == pytest.approx((8, 4))
    assert {axis.get_title() for axis in figure.axes} >= {"原生特征重要性", "SHAP特征重要性"}
    assert figure._suptitle.get_text() == "重要性比较"


def test_shap_importance_honors_figure_size_and_title(plotted):
    """SHAP 便捷绘图不能把覆盖参数吞进未使用的 kwargs。"""
    explainer, result = plotted

    figure = plot_shap_importance(
        explainer.model,
        result.data,
        top_n=2,
        figsize=(6, 3),
        title="SHAP重要性覆盖",
        show=False,
    )

    assert tuple(figure.get_size_inches()) == pytest.approx((6, 3))
    assert figure.axes[0].get_title() == "SHAP重要性覆盖"


def test_feature_importance_honors_explicit_color(plotted):
    """传统重要性图的 color 参数必须决定条形颜色。"""
    explainer, _ = plotted

    figure = plot_feature_importance(explainer.model, top_n=2, color="#ff0000", show=False)

    assert figure.axes[0].patches[0].get_facecolor() == pytest.approx(to_rgba("#ff0000"))


def test_named_waterfall_and_force_methods_return_named_visuals(plotted):
    """waterfall/force 兼容名不能继续返回普通决策条形图。"""
    explainer, result = plotted

    waterfall = explainer.plot_shap_waterfall(result, sample_idx=0, show=False)
    force = explainer.plot_shap_force(result, sample_idx=0, show=False)

    assert any("瀑布" in axis.get_title() for axis in waterfall.axes)
    assert any("力图" in axis.get_title() for axis in force.axes)


def test_plot_limits_reject_zero_before_matplotlib_receives_empty_arrays(plotted):
    """展示数量为零时应得到项目错误而非 NumPy/Matplotlib 底层异常。"""
    explainer, result = plotted

    with pytest.raises(ValidationError, match="max_display"):
        explainer.plot_heatmap(result, max_display=0, show=False)


def test_plot_decision_rejects_out_of_range_positions(plotted):
    """负位置和越界位置不能被 NumPy 静默解释或抛底层 IndexError。"""
    explainer, result = plotted

    for position in (-1, len(result.data)):
        with pytest.raises(ValidationError, match="样本位置"):
            explainer.plot_decision(result, position=position, show=False)


def test_summary_bar_reuses_existing_explanation_result(plotted):
    """bar 分支收到 ExplanationResult 时不能再次把它当二维输入解释。"""
    explainer, result = plotted

    figure = explainer.plot_shap_summary(result, plot_type="bar", show=False)

    assert isinstance(figure, matplotlib.figure.Figure)
    assert figure.axes[0].get_title() == "SHAP特征重要性"
