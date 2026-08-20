"""中文模型解释图形 smoke tests。"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from hscredit.core.models.evaluation import ModelExplainer


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
