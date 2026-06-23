import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier

from hscredit.core.viz import plot_tree, plot_tree_matplotlib
from hscredit.report.mining import DecisionTreeAnalyzer, ManualTreeExtractor


def _tree_data():
    return pd.DataFrame(
        {
            "score": [580, 610, 650, 690, 720, 760, 800, 830],
            "multi": [6, 5, 4, 3, 2, 2, 1, 1],
            "target": [1, 1, 1, 0, 0, 0, 0, 0],
        }
    )


def test_plot_tree_supports_sklearn_decision_tree_classifier():
    df = _tree_data()
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(df[["score", "multi"]], df["target"])

    fig = plot_tree_matplotlib(clf, feature_names=["score", "multi"])

    assert isinstance(fig, Figure)
    assert fig.axes
    plt.close(fig)


def test_decision_tree_analyzer_plot_method_returns_figure():
    df = _tree_data()
    analyzer = DecisionTreeAnalyzer(target="target", features=["score", "multi"], tree_params={"max_depth": 2})
    analyzer.fit(df)

    fig = analyzer.plot()

    assert isinstance(fig, Figure)
    assert fig.axes
    plt.close(fig)


def test_manual_tree_extractor_plot_method_returns_figure_after_manual_split():
    df = _tree_data()
    extractor = ManualTreeExtractor(target="target", max_depth=1, min_samples_leaf=1)
    extractor.fit(df, features=["score", "multi"])
    extractor.manual_split(df, feature="score", threshold=700, node=0)

    fig = extractor.plot()

    assert isinstance(fig, Figure)
    assert fig.axes
    plt.close(fig)


def test_unified_plot_tree_supports_decision_tree_analyzer():
    df = _tree_data()
    analyzer = DecisionTreeAnalyzer(target="target", features=["score", "multi"], tree_params={"max_depth": 2})
    analyzer.fit(df)

    fig = plot_tree(analyzer, backend="matplotlib")

    assert isinstance(fig, Figure)
    assert fig.axes
    plt.close(fig)
