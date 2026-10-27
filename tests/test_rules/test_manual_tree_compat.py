"""手工树版本判断、完整展示与原始行统计回归。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

from hscredit.report.mining import DecisionTreeAnalyzer, ManualTreeExtractor
from hscredit.report.mining import manual_tree_extractor as manual_tree
from hscredit.utils import style_rule_table


@pytest.mark.parametrize(
    "version, expected", [("0.24.2", False), ("1.2.2", False), ("1.3.0", True), ("1.6.1", True), ("1.10.0rc1", True)]
)
def test_native_missing_uses_parsed_version(monkeypatch, version, expected):
    monkeypatch.setattr(manual_tree.sklearn, "__version__", version)
    assert manual_tree._sklearn_supports_native_missing() is expected


@pytest.fixture
def tree():
    data = pd.DataFrame({"评分": [1, 2, 3, 4, 5, 6], "逾期": [1, 0, 1, 0, 0, 0]})
    return ManualTreeExtractor(target="逾期", max_depth=1, min_samples_split=2, min_samples_leaf=1, n_jobs=1).fit(
        data, features=["评分"]
    )


def test_display_renders_both_figure_and_html(monkeypatch, tree):
    outputs = []

    def render(value):
        if isinstance(value, Styler):
            outputs.append(value.to_html())
        else:
            outputs.append(value)

    monkeypatch.setattr("IPython.display.display", render)
    assert tree.display() is tree
    assert isinstance(outputs[0], Figure)
    assert not plt.fignum_exists(outputs[0].number)
    assert "是否叶子" in outputs[1] and "background-color:" in outputs[1]


def test_display_does_not_swallow_render_error_and_closes_figure(monkeypatch, tree):
    figures = []

    def fail(figure):
        figures.append(figure)
        raise RuntimeError("绘图展示失败")

    monkeypatch.setattr("IPython.display.display", fail)
    with pytest.raises(RuntimeError, match="绘图展示失败"):
        tree.display()
    assert not plt.fignum_exists(figures[0].number)


def test_display_does_not_swallow_table_error(monkeypatch, tree):
    def fail(*args, **kwargs):
        raise RuntimeError("规则表展示失败")

    monkeypatch.setattr("IPython.display.display", lambda value: None)
    monkeypatch.setattr(manual_tree, "style_rule_table", fail)
    with pytest.raises(RuntimeError, match="规则表展示失败"):
        tree.display()


def test_real_data_manual_split_table_matches_original_rows(tmp_path):
    """970 条真实数据：训练、手工/自动分裂、规则表、图形和保存加载。"""
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("演示工作簿 examples/hscredit_yyp.xlsx 不存在")
    data = pd.read_excel(path)
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    analyzer = DecisionTreeAnalyzer(target="FPD", features=features, missing=-999, n_jobs=1).fit(data)
    model_path = tmp_path / "决策树.pkl"
    analyzer.save(model_path)
    loaded = DecisionTreeAnalyzer.load(model_path)
    np.testing.assert_allclose(loaded.predict_proba(data), analyzer.predict_proba(data))
    assert "坏样本率" in style_rule_table(analyzer.get_rule_table()).to_html()

    tree = ManualTreeExtractor(target="FPD", features=features, missing=-999, n_jobs=1).fit(data)
    threshold = float(data[features[0]].median())
    tree.manual_split(data, feature=features[0], threshold=threshold, node=0)
    table = tree.get_rule_table()
    left = data[features[0]].fillna(-999) <= threshold
    for node, mask in ((1, left), (2, ~left)):
        row = table.loc[table["节点编号"] == node].iloc[0]
        assert row["样本总数"] == int(mask.sum())
        assert row["坏样本数"] == int(data.loc[mask, "FPD"].sum())
    assert table.loc[table["是否叶子"] == "是", "样本总数"].sum() == len(data)
    assert "是否叶子" in style_rule_table(table).to_html()
    tree.manual_split(data, feature=features[1], threshold=None, node=0)
    assert "坏样本率" in style_rule_table(tree.get_rule_table()).to_html()
    fig = tree.plot()
    assert isinstance(fig, Figure)
    plt.close(fig)
