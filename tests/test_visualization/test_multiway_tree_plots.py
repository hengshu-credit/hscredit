"""复用树图后端的多叉分支、布局与多分类分布验证。"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hscredit import MultiwayTreeExtractor, plot_tree
from hscredit.core.viz.tree_plots import _extract_tree_data, _reingold_tilford_layout


@pytest.mark.parametrize("multiclass", [False, True])
def test_all_backends_keep_every_branch_and_class_label(multiclass):
    labels = ["正常", "逾期", "结清"] if multiclass else [0, 1, 0]
    data = pd.DataFrame({"类型": ['A"类别', "B", "C"] * 30, "target": labels * 30})
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(feature="类型", category_groups=[['A"类别'], ["B"], ["C"]])
    adapted = _extract_tree_data(tree)
    assert len(adapted["nodes"]) == 4 and len(adapted["edges"]) == 3
    positions = _reingold_tilford_layout(adapted["nodes"], adapted["edges"])
    child_x = [positions[child][0] for child in tree.nodes_[0].children]
    assert child_x == sorted(child_x) and len(set(child_x)) == 3
    assert positions[0][0] == pytest.approx(np.mean([child_x[0], child_x[-1]]))
    figure = plot_tree(tree)
    texts = [text.get_text() for ax in figure.axes for text in ax.texts]
    assert any("或缺失" in text for text in texts)
    if multiclass:
        assert all(label in texts for label in labels)
    plt.close(figure)
    chart = plot_tree(tree, backend="pyecharts")
    options = json.loads(chart.dump_options())
    assert len(options["series"][0]["links"]) == 3
    source = plot_tree(tree, backend="graphviz").source
    assert source.count(" -> ") == 3
    assert '\\"' in source
    if multiclass:
        assert all(label in source for label in labels)


@pytest.mark.parametrize("categorical", [False, True])
def test_separate_missing_has_its_own_edge_in_every_backend(categorical):
    values = ["A", "B", None] if categorical else [0.5, 1.5, np.nan]
    data = pd.DataFrame({"特征": values * 10, "target": [0, 1, 1] * 10})
    tree = MultiwayTreeExtractor(max_depth=1).fit(data)
    kwargs = {"category_groups": [["A"], [np.nan], ["B"]]} if categorical else {"thresholds": [1.0]}
    tree.manual_split(feature="特征", missing_separate=True, **kwargs)
    adapted = _extract_tree_data(tree)
    assert len(adapted["edges"]) == 3
    assert sum(edge["label"] == "缺失" for edge in adapted["edges"]) == 1
    figure = plot_tree(tree)
    assert "缺失" in [text.get_text() for ax in figure.axes for text in ax.texts]
    plt.close(figure)
    options = json.loads(plot_tree(tree, backend="pyecharts").dump_options())
    assert len(options["series"][0]["links"]) == 3
    source = plot_tree(tree, backend="graphviz").source
    assert source.count(" -> ") == 3 and "缺失" in source
