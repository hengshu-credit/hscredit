"""普通分支上限与独立缺失分支、旧参数的兼容契约。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit import MultiwayTreeExtractor
from hscredit.exceptions import InputValidationError


def data(categorical=False, missing=True):
    values = np.repeat(list("ABCD"), 30).tolist() if categorical else np.arange(120, dtype=float).tolist()
    labels = np.repeat(np.arange(4), 30).tolist()
    if missing:
        values.append(None if categorical else np.nan)
        labels.append(4)
    return pd.DataFrame({"特征": values, "target": labels})


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("limit", [None, 4])
def test_auto_four_ordinary_branches_plus_independent_missing(categorical, missing, limit):
    frame = data(categorical, missing)
    tree = MultiwayTreeExtractor(max_n_bins=limit, missing_separate=True, criterion="gini", max_depth=1).fit(frame)
    root = tree.nodes_[0]
    assert len(root.children) == 5
    assert len([b for b in root.branches if b.kind != "missing"]) == 4
    assert root.missing_separate
    assert tree.nodes_[root.children[root.missing_branch]].counts.sum() == int(missing)
    np.testing.assert_array_equal(tree.predict(frame), frame.target)
    assert clone(tree).get_params() == tree.get_params()


@pytest.mark.parametrize("categorical", [False, True])
def test_manual_split_and_missing_marker_do_not_consume_normal_bin(categorical):
    frame = data(categorical)
    tree = MultiwayTreeExtractor(max_n_bins=4, missing_separate=True, max_depth=1).fit(frame)
    rule = (
        {"category_groups": [["A"], [None], ["B"], ["C"], ["D"]]} if categorical else {"thresholds": [29.5, 59.5, 89.5]}
    )
    tree.manual_split(feature="特征", **rule)
    assert len(tree.nodes_[0].children) == 5
    if categorical:
        tree.set_params(max_n_bins=3)
        with pytest.raises(InputValidationError, match="分支数"):
            tree.manual_split(feature="特征", **rule)
    else:
        tree.set_thresholds(0, [28.5, 58.5, 88.5])
        assert len(tree.nodes_[0].children) == 5
        with pytest.raises(InputValidationError, match="分支数"):
            tree.set_thresholds(0, [10.5, 28.5, 58.5, 88.5])
        tree.manual_split(feature="特征", thresholds=[29.5, np.nan, 59.5, 89.5])
        assert len(tree.nodes_[0].children) == 4
        assert not tree.nodes_[0].missing_separate


def test_alias_precedence_set_params_and_old_pickle(tmp_path):
    frame = data()
    old = MultiwayTreeExtractor(max_branches=4, missing_separate=True, criterion="gini", max_depth=1).fit(frame)
    assert len(old.nodes_[0].children) == 4  # 显式旧参数保留总数含缺失口径。
    del old.max_n_bins
    path = old.save(tmp_path / "旧模型.pkl")
    loaded = MultiwayTreeExtractor.load(path)
    assert loaded.max_n_bins is None
    loaded.fit(frame)
    assert len(loaded.nodes_[0].children) == 4
    loaded.set_params(max_n_bins=4).fit(frame)
    assert len(loaded.nodes_[0].children) == 5
    loaded.set_params(max_n_bins=2).fit(frame)
    assert len(loaded.nodes_[0].children) == 3


@pytest.mark.parametrize("categorical", [False, True])
def test_one_normal_bin_can_split_only_missing(categorical):
    frame = data(categorical)
    tree = MultiwayTreeExtractor(max_n_bins=1, missing_separate=True, criterion="gini", max_depth=1).fit(frame)
    root = tree.nodes_[0]
    assert len(root.children) == 2
    assert sorted(tree.nodes_[n].counts.sum() for n in root.children) == [1, 120]
    tree.set_params(missing_separate=False).fit(frame)
    assert not tree.nodes_[0].children


@pytest.mark.parametrize("limit", [0, -1, 2.5, True, "4"])
def test_invalid_normal_branch_limit(limit):
    with pytest.raises(InputValidationError, match="max_n_bins"):
        MultiwayTreeExtractor(max_n_bins=limit).fit(data())
