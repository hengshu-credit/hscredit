"""多叉树与 binner 一致的缺失声明、独立路由和持久化契约。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit import MultiwayTreeExtractor, OptimalBinning
from hscredit.exceptions import InputValidationError


def numeric_data():
    return pd.DataFrame({"评分": [0.5, 1.5, 2.5, np.nan] * 10, "target": [0, 1, 0, 1] * 10})


def assert_partition(tree, data):
    masks = []
    for node, rule in zip(tree.get_leaf_node_ids(), tree.get_rules()):
        mask = rule.predict(data).to_numpy(dtype=bool)
        np.testing.assert_array_equal(mask, tree.apply(data) == node)
        masks.append(mask)
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(data)))


@pytest.mark.parametrize("marker", [np.nan, None, pd.NA])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_numeric_marker_position_matches_binner_and_raw_rows(marker, position):
    data = numeric_data()
    splits = [1.0, 2.0]
    splits.insert(position, marker)
    binner = OptimalBinning(
        method="quantile",
        user_splits={"评分": splits},
        user_splits_fixed=True,
        max_n_bins=3,
        min_n_bins=1,
        min_bin_size=None,
        n_jobs=1,
    ).fit(data[["评分"]], data.target)
    tree = MultiwayTreeExtractor(max_depth=1, missing_separate=True).fit(data)
    tree.manual_split(feature="评分", thresholds=splits)
    root = tree.nodes_[0]
    assert not root.missing_separate and root.missing_branch == position
    assert len(root.children) == 3
    binned = binner.transform(data, metric="indices")["评分"].to_numpy()
    routed = np.array([root.children.index(node) for node in tree.apply(data)])
    np.testing.assert_array_equal(routed, binned)
    expected = np.searchsorted([1.0, 2.0], data["评分"].to_numpy(), side="left")
    expected[data["评分"].isna()] = position
    np.testing.assert_array_equal(routed, expected)
    for branch, child in enumerate(root.children):
        selected = data.loc[expected == branch, "target"]
        np.testing.assert_array_equal(tree.nodes_[child].counts, [(selected == 0).sum(), (selected == 1).sum()])
    assert_partition(tree, data)


def test_screenshot_trailing_nan_explicit_branch_wins():
    data = pd.DataFrame({"评分": [0.1, 0.15, 0.2, 0.3, np.nan] * 10, "target": [0, 1, 0, 1, 1] * 10})
    tree = MultiwayTreeExtractor(max_depth=1, max_branches=4).fit(data)
    tree.manual_split(feature="评分", thresholds=[0.12, 0.18, 0.25, np.nan], missing_branch=0)
    assert len(tree.nodes_[0].children) == 4
    assert not tree.nodes_[0].missing_separate
    assert tree.apply(data.loc[data["评分"].isna()]).tolist() == [tree.nodes_[0].children[0]] * 10
    assert_partition(tree, data)


@pytest.mark.parametrize("nullable", [False, True])
def test_dedicated_numeric_branch_is_only_missing_and_boundary_exact(nullable):
    data = numeric_data()
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(feature="评分", thresholds=[1.0, 2.0], missing_separate=True)
    root = tree.nodes_[0]
    assert root.missing_separate and root.missing_branch == 3
    assert root.branches[3].kind == "missing"
    raw = pd.DataFrame({"评分": [1.0, np.nextafter(1.0, np.inf), 2.0, np.nan, -np.inf, np.inf]})
    if nullable:
        raw["评分"] = raw["评分"].astype("Float64")
    assert tree.apply(raw).tolist() == [root.children[i] for i in [0, 1, 1, 3, 0, 2]]
    assert_partition(tree, raw.set_axis(["重复"] * len(raw)))
    table = tree.get_rule_table(leaf_only=True)
    missing_row = table.loc[table["节点编号"] == root.children[3]].iloc[0]
    assert missing_row["样本总数"] == 10 and missing_row["坏样本数"] == 10


@pytest.mark.parametrize("feature, values", [("评分", [0.5, 1.5, 2.5]), ("类别", ["A", "B", "C"])])
def test_auto_separate_keeps_rare_missing_and_obeys_total_branch_limit(feature, values):
    data = pd.DataFrame(
        {feature: np.repeat(values, 10).tolist() + [None], "target": [0] * 10 + [1] * 10 + [0] * 10 + [1]}
    )
    tree = MultiwayTreeExtractor(
        max_depth=1,
        max_branches=4,
        min_samples_split=2,
        min_samples_leaf=10,
        criterion="gini",
        missing_separate=True,
    ).fit(data)
    root = tree.nodes_[0]
    assert root.missing_separate
    assert 2 <= len(root.children) <= 4
    assert tree.nodes_[root.children[root.missing_branch]].counts.sum() == 1
    assert all(
        tree.nodes_[child].counts.sum() >= 10 for i, child in enumerate(root.children) if i != root.missing_branch
    )
    assert clone(tree).missing_separate is True
    assert_partition(tree, data)


@pytest.mark.parametrize("values", [[1.0] * 20 + [None], ["A"] * 20 + [None]])
def test_missing_vs_constant_can_be_learned_without_ordinary_cut(values):
    data = pd.DataFrame({"特征": values, "target": [0] * 20 + [1]})
    tree = MultiwayTreeExtractor(missing_separate=True, min_samples_split=2).fit(data)
    assert len(tree.nodes_[0].children) == 2
    np.testing.assert_array_equal(tree.predict(data), data.target)
    assert_partition(tree, data)


def test_future_missing_empty_leaf_inherits_parent_probability():
    data = numeric_data().dropna()
    tree = MultiwayTreeExtractor(missing_separate=True, criterion="gini", max_depth=1).fit(data)
    root = tree.nodes_[0]
    assert root.children and root.missing_separate
    assert tree.nodes_[root.children[root.missing_branch]].counts.sum() == 0
    np.testing.assert_allclose(tree.predict_proba(pd.DataFrame({"评分": [np.nan]}))[0], [2 / 3, 1 / 3])
    assert_partition(tree, pd.DataFrame({"评分": [None, 0.5, 1.5]}))


def test_older_model_without_missing_config_keeps_merge_default(tmp_path):
    tree = MultiwayTreeExtractor().fit(numeric_data())
    del tree.__dict__["missing_separate"]
    for node in tree.nodes_.values():
        node.__dict__.pop("missing_separate", None)
        node.__dict__.pop("unknown_branch", None)
    assert clone(tree).missing_separate is False
    path = tmp_path / "旧字段布局.pkl"
    tree.save(path)
    restored = MultiwayTreeExtractor.load(path)
    assert restored.get_params()["missing_separate"] is False
    np.testing.assert_allclose(restored.predict_proba(numeric_data()), tree.predict_proba(numeric_data()))
    restored.manual_split(feature="评分", thresholds=[1.0, 2.0], missing_separate=True)
    assert restored.nodes_[0].missing_separate


def test_categorical_missing_group_and_unknown_are_distinct():
    data = pd.DataFrame({"类别": ["A"] * 20 + ["B", "C", None] * 5, "target": [0] * 20 + [1, 0, 1] * 5})
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(feature="类别", category_groups=[["A"], [np.nan], ["B", "C"]])
    root = tree.nodes_[0]
    assert root.missing_branch == 1 and root.missing_separate and root.unknown_branch == 0
    raw = pd.DataFrame({"类别": pd.Series([None, "未知", "A", "B", "C"], dtype="string")})
    assert tree.apply(raw).tolist() == [root.children[i] for i in [1, 0, 0, 2, 2]]
    assert_partition(tree, raw)
    tree.manual_split(feature="类别", category_groups=[["A"], ["B", "C", np.nan]])
    assert not tree.nodes_[0].missing_separate and tree.nodes_[0].missing_branch == 1
    assert_partition(tree, raw)
    tree.manual_split(feature="类别", category_groups=[["A"], ["B", "C", np.nan]], missing_branch=0)
    assert tree.nodes_[0].missing_branch == 0
    assert_partition(tree, raw)


def test_categories_without_marker_can_request_separate_missing():
    data = pd.DataFrame({"类别": ["A", "B", None] * 10, "target": [0, 1, 1] * 10})
    tree = MultiwayTreeExtractor().fit(data)
    tree.manual_split(feature="类别", category_groups=[["A"], ["B"]], missing_separate=True)
    assert len(tree.nodes_[0].children) == 3 and tree.nodes_[0].missing_branch == 2
    assert_partition(tree, data)


def test_node_edits_and_roundtrip_keep_missing_policy(tmp_path):
    data = numeric_data()
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(feature="评分", thresholds=[1.0, 2.0], missing_separate=True)
    tree.set_thresholds(0, [1.2])
    assert tree.nodes_[0].missing_separate and tree.nodes_[0].missing_branch == 2
    tree.grow()
    assert_partition(tree, data)
    path = tmp_path / "含独立缺失分支.pkl"
    tree.save(path)
    restored = MultiwayTreeExtractor.load(path)
    assert restored.nodes_[0].missing_separate
    np.testing.assert_allclose(restored.predict_proba(data), tree.predict_proba(data))
    assert_partition(restored, data)
    restored.set_thresholds(0, [np.nan, 1.0, 2.0])
    assert not restored.nodes_[0].missing_separate and restored.nodes_[0].missing_branch == 0
    restored.set_thresholds(0, [1.1, 2.1])
    assert not restored.nodes_[0].missing_separate and restored.nodes_[0].missing_branch == 0
    restored.prune(0)
    assert not restored.nodes_[0].missing_separate and not restored.nodes_[0].children


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thresholds": [1, np.nan, 2, None]},
        {"thresholds": [1, 2, 3], "missing_separate": True},
        {"thresholds": [1, 2], "missing_branch": 5},
        {"thresholds": [1, 2], "missing_branch": True},
        {"thresholds": [1, 2], "missing_separate": "yes"},
    ],
)
def test_invalid_missing_configuration_is_atomic(kwargs):
    tree = MultiwayTreeExtractor().fit(numeric_data())
    original = tree.get_node_table()
    probability = tree.predict_proba(numeric_data())
    with pytest.raises(InputValidationError):
        tree.manual_split(feature="评分", **kwargs)
    pd.testing.assert_frame_equal(tree.get_node_table(), original)
    np.testing.assert_array_equal(tree.predict_proba(numeric_data()), probability)


def test_dedicated_category_group_conflict_does_not_mutate_tree():
    data = pd.DataFrame({"类别": ["A", "B", None] * 10, "target": [0, 1, 1] * 10})
    tree = MultiwayTreeExtractor().fit(data)
    original = tree.get_node_table()
    with pytest.raises(InputValidationError, match="冲突"):
        tree.manual_split(feature="类别", category_groups=[["A"], [None], ["B"]], missing_branch=0)
    pd.testing.assert_frame_equal(tree.get_node_table(), original)


def test_real_data_missing_node_counts_and_amount_report():
    path = Path(__file__).resolve().parents[2] / "examples/hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    data = pd.read_excel(path)
    feature = "衡枢鉴真分老客版"
    data.loc[data.index[:8], feature] = np.nan
    tree = MultiwayTreeExtractor(target="FPD", features=[feature], max_depth=2).fit(data)
    tree.manual_split(feature=feature, thresholds=[0.12, 0.20], missing_separate=True)
    mask = data[feature].isna()
    leaf = tree.nodes_[0].children[-1]
    row = tree.get_rule_table(leaf_only=True).set_index("节点编号").loc[leaf]
    assert row["样本总数"] == mask.sum()
    assert row["坏样本数"] == data.loc[mask, "FPD"].sum()
    assert tree.get_rule_table(leaf_only=True)["样本总数"].sum() == len(data) == 970
    report = tree.report(data, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额", leaf_only=True)
    assert not report.empty
    assert_partition(tree, data)
