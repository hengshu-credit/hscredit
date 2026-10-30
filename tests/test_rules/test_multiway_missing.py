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
    tree = MultiwayTreeExtractor(max_depth=1, max_n_bins=4).fit(data)
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
        max_n_bins=4,
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


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("method", ["fit", "manual", "automatic_edit", "grow"])
@pytest.mark.parametrize("criterion", ["chi2", "gini", "entropy"])
def test_future_missing_defaults_left_when_node_training_has_no_missing(categorical, method, criterion, tmp_path):
    values = ["A", "B", "C"] if categorical else [0.5, 1.5, 2.5]
    data = pd.DataFrame({"特征": np.repeat(values, [10, 20, 30]), "target": [0] * 10 + [1] * 50})
    tree = MultiwayTreeExtractor(
        max_depth=1 if method == "fit" else 0, min_samples_leaf=1, criterion=criterion, n_jobs=1
    ).fit(data)
    tree.set_params(max_depth=1)
    if method == "manual":
        kwargs = {"category_groups": [[value] for value in values]} if categorical else {"thresholds": [1.0, 2.0]}
        tree.manual_split(feature="特征", **kwargs)
    elif method == "automatic_edit":
        tree.manual_split(feature="特征")
    elif method == "grow":
        tree.grow()
    root = tree.nodes_[0]
    assert len(root.children) >= 2
    assert tree.nodes_[root.children[0]].counts.sum() < max(tree.nodes_[c].counts.sum() for c in root.children)
    assert root.missing_branch == 0 and not root.missing_separate
    assert tree.get_split_candidates()["缺失分支"].eq(0).all()
    future = pd.DataFrame({"特征": pd.Series([None, np.nan, pd.NA], dtype="string" if categorical else "Float64")})
    assert tree.apply(future).tolist() == [root.children[0]] * len(future)
    # 最左分支来自原始 A / 0.5 行，标签全为 0，不能使用最大分支或父节点的概率。
    np.testing.assert_array_equal(tree.predict_proba(future), [[1.0, 0.0]] * len(future))
    assert_partition(tree, future)
    matched, _ = tree.to_ruleset(nodes=[root.children[0]]).evaluate(future)
    assert matched.all()
    path = tmp_path / "训练无缺失.pkl"
    tree.save(path)
    restored = MultiwayTreeExtractor.load(path)
    np.testing.assert_array_equal(restored.apply(future), tree.apply(future))
    assert_partition(restored, future)


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("policy", ["marker", "explicit", "separate"])
def test_future_missing_respects_explicit_policy_without_training_missing(categorical, policy):
    values = ["A", "B", "C"] if categorical else [0.5, 1.5, 2.5]
    data = pd.DataFrame({"特征": values * 10, "target": [0, 1, 0] * 10})
    tree = MultiwayTreeExtractor(max_depth=1, n_jobs=1).fit(data)
    kwargs = {"category_groups": [[v] for v in values]} if categorical else {"thresholds": [1.0, 2.0]}
    if policy == "marker":
        if categorical:
            kwargs["category_groups"][1].append(np.nan)
        else:
            kwargs["thresholds"].insert(1, np.nan)
    elif policy == "explicit":
        kwargs["missing_branch"] = 1
    else:
        kwargs["missing_separate"] = True
    tree.manual_split(feature="特征", **kwargs)
    root = tree.nodes_[0]
    destination = 3 if policy == "separate" else 1
    future = pd.DataFrame({"特征": [None, np.nan]})
    assert root.missing_branch == destination
    assert tree.apply(future).tolist() == [root.children[destination]] * 2
    np.testing.assert_allclose(tree.predict_proba(future), [[2 / 3, 1 / 3] if policy == "separate" else [0, 1]] * 2)
    assert_partition(tree, future)


@pytest.mark.parametrize("method", ["manual", "grow"])
def test_missing_default_uses_local_node_rows_and_survives_edits(method):
    data = pd.DataFrame(
        {
            "分组": ["节点"] * 60 + ["其他"] * 10,
            "评分": [0.5] * 10 + [1.5] * 20 + [2.5] * 30 + [np.nan] * 10,
            "target": [0] * 10 + [1] * 20 + [0] * 30 + [1] * 10,
        }
    )
    tree = MultiwayTreeExtractor(max_depth=2, criterion="gini", min_samples_leaf=1).fit(data)
    tree.manual_split(feature="分组", category_groups=[["节点"], ["其他"]])
    node = tree.nodes_[0].children[0]
    if method == "manual":
        tree.manual_split(feature="评分", thresholds=[1.0, 2.0], node=node)
    else:
        tree.grow(node=node)
    future = pd.DataFrame({"分组": ["节点"], "评分": [np.nan]})
    for thresholds in (None, [0.9, 1.9]):
        if thresholds is not None:
            tree.set_thresholds(node, thresholds)
        assert tree.nodes_[node].missing_branch == 0
        assert tree.apply(future).tolist() == [tree.nodes_[node].children[0]]
        np.testing.assert_array_equal(tree.predict_proba(future), [[1.0, 0.0]])
        assert_partition(tree, future)


def test_future_missing_in_empty_left_branch_uses_ancestor_probability():
    data = numeric_data().dropna()
    tree = MultiwayTreeExtractor(max_depth=1).fit(data).manual_split(feature="评分", thresholds=[0.0, 2.0])
    root = tree.nodes_[0]
    assert root.missing_branch == 0
    assert tree.nodes_[root.children[0]].counts.sum() == 0
    future = pd.DataFrame({"评分": [np.nan]})
    assert tree.apply(future).tolist() == [root.children[0]]
    np.testing.assert_allclose(tree.predict_proba(future), [[2 / 3, 1 / 3]])
    assert_partition(tree, future)


@pytest.mark.parametrize("categorical", [False, True])
def test_training_missing_still_learns_nonleft_destination(categorical):
    values = ["A", "B"] if categorical else [0.5, 1.5]
    data = pd.DataFrame({"特征": values * 20 + [None] * 10, "target": [0, 1] * 20 + [1] * 10})
    tree = MultiwayTreeExtractor(max_depth=1, criterion="gini").fit(data)
    root = tree.nodes_[0]
    assert root.missing_branch == 1
    assert tree.nodes_[root.children[1]].counts.tolist() == [0, 30]
    np.testing.assert_array_equal(tree.predict_proba(pd.DataFrame({"特征": [None]})), [[0, 1]])
    assert_partition(tree, data)


@pytest.mark.parametrize("destination", range(4))
def test_screenshot_merge_marker_and_matching_explicit_branch_agree(destination):
    data = pd.DataFrame({"评分": [0.1, 0.15, 0.2, 0.3, np.nan] * 10, "target": [0, 1, 0, 1, 1] * 10})
    cuts = [0.12, 0.18, 0.25]
    splits = cuts.copy()
    splits.insert(destination, np.nan)
    marked = MultiwayTreeExtractor(max_depth=1, missing_separate=True, max_n_bins=5).fit(data)
    explicit = clone(marked).fit(data)
    marked.manual_split(feature="评分", thresholds=splits)
    explicit.manual_split(feature="评分", thresholds=cuts + [np.nan], missing_branch=destination)
    assert marked.nodes_[0].missing_branch == explicit.nodes_[0].missing_branch == destination
    np.testing.assert_array_equal(marked.predict_proba(data), explicit.predict_proba(data))
    # 用原始行独立核算：指定普通区间的 10 行加上 10 行缺失。
    selected = data["评分"].eq([0.1, 0.15, 0.2, 0.3][destination]) | data["评分"].isna()
    child = marked.nodes_[0].children[destination]
    assert marked.nodes_[child].counts.sum() == selected.sum() == 20
    assert marked.nodes_[child].counts[1] == data.loc[selected, "target"].sum()
    assert_partition(marked, data)


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
        {"thresholds": [1, 2, 3, 4], "missing_separate": True},
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


def test_real_data_future_missing_predictions_rules_and_report():
    path = Path(__file__).resolve().parents[2] / "examples/hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    feature = "衡枢鉴真分老客版"
    data = pd.read_excel(path).dropna(subset=[feature])
    tree = MultiwayTreeExtractor(target="FPD", features=[feature], max_depth=1, n_jobs=1).fit(data)
    tree.manual_split(feature=feature, thresholds=[0.12, 0.18, 0.25])
    root = tree.nodes_[0]
    assert root.missing_branch == 0
    left_rows = data.loc[data[feature] <= 0.12]
    np.testing.assert_array_equal(
        tree.nodes_[root.children[0]].counts, [(left_rows.FPD == 0).sum(), (left_rows.FPD == 1).sum()]
    )
    production = data.iloc[:8].copy()
    production[feature] = np.nan
    assert tree.apply(production).tolist() == [root.children[0]] * len(production)
    np.testing.assert_allclose(tree.predict_proba(production)[:, 1], left_rows.FPD.mean())
    assert_partition(tree, production)
    report = tree.get_rule_table(production, leaf_only=True).set_index("节点编号")
    assert report.loc[root.children[0], "样本总数"] == len(production)
    assert report.loc[root.children[0], "坏样本数"] == production.FPD.sum()
    assert report.loc[root.children[1:], "样本总数"].sum() == 0
