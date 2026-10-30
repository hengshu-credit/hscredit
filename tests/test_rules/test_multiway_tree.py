"""多叉树独立公式、原始行、编辑事务与模型交付契约。"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier

from hscredit import MultiwayTreeExtractor
from hscredit.core.models import RuleSet
from hscredit.exceptions import InputValidationError, NotFittedError
from hscredit.report.mining.multiway_tree import _chi2_log_pvalue, _split_statistics


def sample():
    return pd.DataFrame({"评分": np.arange(120, dtype=float), "target": [0] * 40 + [1] * 40 + [0] * 40})


def assert_rule_partition(tree, data):
    leaves = tree.apply(data)
    masks = []
    for node, rule in zip(tree.get_leaf_node_ids(), tree.get_rules()):
        mask = rule.predict(data).to_numpy(dtype=bool)
        np.testing.assert_array_equal(mask, leaves == node)
        masks.append(mask)
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(data)))


def test_split_statistic_matches_hand_calculation():
    # 三组各 20 人，坏人数分别为 0、20、0；期望坏数每组 20/3。
    counts = [[20, 0], [0, 20], [20, 0]]
    statistic, pvalue, gain = _split_statistics(counts, "gini")
    assert statistic == pytest.approx(60)
    assert pvalue == pytest.approx(np.exp(-30))  # 自由度 2 的尾概率可直接手算。
    assert gain == pytest.approx(4 / 9)
    assert _chi2_log_pvalue(10000, 2) == pytest.approx(-5000)
    assert _split_statistics(counts, "entropy")[2] == pytest.approx(-2 / 3 * np.log2(2 / 3) - 1 / 3 * np.log2(1 / 3))


@pytest.mark.parametrize("criterion", ["chi2", "gini", "entropy"])
def test_auto_grows_true_multiway_node_and_exact_probability(criterion):
    data = sample()
    tree = MultiwayTreeExtractor(max_depth=1, criterion=criterion).fit(data)
    assert len(tree.nodes_[0].children) == 3
    np.testing.assert_array_equal(tree.predict(data), data.target)
    np.testing.assert_allclose(tree.predict_proba(data)[:, 1], data.target)
    assert tree.evaluate().iloc[0]["AUC"] == 1
    assert_rule_partition(tree, data)
    assert is_classifier(tree)
    assert clone(tree).get_params() == tree.get_params()


def test_bonferroni_adjusts_actual_numeric_search_count(monkeypatch):
    tree = MultiwayTreeExtractor(max_depth=0, max_n_bins=3).fit(sample())
    monkeypatch.setattr(tree, "_numeric_pool", lambda values: (39.5, 79.5))
    row = tree.get_split_candidates().iloc[0]
    assert row["调整后P值"] == pytest.approx(row["P值"] * 3)
    assert row["分裂价值"] == pytest.approx(-np.log10(row["调整后P值"]))


def test_binary_and_multiway_branch_limits_and_greedy_search():
    data = sample()
    binary = MultiwayTreeExtractor(max_depth=1, max_n_bins=2).fit(data)
    assert all(len(node.children) <= 2 for node in binary.nodes_.values())
    greedy = MultiwayTreeExtractor(max_depth=1, max_candidates=1).fit(data)
    assert len(greedy.nodes_[0].children) >= 3
    assert_rule_partition(greedy, data)


def test_large_sample_logworth_still_distinguishes_underflowed_pvalues():
    data = pd.concat([sample()] * 100, ignore_index=True)
    tree = MultiwayTreeExtractor(max_depth=1).fit(data)
    assert len(tree.nodes_[0].children) == 3
    np.testing.assert_array_equal(tree.predict(data), data.target)


def test_feature_map_is_preserved_in_reports_and_manual_rules():
    tree = MultiwayTreeExtractor(feature_map={"评分": "业务评分"}).fit(sample())
    assert "业务评分" in tree.get_rule_table()["字段含义"].to_string()


def test_manual_boundary_missing_nullable_and_duplicate_index():
    data = sample()
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(feature="评分", thresholds=[40, 80], missing_branch=1)
    raw = pd.DataFrame(
        {"评分": [40, np.nextafter(40.0, -np.inf), np.nextafter(40.0, np.inf), 80, np.nan, -np.inf, np.inf]}
    )
    raw.index = ["重复"] * len(raw)
    children = tree.nodes_[0].children
    assert tree.apply(raw).tolist() == [children[i] for i in [0, 0, 1, 1, 1, 0, 2]]
    assert_rule_partition(tree, raw)
    assert_rule_partition(tree, pd.DataFrame({"评分": pd.Series([40, 80, pd.NA], dtype="Int64")}))


def test_native_missing_is_learned_and_report_counts_match_original_rows():
    data = pd.DataFrame({"评分": [0.0] * 20 + [1.0] * 20 + [np.nan] * 10, "target": [1] * 20 + [0] * 20 + [1] * 10})
    tree = MultiwayTreeExtractor(max_depth=1).fit(data)
    assert tree.nodes_[0].missing_branch == 0
    np.testing.assert_array_equal(tree.predict(data), data.target)
    table = tree.get_rule_table(leaf_only=True)
    assert table["样本总数"].tolist() == [30, 20]
    assert table["坏样本数"].tolist() == [30, 0]
    assert_rule_partition(tree, data)


def test_local_edit_keeps_other_node_ids_and_prune_probability():
    data = sample()
    tree = MultiwayTreeExtractor(max_depth=3).fit(data)
    tree.manual_split(feature="评分", thresholds=[30, 90])
    left, middle, right = tree.nodes_[0].children
    untouched = copy.deepcopy(tree.nodes_[left])
    tree.manual_split(feature="评分", thresholds=[40, 80], node=middle)
    assert left in tree.nodes_ and right in tree.nodes_
    np.testing.assert_array_equal(tree.nodes_[left].counts, untouched.counts)
    old_children = list(tree.nodes_[middle].children)
    tree.set_thresholds(middle, [45, 75])
    assert not any(child in tree.nodes_ for child in old_children)
    assert_rule_partition(tree, data)
    tree.prune(middle)
    mask = (data["评分"] > 30) & (data["评分"] <= 90)
    np.testing.assert_allclose(tree.predict_proba(data.loc[mask])[:, 1], data.loc[mask, "target"].mean())
    tree.grow(middle)
    assert tree.nodes_[middle].children
    tree.prune(0)
    assert tree.get_leaf_node_ids() == [0]
    assert tree.get_rules()[0].expr == "True"
    assert_rule_partition(tree, data)


@pytest.mark.parametrize("cuts", [[20, 10], [10, 10], [], [np.nan], [np.inf], [1, 2, 3, 4]])
def test_invalid_edit_preserves_predictions_and_structure(cuts):
    tree = MultiwayTreeExtractor().fit(sample())
    table = tree.get_node_table()
    prediction = tree.predict_proba(sample())
    with pytest.raises(InputValidationError):
        tree.manual_split(feature="评分", thresholds=cuts)
    pd.testing.assert_frame_equal(tree.get_node_table(), table)
    np.testing.assert_array_equal(tree.predict_proba(sample()), prediction)


def test_empty_manual_leaf_inherits_parent_but_counts_stay_zero():
    tree = MultiwayTreeExtractor().fit(sample()).manual_split(feature="评分", thresholds=[-100, 40, 80])
    empty = tree.nodes_[0].children[0]
    assert tree.nodes_[empty].counts.sum() == 0
    np.testing.assert_allclose(tree.predict_proba(pd.DataFrame({"评分": [-101]}))[0], [2 / 3, 1 / 3])
    assert_rule_partition(tree, pd.DataFrame({"评分": [-101, np.nan, 40, 90]}))


def test_categorical_groups_missing_unknown_and_quoted_categories():
    categories = ["低,风险", '含"引号', "高风险", "其他"]
    data = pd.DataFrame({"类型": categories * 30, "target": [0, 0, 1, 1] * 30})
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    tree.manual_split(
        feature="类型", category_groups=[categories[:2], [categories[2]], [categories[3]]], missing_branch=2
    )
    raw = pd.DataFrame({"类型": pd.Series(categories + [None, "新类别"], dtype="string")})
    assert_rule_partition(tree, raw)
    assert tree.apply(raw)[-2:].tolist() == [tree.nodes_[0].children[2]] * 2
    with pytest.raises(InputValidationError, match="完整覆盖"):
        tree.manual_split(feature="类型", category_groups=[[categories[0]], [categories[1]]])
    with pytest.raises(InputValidationError, match="重复"):
        tree.manual_split(feature="类型", category_groups=[[categories[0]], categories])


def test_multiclass_models_reports_and_serialization(tmp_path):
    data = pd.DataFrame({"类型": ["A", "B", "C"] * 40, "target": ["正常", "逾期", "结清"] * 40})
    tree = MultiwayTreeExtractor(max_depth=2).fit(data)
    assert len(tree.nodes_[0].children) == 3
    assert tree.evaluate().iloc[0]["准确率"] == 1
    assert isinstance(tree.get_rule_table(long_format=False).columns, pd.MultiIndex)
    assert "目标标签" in tree.get_rule_table().columns
    assert_rule_partition(tree, data)
    path = tmp_path / "多叉树.pkl"
    tree.save(path)
    restored = MultiwayTreeExtractor.load(path)
    np.testing.assert_array_equal(restored.classes_, tree.classes_)
    np.testing.assert_array_equal(restored.predict_proba(data), tree.predict_proba(data))
    ruleset = restored.to_ruleset([restored.nodes_[0].children[0]])
    assert isinstance(ruleset, RuleSet)
    np.testing.assert_array_equal(
        ruleset.evaluate(data, return_details=False)[0], restored.apply(data) == restored.nodes_[0].children[0]
    )


def test_dual_api_external_y_and_refit_failure():
    data = sample()
    external = 1 - data.target.to_numpy()
    tree = MultiwayTreeExtractor(features=["评分"]).fit(data, external)
    np.testing.assert_array_equal(tree.predict(data), external)
    original = tree.predict(data)
    with pytest.raises(InputValidationError, match="目标不能缺失"):
        tree.fit(data, np.full(len(data), np.nan))
    np.testing.assert_array_equal(tree.predict(data), original)
    numpy_tree = clone(tree).fit(data[["评分"]].to_numpy(), external)
    np.testing.assert_array_equal(numpy_tree.predict(data[["评分"]].to_numpy()), external)


def test_single_class_all_missing_and_not_fitted():
    data = pd.DataFrame({"评分": [np.nan] * 20, "target": [1] * 20})
    tree = MultiwayTreeExtractor().fit(data)
    assert tree.get_leaf_node_ids() == [0]
    np.testing.assert_array_equal(tree.predict(data), data.target)
    assert tree.evaluate().iloc[0]["准确率"] == 1
    with pytest.raises(NotFittedError):
        MultiwayTreeExtractor().predict(data)


@pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
def test_report_overdue_operator_reaches_each_dataset_and_matches_raw_counts(operator):
    import operator as comparison

    data = sample().assign(MOB1=np.tile([0, 3, 7, 8], 30))
    tree = MultiwayTreeExtractor(max_depth=1, n_jobs=1).fit(data)
    compare = {">": comparison.gt, ">=": comparison.ge, "<": comparison.lt, "<=": comparison.le}[operator]
    reports = tree.report({"完整": data, "子集": data.iloc[:11]}, overdue="MOB1", dpds=7, overdue_operator=operator)
    for name, table in reports.items():
        rows = data if name == "完整" else data.iloc[:11]
        bad_column = next(c for c in table if (c[-1] if isinstance(c, tuple) else c) == "坏样本数")
        assert table.iloc[0][bad_column] == compare(rows.MOB1, 7).sum()


def test_real_loan_counts_amount_overdue_and_model_effects():
    path = Path(__file__).resolve().parents[2] / "examples/hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    data = pd.read_excel(path)
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别"]
    tree = MultiwayTreeExtractor(target="FPD", features=features, max_depth=2, criterion="gini").fit(data)
    assert_rule_partition(tree, data)
    table = tree.get_rule_table(leaf_only=True)
    assert table["样本总数"].sum() == len(data) == 970
    assert table["坏样本数"].sum() == data.FPD.sum()
    for node, row in zip(tree.get_leaf_node_ids(), table.itertuples(index=False)):
        mask = tree.apply(data) == node
        assert row.样本总数 == mask.sum()
        assert row.坏样本数 == data.loc[mask, "FPD"].sum()
    report = tree.report(data, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额", leaf_only=True, long_format=False)
    assert isinstance(report.columns, pd.MultiIndex)
    pd.testing.assert_frame_equal(
        report,
        tree.get_rule_table(overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额", leaf_only=True, long_format=False),
    )
    assert set(tree.evaluate({"训练集": data, "验证集": data.iloc[:100]})["数据集"]) == {"训练集", "验证集"}


@pytest.mark.parametrize("comparison, method", [(">", "gt"), (">=", "ge"), ("<", "lt"), ("<=", "le")])
def test_overdue_operator_is_forwarded_to_each_dataset(comparison, method):
    data = pd.DataFrame({"评分": range(6), "target": [0, 1] * 3, "MOB1": [0, 2, 3, 7, 8, 0]})
    tree = MultiwayTreeExtractor(features=["评分"], max_depth=0).fit(data)
    reports = tree.report({"甲": data, "乙": data}, overdue=["MOB1"], dpds=[3], overdue_operator=comparison)
    expected = int(getattr(data.MOB1, method)(3).sum())
    for report in reports.values():
        bad_column = next(c for c in report if (c[-1] if isinstance(c, tuple) else c) == "坏样本数")
        assert report.iloc[0][bad_column] == expected
