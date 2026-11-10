"""树路径共用 Rule 区间简化，导出、预测与报告保持一致。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning.spec import BinSpec
from hscredit.core.rules import Rule
from hscredit.report.mining import DecisionTreeAnalyzer, ManualTreeExtractor, MultiwayTreeExtractor, TreeRuleExtractor


@pytest.mark.parametrize("extractor", [DecisionTreeAnalyzer, ManualTreeExtractor])
@pytest.mark.parametrize("missing", [False, True])
def test_binary_path_exports_compact_interval(extractor, missing):
    path = [["信用 评分", ">", -10, missing], ["信用 评分", ">", -2, missing], ["信用 评分", "<=", 3, missing]]
    rule = extractor()._format_rule(path)
    assert isinstance(rule, Rule)
    assert rule.expr == "`信用 评分` > -2.0 & `信用 评分` <= 3.0" + (" | `信用 评分`.isna()" if missing else "")
    raw = pd.DataFrame({"信用 评分": pd.Series([-10, -2, 0, 3, 4, pd.NA], dtype="Float64")})
    expected = raw["信用 评分"].gt(-2) & raw["信用 评分"].le(3)
    expected = expected.fillna(missing)
    np.testing.assert_array_equal(rule.predict(raw), expected)


@pytest.mark.parametrize("extractor", [DecisionTreeAnalyzer, ManualTreeExtractor])
def test_binary_fitted_rules_keep_node_hits_and_report(extractor):
    data = pd.DataFrame({"评分": np.tile(np.arange(-10.0, 11.0), 6)})
    data["target"] = data["评分"].between(-2, 3).astype(int)
    tree = extractor(max_depth=3, min_samples_leaf=1, min_samples_split=2, missing=-1, n_jobs=1).fit(data)
    raw = pd.DataFrame({"评分": [-np.inf, -10, -2.5, np.nextafter(-2.5, np.inf), 0, 3.5, np.nextafter(3.5, np.inf), 10, np.inf, np.nan]})
    rules = tree.get_rules()
    assert any("评分 > -2.5 & 评分 <= 3.5" in rule.expr for rule in rules)
    for (_, row), rule in zip(tree._df_rules.loc[tree._df_rules.if_leaf].iterrows(), rules):
        expected = pd.Series(True, index=raw.index)
        for feature, operator, threshold, include_missing in row.rule_list:
            values = raw[feature]
            hit = values.le(threshold) if operator == "<=" else values.gt(threshold)
            expected &= hit | (values.isna() & include_missing)
        np.testing.assert_array_equal(rule.predict(raw), expected)
    table = tree.get_rule_table(leaf_only=True, apply_rule_filter=False)
    assert set(table["组合策略"]) == {rule.expr for rule in rules}
    assert table["样本总数"].sum() == len(data)


def test_multiway_repeated_splits_merge_bounds_and_missing_routes():
    data = pd.DataFrame({"信用 评分": np.arange(120.0), "target": np.arange(120) % 2})
    tree = MultiwayTreeExtractor(max_depth=2, min_samples_leaf=1, n_jobs=1).fit(data)
    tree.manual_split(feature="信用 评分", thresholds=[30, 90], missing_branch=1)
    middle = tree.nodes_[0].children[1]
    tree.manual_split(feature="信用 评分", thresholds=[40, 80], node=middle, missing_branch=1)
    raw = pd.DataFrame({"信用 评分": [30, 40, np.nextafter(40.0, np.inf), 80, np.nextafter(80.0, np.inf), 90, -np.inf, np.inf, np.nan]})
    for node, rule in zip(tree.get_leaf_node_ids(), tree.get_rules()):
        np.testing.assert_array_equal(rule.predict(raw), tree.apply(raw) == node)
        assert rule.expr.count("`信用 评分`") <= 3  # 上下界各引用一次字段，再加可选缺失分支。
    assert tree._format_rule(tree.nodes_[tree.nodes_[middle].children[1]].path).expr == "`信用 评分` > 40.0 & `信用 评分` <= 80.0 | `信用 评分`.isna()"
    table = tree.get_rule_table(leaf_only=True, apply_rule_filter=False)
    assert set(table["组合策略"]) == {rule.expr for rule in tree.get_rules()}
    assert table["样本总数"].sum() == len(data)


def test_multiway_category_and_interval_path_preserves_unknown_and_missing():
    path = [BinSpec("类型", 0, kind="categories", values=("a'b&c", "双  空格"), include_missing=True, unknown=True, known_values=("a'b&c", "双  空格", "其他")), BinSpec("x", 0, lower=-2, upper=3, closed_left=False, closed_right=True, include_missing=True)]
    raw = pd.DataFrame({"类型": ["a'b&c", "双  空格", "其他", "未知", None], "x": [-1, 3, 0, 0, np.nan]})
    expected = path[0].mask(raw) & path[1].mask(raw)
    rule = MultiwayTreeExtractor._format_rule(path)
    np.testing.assert_array_equal(rule.predict(raw), expected)
    assert "x > -2 & x <= 3" in rule.expr


def test_automatic_tree_rules_preserve_float32_boundaries_and_artifacts():
    data = pd.DataFrame({"信用 评分": np.tile(np.arange(-10.0, 11.0), 10)})
    data["target"] = data["信用 评分"].between(-2, 3).astype(int)
    miner = TreeRuleExtractor(algorithm="dt", max_depth=3, min_samples_leaf=1, min_samples_split=2, n_jobs=1, random_state=4).fit(data)
    cuts = miner.model_.get_native_model().tree_.threshold
    values = [value for cut in cuts for value in (cut, np.nextafter(cut, -np.inf), np.nextafter(cut, np.inf))]
    raw = pd.DataFrame({"信用 评分": values + [np.nan, -100, 100]})
    transformed = raw.fillna(0).astype(np.float32)
    rules = miner.get_rules(min_samples=1, datasets=data)
    assert any("`信用 评分` > " in rule.expr and " & `信用 评分` <= " in rule.expr for rule in rules)
    for rule in rules:
        expected = miner._apply_conditions(rule.metadata_["conditions"], transformed)
        np.testing.assert_array_equal(rule.predict(raw), expected)
        np.testing.assert_array_equal(rule.artifact_.predict(raw), expected)
        assert rule.metadata_["命中样本数"] == rule.predict(data).sum()
        assert rule.expr.count("`信用 评分`") <= 3
    table = miner.get_rules_dataframe(min_samples=1, datasets=data)
    assert set(table["规则表达式"]) == {rule.expr for rule in rules}
