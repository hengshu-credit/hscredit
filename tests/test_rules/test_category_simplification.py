"""类别集合合并的独立掩码、缺失路由及树导出回归。"""

import itertools

import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule, optimize_expr
from hscredit.report.mining import MultiwayTreeExtractor, TreeRuleExtractor


@pytest.mark.parametrize(
    "left, right, op, expected",
    [
        ("in", "in", "&", "x in ['B']"),
        ("in", "in", "|", "x in ['A', 'B', 'C']"),
        ("in", "not in", "&", "x in ['A']"),
        ("in", "not in", "|", "x not in ['C']"),
        ("not in", "in", "&", "x in ['C']"),
        ("not in", "in", "|", "x not in ['A']"),
        ("not in", "not in", "&", "x not in ['A', 'B', 'C']"),
        ("not in", "not in", "|", "x not in ['B']"),
    ],
)
def test_category_intersection_union_and_difference(left, right, op, expected):
    first, second = Rule(f"x {left} ['A', 'B']"), Rule(f"x {right} ['B', 'C']")
    combined = first & second if op == "&" else first | second
    assert combined.expr == expected
    assert optimize_expr(combined.expr) == expected


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("left_excluded,right_excluded,op", list(itertools.product((False, True), (False, True), ("&", "|"))))
@pytest.mark.parametrize("left_missing,right_missing", list(itertools.product((False, True), repeat=2)))
def test_membership_merging_preserves_missing_unknown_and_negation(dtype, left_excluded, right_excluded, op, left_missing, right_missing):
    data = pd.DataFrame({"x": pd.Series(["A", "B", "C", "未知", "", None, np.nan, pd.NA], dtype=dtype)})
    data.index = ["重复"] * len(data)

    def condition(values, excluded, missing):
        expr = f"x {'not in' if excluded else 'in'} {values!r}"
        if excluded and not missing:
            expr += " & x.notna()"
        elif not excluded and missing:
            expr += " | x.isna()"
        mask = data.x.isin(values)
        if excluded:
            mask = ~mask
        return Rule(expr), mask.mask(data.x.isna(), missing)

    first, left_mask = condition(["A", "B"], left_excluded, left_missing)
    second, right_mask = condition(["B", "C"], right_excluded, right_missing)
    rule = first & second if op == "&" else first | second
    expected = left_mask & right_mask if op == "&" else left_mask | right_mask
    pd.testing.assert_series_equal(rule.predict(data), expected, check_names=False)
    pd.testing.assert_series_equal((~rule).predict(data), ~expected, check_names=False)
    assert rule.expr.count(" in ") <= 1
    assert optimize_expr(rule.expr) == rule.expr


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("x.isin(['A', 'B']) & ~x.isin(['B', 'C'])", "x in ['A']"),
        ("x == 'A' | x == 'B'", "x in ['A', 'B']"),
        ("x != 'A' & x != 'B'", "x not in ['A', 'B']"),
        ("x == 'A' & x in ['A', 'B']", "x in ['A']"),
        ("x in ['A', 'B', 'A'] & x in ('B', 'A')", "x in ['A', 'B']"),
        ("x in ['A'] & x in ['B']", "x in []"),
        ("x in ['A'] | x not in ['A']", "x not in []"),
        ("(x in ['A'] | x.isna()) & (x in ['B'] | x.isna())", "x.isna()"),
        ("(x in ['A'] | x not in ['A']) & x.notna()", "x.notna()"),
        ("x in ['A', 'B'] | (x not in ['A', 'B', 'C'] & x.notna())", "x not in ['C'] & x.notna()"),
    ],
)
def test_category_forms_empty_sets_and_complements(expression, expected):
    data = pd.DataFrame({"x": pd.Series(["A", "B", "C", "未知", None], dtype="string")})
    optimized = optimize_expr(expression)
    assert optimized == expected
    pd.testing.assert_series_equal(Rule(optimized).predict(data), Rule(expression).predict(data), check_names=False)
    pd.testing.assert_series_equal((~Rule(optimized)).predict(data), (~data.eval(expression)), check_names=False)


def test_quoted_fields_escaped_literals_and_interleaved_numeric_conditions():
    first = Rule("`来源 渠道` in [\"a'b&c\", '双  空格', '排除']")
    second = Rule("`来源 渠道`.isin([\"a'b&c\", '双  空格', '未知'])")
    rule = first & Rule("评分 > 600") & second & Rule("评分 <= 750")
    assert rule.expr == "`来源 渠道` in [\"a'b&c\", '双  空格'] & 评分 > 600 & 评分 <= 750"
    data = pd.DataFrame({"来源 渠道": ["a'b&c", "双  空格", "双 空格", "排除", None], "评分": [700, 700, 700, 700, 700]})
    assert rule.predict(data).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("dtype,values", [("Int64", [0, 1, 2, 3, pd.NA]), ("float32", [0, 1, 2, 3, np.nan]), ("boolean", [True, False, pd.NA])])
def test_numeric_and_boolean_category_codes(dtype, values):
    data = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    rule = Rule("x in [0, 1, 2]") & Rule("x not in [0, 2]")
    assert rule.expr == "x in [1]"
    np.testing.assert_array_equal(rule.predict(data), data.x.isin([1]))


def test_missing_literals_are_not_generalized_to_all_missing_values():
    data = pd.DataFrame({"x": pd.Series(["A", "B", None, np.nan, pd.NA], dtype="object")})
    expression = "x in [None, 'A'] & (x in ['B'] | x.isna())"
    optimized = optimize_expr(expression)
    assert "None" in optimized
    assert Rule(optimized).predict(data).tolist() == [False, False, True, False, False]


@pytest.mark.parametrize("expression", ["x in [9007199254740992] | x in [1.0]", "x in [9007199254740992, 1.0] & x not in [1.0]", "x in [9007199254740993] | x in [1]", "x in [True, 1.0] & x in [1]"])
def test_mixed_type_and_large_integer_memberships_keep_comparison_precision(expression):
    data = pd.DataFrame({"x": [2**53, 2**53 + 1, 1, 0]})
    optimized = optimize_expr(expression)
    pd.testing.assert_series_equal(Rule(expression).predict(data), Rule(optimized).predict(data))


def test_nested_boolean_combinations_match_independent_masks():
    data = pd.DataFrame({"x": ["A", "B", "C", "未知", None], "y": [1, 0, 1, 0, 1]})
    leaves = [(Rule("x in ['A', 'B']"), data.x.isin(["A", "B"])), (Rule("x not in ['B', 'C']"), ~data.x.isin(["B", "C"])), (Rule("x.isna()"), data.x.isna()), (Rule("y > 0"), data.y.gt(0))]
    rng = np.random.RandomState(42)
    for _ in range(80):
        rule, expected = leaves[rng.randint(len(leaves))]
        for _ in range(5):
            other, mask = leaves[rng.randint(len(leaves))]
            op = rng.randint(4)
            if op == 0:
                rule, expected = rule & other, expected & mask
            elif op == 1:
                rule, expected = rule | other, expected | mask
            elif op == 2:
                rule, expected = rule ^ other, expected ^ mask
            else:
                rule, expected = ~rule, ~expected
        np.testing.assert_array_equal(rule.predict(data), expected)


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_multiway_repeated_category_splits_keep_partition_and_reports(dtype):
    data = pd.DataFrame({"渠道": ["A", "B", "C", "D"] * 30, "target": [0, 1, 0, 1] * 30})
    tree = MultiwayTreeExtractor(max_depth=2, min_samples_leaf=1, n_jobs=1).fit(data)
    tree.manual_split(feature="渠道", category_groups=[["A", "B", "C"], ["D"]], missing_branch=0)
    middle = tree.nodes_[0].children[0]
    tree.manual_split(feature="渠道", category_groups=[["A", "B"], ["C"]], node=middle, missing_branch=0)
    raw = pd.DataFrame({"渠道": pd.Series(["A", "B", "C", "D", "未知", None], dtype=dtype)})
    rules = tree.get_rules()
    for node, rule in zip(tree.get_leaf_node_ids(), rules):
        np.testing.assert_array_equal(rule.predict(raw), tree.apply(raw) == node)
        assert rule.expr.count(" in ") == 1
    assert any(rule.expr == "渠道 not in ['D', 'C']" for rule in rules)
    table = tree.get_rule_table(leaf_only=True, apply_rule_filter=False)
    assert set(table["组合策略"]) == {rule.expr for rule in rules}
    assert table["样本总数"].sum() == len(data)


def test_automatic_tree_category_rules_match_encoded_paths_and_artifacts():
    data = pd.DataFrame({"渠道": ["A", "B", "C", "D", "E", None] * 30, "target": [0, 1, 0, 1, 0, 1] * 30})
    miner = TreeRuleExtractor(algorithm="dt", max_depth=5, min_samples_leaf=1, min_samples_split=2, n_jobs=1, random_state=42).fit(data)
    rules = miner.get_rules(min_samples=1, datasets=data)
    assert any(len(rule.metadata_["conditions"]) > 1 for rule in rules)
    raw = data.loc[miner.X_train_.index]
    for rule in rules:
        expected = miner._apply_conditions(rule.metadata_["conditions"], miner.X_train_)
        np.testing.assert_array_equal(rule.predict(raw), expected)
        np.testing.assert_array_equal(rule.artifact_.predict(raw), expected)
        assert rule.expr.count(" in ") <= 1
        assert rule.metadata_["命中样本数"] == rule.predict(data).sum()
