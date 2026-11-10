"""规则表达式优化/美化的语义一致性测试.

核心不变式：任意复杂嵌套的规则在经过 :func:`beautify_expr` 美化与
:func:`optimize_expr` 化简后，对任意数据的求值结果必须与化简前 **完全一致**。

这里覆盖三类历史 bug：

1. 嵌套括号内的 ``&``/``|`` 优先级丢失（``(A | B) & C`` 被错误解析为 ``A | (B & C)``）；
2. 吸收律在内外层运算符相同时错误丢弃操作数（``A & (A & B)`` 被错误化简为 ``A``）；
3. 规范化比较时的有损处理（小写化 / 子串替换）导致误判等价（如列名含 ``and``/``or``）。
"""

import random

import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.core.rules.expr_optimizer import optimize_expr, beautify_expr


@pytest.fixture(scope="module")
def df():
    rng = np.random.RandomState(20240624)
    n = 4000
    return pd.DataFrame(
        {
            "age": rng.randint(10, 80, n),
            "income": rng.randint(0, 30000, n),
            "score": rng.rand(n),
            # 列名含 'and' / 'or' 子串，用于验证规范化比较不再做有损子串替换
            "brand": rng.randint(0, 5, n),
            "ord_x": rng.randint(0, 3, n),
        }
    )


def _assert_eval_equal(df, expr_a, expr_b, msg=""):
    """断言两个表达式对 df 的求值结果逐元素一致。"""
    a = np.asarray(df.eval(expr_a))
    b = np.asarray(df.eval(expr_b))
    assert (a == b).all(), f"{msg}\n  A: {expr_a!r}\n  B: {expr_b!r}"


# --------------------------------------------------------------------------- #
# bug 1: 嵌套括号内运算符优先级 / 括号丢失
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "combined",
    [
        "(age > 18 | income > 5000) & (score > 0.5)",  # (A|B)&C，不可塌缩为 A|(B&C)
        "(age > 18) & (income > 5000 | score > 0.5)",  # A&(B|C)
        "(age > 18 & income > 5000) | (score > 0.5)",  # (A&B)|C
        "((age > 18 | income > 5000) & score > 0.5) | (age > 18)",
        "~(age > 18 | income > 5000) & (score > 0.5)",
        "(age > 18 | income > 5000) | (score > 0.5 & income > 100)",
    ],
)
def test_nested_precedence_preserved(df, combined):
    beautified = beautify_expr(combined)
    optimized = optimize_expr(beautified)
    _assert_eval_equal(df, combined, beautified, "beautify 改变了语义")
    _assert_eval_equal(df, combined, optimized, "optimize 改变了语义")


def test_or_inside_parens_not_absorbed_into_outer_and(df):
    # 历史 bug：beautify 产出 'age > 18 | income > 5000 & score > 0.5'，
    # 等价于 A | (B & C)，与原意 (A | B) & C 不符
    combined = "(age > 18 | income > 5000) & (score > 0.5)"
    optimized = optimize_expr(beautify_expr(combined))
    # 修复后括号必须保留
    _assert_eval_equal(df, combined, optimized)


# --------------------------------------------------------------------------- #
# bug 2: 吸收律
# --------------------------------------------------------------------------- #


def test_valid_absorption_still_applies():
    # A | (A & B) = A,  A & (A | B) = A，以及交换形式
    assert optimize_expr("(a > 1) | ((a > 1) & (b > 1))") == "a > 1"
    assert optimize_expr("((a > 1) & (b > 1)) | (a > 1)") == "a > 1"
    assert optimize_expr("(a > 1) & ((a > 1) | (b > 1))") == "a > 1"
    assert optimize_expr("((a > 1) | (b > 1)) & (a > 1)") == "a > 1"


def test_idempotent_law():
    assert optimize_expr("(a > 1) & (a > 1)") == "a > 1"
    assert optimize_expr("(a > 1) | (a > 1)") == "a > 1"


def test_same_op_nesting_does_not_drop_operand(df):
    # 历史 bug：A & (A & B) 被错误化简为 A（丢弃 B）；A | (A | B) 同理
    for combined in [
        "(a > 1) & ((a > 1) & (b > 1))",
        "(a > 1) | ((a > 1) | (b > 1))",
        "((a > 1) | (b > 1)) | (a > 1)",
    ]:
        optimized = optimize_expr(combined)
        data = pd.DataFrame({"a": [0, 2, 2, 0], "b": [0, 2, 0, 2]})
        _assert_eval_equal(data, combined, optimized, "同运算符嵌套被错误吸收")


def test_double_negation():
    assert optimize_expr("~~(age > 18)") == "age > 18"
    assert optimize_expr("~~~(age > 18)") == "~(age > 18)"


# --------------------------------------------------------------------------- #
# bug 3: 列名含 and/or 子串不应影响等价判断
# --------------------------------------------------------------------------- #


def test_column_name_with_keyword_substring(df):
    combined = "(brand == 2 | ord_x == 1) & (score > 0.5)"
    optimized = optimize_expr(beautify_expr(combined))
    _assert_eval_equal(df, combined, optimized)


def test_xor_is_not_idempotently_collapsed(df):
    # ^ 不满足幂等/吸收律，A ^ A = False，不可化简为 A
    r = Rule("age > 18") ^ Rule("age > 18")
    pred = r.predict(df)
    assert not pred.any(), "A XOR A 必须恒为 False"


# --------------------------------------------------------------------------- #
# Rule 组合的端到端一致性
# --------------------------------------------------------------------------- #


def test_string_literal_with_operator_chars_preserved():
    # 字符串字面量内部的 & | ~ 不应被转换为 and/or/not
    data = pd.DataFrame({"cat": ["A&B", "C|D", "E"], "age": [20, 30, 40]})
    r = Rule("cat == 'A&B'") & Rule("age > 18")
    assert r.predict(data).tolist() == [True, False, False]


def test_backtick_columns_survive_combination():
    data = pd.DataFrame({"商品 类别": ["礼包", "手机", "礼包"], "age": [20, 16, 40]})
    r = Rule("`商品 类别` == '礼包'") & Rule("age > 18")
    truth = ((data["商品 类别"] == "礼包") & (data["age"] > 18)).tolist()
    assert r.predict(data).tolist() == truth


def test_random_nested_rules_are_consistent():
    """随机生成任意嵌套规则树，断言其求值结果与独立计算的真值完全一致。"""
    rng = np.random.RandomState(99)
    n = 2000
    data = pd.DataFrame(
        {
            "age": rng.randint(10, 80, n),
            "income": rng.randint(0, 30000, n),
            "score": rng.rand(n),
            "brand": rng.randint(0, 5, n),
        }
    )
    leaves = [
        (lambda: Rule("age > 30"), data["age"] > 30),
        (lambda: Rule("income < 5000"), data["income"] < 5000),
        (lambda: Rule("score >= 0.5"), data["score"] >= 0.5),
        (lambda: Rule("brand == 2"), data["brand"] == 2),
    ]

    def rand_tree(depth):
        if depth <= 0 or random.random() < 0.3:
            make, truth = random.choice(leaves)
            return make(), truth
        op = random.choice(["&", "|", "~", "^"])
        if op == "~":
            r, t = rand_tree(depth - 1)
            return ~r, ~t
        lr, lt = rand_tree(depth - 1)
        rr, rt = rand_tree(depth - 1)
        if op == "&":
            return lr & rr, lt & rt
        if op == "|":
            return lr | rr, lt | rt
        return lr ^ rr, lt ^ rt

    random.seed(2024)
    for _ in range(1500):
        rule, truth = rand_tree(random.randint(1, 5))
        pred = np.asarray(rule.predict(data))
        assert (pred == np.asarray(truth)).all(), f"不一致: {rule.expr!r}"


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("(x > -10) & (x > -2) & (x <= 3)", "x > -2 & x <= 3"),
        ("(x >= -2) & (x > -2) & (x <= 3) & (x < 3)", "x > -2 & x < 3"),
        ("(x <= 3) & (x >= -2) & (x < 3)", "x >= -2 & x < 3"),
        ("(-2 <= x <= 3) & (x > -1)", "x > -1 & x <= 3"),
        ("-2 < x <= 3", "x > -2 & x <= 3"),
        ("-2 < x", "x > -2"),
        ("(x > 3) & (x <= -2)", "x > 3 & x <= -2"),
        ("((x > -2) | x.isna()) & ((x <= 3) | x.isna())", "x > -2 & x <= 3 | x.isna()"),
        ("((x > -2) | x.isna()) & (x <= 3)", "x > -2 & x <= 3"),
        ("(x > -2) & ((x <= 3) | x.isna())", "x > -2 & x <= 3"),
        ("(x > -1e999) & (x <= 1e999)", "x > -1e309 & x <= 1e309"),
    ],
)
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_numeric_intersections_keep_boundaries_and_missing(expression, expected, dtype):
    values = [-np.inf, -10, -2, np.nextafter(-2.0, -np.inf), np.nextafter(-2.0, np.inf), -1, 0, 3, np.nextafter(3.0, -np.inf), np.nextafter(3.0, np.inf), np.inf, np.nan]
    data = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    optimized = optimize_expr(expression)
    assert optimized == expected
    assert optimize_expr(optimized) == optimized
    # 同时比较取反结果，防止把可空布尔值错误折叠成 False。
    for negate in (False, True):
        before = f"~({expression})" if negate else expression
        after = f"~({optimized})" if negate else optimized
        pd.testing.assert_series_equal(Rule(before).predict(data), Rule(after).predict(data))


def test_rule_combination_merges_interleaved_quoted_fields():
    data = pd.DataFrame({"信用 评分": [-2, 0, 3, 4, np.nan], "y": [1, 0, 2, 3, 4]})
    rule = Rule("`信用 评分` > -2") & Rule("y > 0") & Rule("`信用 评分` <= 3")
    assert rule.expr == "`信用 评分` > -2 & `信用 评分` <= 3 & y > 0"
    np.testing.assert_array_equal(rule.predict(data), data["信用 评分"].gt(-2) & data["信用 评分"].le(3) & data.y.gt(0))


@pytest.mark.parametrize("expression", ["(x > -2) | (x < 3)", "((x > -2) | y.isna()) & (x <= 3)", "((x + 1) * 2 > -2) & (x <= 3)", "(类别 in [\"a'b&c\", '双  空格']) & (x > -2)", "(类别 == '双  空格') | (类别 == '双 空格')"])
def test_unrelated_branches_arithmetic_and_literals_keep_semantics(expression):
    data = pd.DataFrame({"x": [-3, -2, -1, 0, 3, 4, np.nan], "y": [0, np.nan, 0, 0, np.nan, 0, 0], "类别": ["a'b&c", "双  空格", "双 空格", "其他", None, "其他", "其他"]})
    pd.testing.assert_series_equal(Rule(expression).predict(data), Rule(optimize_expr(expression)).predict(data))


def test_escaped_quote_keeps_literal_operators():
    expression = r"(类别 == 'a\'b&c') & (x > -2)"
    data = pd.DataFrame({"类别": ["a'b&c", "其他"], "x": [0, 0]})
    assert Rule(optimize_expr(expression)).predict(data).tolist() == [True, False]


@pytest.mark.parametrize(
    "expression, dtype, values",
    [
        ("x > 1.0 & x >= 1.0000000000000002", "float32", [1, 2, np.nan]),
        ("x < 1.0 & x <= 0.9999999999999999", "float32", [0, 1, np.nan]),
        ("(x > 1.0 | x.isna()) & (x >= 1.0000000000000002 | x.isna())", "float32", [1, 2, np.nan]),
        ("x > 9007199254740992 & x >= 9007199254740993", "float64", [2**53, 2**53 + 2, np.nan]),
    ],
)
def test_precision_sensitive_bounds_are_not_discarded(expression, dtype, values):
    data = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    optimized = optimize_expr(expression)
    pd.testing.assert_series_equal(Rule(expression).predict(data), Rule(optimized).predict(data))
