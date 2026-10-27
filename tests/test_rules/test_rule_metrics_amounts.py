"""规则评估金额输入的独立业务预期，避免件数与金额混算。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.report.mining import RuleMetrics


@pytest.mark.parametrize("as_series", [False, True])
def test_rule_metrics_reports_amounts_without_replacing_sample_counts(as_series):
    data = pd.DataFrame({"x": [0, 0, 1, 1]}, index=[10, 20, 30, 40])
    target = pd.Series([0, 1, 0, 1], index=data.index)
    train_amount = np.array([1.0, 9.0, 9.0, 1.0])
    test_amount = np.array([2.0, 8.0, 2.0, 8.0])
    if as_series:
        train_amount = pd.Series(train_amount, index=data.index)
        test_amount = pd.Series(test_amount, index=data.index)
    calculator = RuleMetrics(n_jobs=1)
    rule = Rule("x >= 1")
    baseline = calculator.evaluate_rule(rule, data, target, data, target)
    result = calculator.evaluate_rule(
        rule, data, target, data, target, amount_train=train_amount, amount_test=test_amount
    )
    for name, value in baseline.items():
        assert result[name] == pytest.approx(value), name
    for prefix, loss, lift in [("训练", 0.1, 0.2), ("测试", 0.8, 1.0)]:
        assert result[f"{prefix}_总金额"] == pytest.approx(20)
        assert result[f"{prefix}_命中金额"] == pytest.approx(10)
        assert result[f"{prefix}_命中金额占比"] == pytest.approx(0.5)
        assert result[f"{prefix}_命中损失率"] == pytest.approx(loss)
        assert result[f"{prefix}_命中金额LIFT值"] == pytest.approx(lift)


def test_rule_metrics_batch_keeps_per_dataset_amounts():
    data = pd.DataFrame({"x": [0, 0, 1, 1]})
    target = pd.Series([0, 1, 0, 1])
    amounts = pd.Series([1.0, 9.0, 9.0, 1.0])
    result = RuleMetrics(n_jobs=2, parallel_backend="threading").evaluate_rules(
        [Rule("x >= 1"), Rule("x < 1")], data, target, amount_train=amounts
    )
    np.testing.assert_allclose(result["训练_命中损失率"], [0.1, 0.9])
    np.testing.assert_allclose(result["训练_命中样本数"], [2, 2])


def test_rule_metrics_respects_explicit_positive_label():
    data = pd.DataFrame({"x": [0, 0, 1, 1]})
    target = pd.Series([0, 0, 1, 1])
    result = RuleMetrics(target_positive=0, n_jobs=1).evaluate_rule(
        Rule("x == 0"), data, target, amount_train=pd.Series([1, 2, 3, 4])
    )
    for name in ["召回率", "精确率", "准确率", "命中坏样本率", "命中损失率"]:
        assert result[f"训练_{name}"] == pytest.approx(1)
    assert result["训练_拦截后坏样本率"] == pytest.approx(0)


def test_rule_metrics_single_class_and_no_hits_has_four_confusion_counts():
    result = RuleMetrics(n_jobs=1).evaluate_rule(Rule("x > 10"), pd.DataFrame({"x": [1, 2]}), pd.Series([0, 0]))
    assert result["训练_真负例"] == 2
    assert result["训练_真正例"] == 0
    assert result["训练_假正例"] == 0
    assert result["训练_假负例"] == 0


def test_rule_metrics_train_test_bad_rate_difference_is_not_lift_difference():
    data = pd.DataFrame({"x": [0, 1, 2, 3]})
    result = RuleMetrics(n_jobs=1).evaluate_rule(
        Rule("x > 1"), data, pd.Series([0, 0, 0, 1]), data, pd.Series([1, 0, 1, 1])
    )
    assert result["训练_命中坏样本率"] == 0.5
    assert result["测试_命中坏样本率"] == 1
    assert result["badrate_diff"] == -0.5


def test_rule_metrics_bad_rate_difference_is_undefined_for_empty_hit_population():
    train = pd.DataFrame({"x": [0, 1, 2, 3]})
    test = pd.DataFrame({"x": [4, 5, 6, 7]})
    result = RuleMetrics(n_jobs=1).evaluate_rule(
        Rule("x > 3"), train, pd.Series([0, 1, 0, 1]), test, pd.Series([0, 1, 0, 1])
    )
    assert pd.isna(result["badrate_diff"])
