"""逐项复核多特征合计、精度与规则字段隔离。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.report import feature_bin_stats
from hscredit.report.mining import RuleMetrics


class _ColumnRule(Rule):
    def predict(self, data):
        return data["__target__"].eq(1).to_numpy()


def test_rule_metrics_does_not_overwrite_feature_used_by_rule():
    data = pd.DataFrame({"__target__": [1, 1, 0, 0]})
    result = RuleMetrics(n_jobs=1).evaluate_rule(_ColumnRule("__target__ == 1"), data, pd.Series([0, 1, 0, 1]))
    assert result["训练_命中坏样本数"] == 1
    assert result["训练_命中坏样本率"] == result["训练_精确率"] == 0.5
    assert data["__target__"].tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("amount_name", ["target", "x == 1", "金额"])
def test_rule_report_amount_names_do_not_collide_with_internal_columns(amount_name):
    data = pd.DataFrame({"x": [0, 0, 1, 1], "标签": [0, 1, 0, 1], amount_name: [1.0, 9.0, 9.0, 1.0]})
    table = Rule("x == 1").report(data, target="标签", amount=amount_name, margins=True).set_index("分箱")
    assert table.loc["命中", "样本总数"] == 10
    assert table.loc["命中", "坏样本率"] == 0.1
    assert table.loc["合计", "样本总数"] == 20


@pytest.mark.parametrize("invalid", [[1, -1, 2, 3], [1, np.inf, 2, 3], [1, np.nan, 2, 3]])
def test_rule_report_rejects_invalid_amounts(invalid):
    data = pd.DataFrame({"x": [0, 0, 1, 1], "标签": [0, 1, 0, 1], "金额": invalid})
    with pytest.raises(ValueError, match="金额|amount"):
        Rule("x == 1").report(data, target="标签", amount="金额")


def test_rule_amount_validation_uses_each_targets_effective_population():
    data = pd.DataFrame({"x": [0, 0, 1, 1], "标签": [0, -1, 1, 0], "金额": [1.0, np.nan, 3.0, 4.0]})
    table = Rule("x == 1").report(data, target="标签", amount="金额", del_grey=True, margins=True)
    assert table.loc[table["分箱"] == "合计", "样本总数"].iloc[0] == 8
    assert pd.isna(data.loc[1, "金额"])


@pytest.mark.parametrize("expression", ["x >= 0", "x < 0"])
def test_rule_rejection_keeps_numeric_compatibility_and_full_rejection_nan(expression):
    data = pd.DataFrame({"x": [0, 1, 2, 3], "标签": [0, 1, 0, 1]})
    table = Rule(expression).report(data, target="标签").set_index("分箱")
    if expression == "x >= 0":
        assert pd.isna(table.loc["命中", "风险拒绝比"])
        assert table.loc["命中", "坏账改善"] == 1
        result = RuleMetrics(n_jobs=1).evaluate_rule(Rule(expression), data[["x"]], data["标签"])
        assert pd.isna(result["训练_拦截后坏样本率"])
        assert result["训练_坏账改善"] == 1
    else:
        assert table.loc["命中", "坏账改善"] == 0
        assert table.loc["命中", "风险拒绝比"] == 0


@pytest.mark.parametrize("long_format", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
def test_feature_margins_preserve_population_per_feature_and_target(long_format, multi_target):
    data = pd.DataFrame({"x": np.arange(8), "z": np.arange(8)[::-1], "y": [0, 0, 0, 1, 0, 1, 1, 1]})
    target = {"甲": data.y, "乙": 1 - data.y} if multi_target else "y"
    table = feature_bin_stats(
        data,
        features=["x", "z"],
        target=target,
        method="uniform",
        max_n_bins=2,
        margins=True,
        long_format=long_format,
        n_jobs=2,
        parallel_backend="threading",
    )
    details = table["分箱详情"] if isinstance(table.columns, pd.MultiIndex) else table
    totals = details.loc[details["分箱标签"] == "合计"]
    expected = 4 if multi_target and long_format else 2
    assert len(totals) == expected
    assert totals["样本总数"].tolist() == [8] * expected


def test_feature_margins_compute_rates_before_selecting_return_columns():
    data = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 0, 1, 1]})
    table = feature_bin_stats(
        data, "x", target="y", method="uniform", max_n_bins=2, margins=True, return_cols=["坏样本率"], n_jobs=1
    )
    assert table.iloc[-1]["坏样本率"] == 0.5


@pytest.mark.parametrize("round_digits", [False, True])
def test_feature_amount_margins_aggregate_before_rounding(round_digits):
    data = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "金额": [0.004, 0.004, 0.004, 0.004]})
    table = feature_bin_stats(
        data,
        "x",
        target="y",
        method="uniform",
        max_n_bins=2,
        amount="金额",
        margins=True,
        round_digits=round_digits,
        n_jobs=1,
    )
    assert table.iloc[-1]["样本总数"] == pytest.approx(0.02 if round_digits else 0.016)
    assert table.iloc[-1]["坏样本率"] == 0.5


def test_multilabel_rule_workers_receive_only_required_standard_rule_columns(monkeypatch):
    import hscredit.core.rules.rule as rule_module

    data = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 0, 1], "金额": [1, 2, 3, 4], "无关字段": [9] * 4})
    original = data.copy()
    worker = rule_module._rule_report_target_worker
    observed = []

    def record(task):
        observed.append(list(task[1].columns))
        return worker(task)

    monkeypatch.setattr(rule_module, "_rule_report_target_worker", record)
    Rule("x > 1", n_jobs=2, parallel_backend="threading").report(
        data, target={"甲": data.y, "乙": 1 - data.y}, amount="金额"
    )
    assert len(observed) == 2
    assert all("无关字段" not in columns and "x" in columns and "金额" in columns for columns in observed)
    pd.testing.assert_frame_equal(data, original)


def test_custom_rule_multilabel_report_keeps_complete_input():
    data = pd.DataFrame({"x": [0, 1, 2, 3], "__target__": [1, 1, 0, 0], "隐藏输入": [1, 0, 1, 0]})

    class CustomRule(Rule):
        def predict(self, frame):
            return frame["隐藏输入"].eq(1)

    table = CustomRule("x > 1", n_jobs=2, parallel_backend="threading").report(
        data, target={"甲": [1, 0, 1, 0], "乙": [0, 1, 0, 1]}
    )
    assert table.loc[0, ("甲", "坏样本率")] == 1
    assert table.loc[0, ("乙", "坏样本率")] == 0
