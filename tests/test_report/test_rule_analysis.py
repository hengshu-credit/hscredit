import pandas as pd
from openpyxl import load_workbook
from inspect import signature

from hscredit.core.rules import Rule
import hscredit.report.rule_analysis as rule_analysis_module
from hscredit.report.rule_analysis import multi_label_rule_analysis, ruleset_analysis


def test_ruleset_analysis_returns_expected_rows():
    data = pd.DataFrame(
        {
            "age": [18, 22, 30, 17],
            "target": [0, 1, 0, 1],
        }
    )

    result = ruleset_analysis(data, [Rule("age >= 21")], target="target")

    assert list(result["分箱"]) == ["原始样本", "age >= 21", "剩余样本", "所有规则"]


def test_ruleset_analysis_handles_multiindex_rule_report_columns():
    data = pd.DataFrame(
        {
            "age": [18, 22, 30, 17],
            "MOB1": [0, 4, 1, 7],
        }
    )

    result = ruleset_analysis(data, [Rule("age >= 21")], overdue="MOB1", dpds=[3, 1])

    assert isinstance(result.columns, pd.MultiIndex)
    assert ("分箱详情", "分箱") in result.columns
    assert ("MOB1 3+", "坏样本率") in result.columns
    assert ("MOB1 1+", "LIFT值") in result.columns
    assert list(result[("分箱详情", "分箱")]) == ["原始样本", "age >= 21", "剩余样本", "所有规则"]


def test_ruleset_analysis_exposes_del_grey_parameter():
    """规则集分析应把灰样本控制作为可发现的正式参数。"""
    assert "del_grey" in signature(ruleset_analysis).parameters


def test_prior_rule_long_report_preserves_each_targets_population():
    data = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "days": [0, 2, 4, 0, 1, None], "amount": [1, 2, 4, 8, 16, 32]})
    rule, prior = Rule("x >= 2"), Rule("x == 0")
    long = rule.report(data, overdue="days", dpds=[0, 3], del_grey=True, prior_rules=prior, amount="amount", margins=True, long_format=True, filter_cols=["样本总数", "坏样本数"])
    assert list(long["逾期标签"].drop_duplicates()) == ["days 0+", "days 3+"]
    for label, threshold in [("days 0+", 0), ("days 3+", 3)]:
        valid = data.loc[data.days.eq(0) | data.days.gt(threshold)].copy()
        valid["target"] = valid.days.gt(threshold)
        expected = rule.report(valid, target="target", prior_rules=prior, amount="amount", margins=True, filter_cols=["样本总数", "坏样本数"])
        actual = long.loc[long["逾期标签"].eq(label)].drop(columns="逾期标签").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        totals = actual.loc[actual["分箱"] == "合计"].set_index("规则分类")["样本总数"]
        assert totals["先验规则"] == valid.amount.sum()
        assert totals["验证规则"] == valid.loc[valid.x != 0, "amount"].sum()


def test_ruleset_long_report_matches_individual_labels():
    data = pd.DataFrame({"age": [18, 22, 25, 31, 33], "first": [0, 0, 1, 1, 0], "later": [0, 1, 1, 1, 0]})
    rules = [Rule("age >= 30"), Rule("age >= 21")]
    table = ruleset_analysis(data, rules, target={"早期": data["first"], "后期": data["later"]}, long_format=True)
    for label, source in [("早期", "first"), ("后期", "later")]:
        expected = ruleset_analysis(data, rules, target=source)
        actual = table.loc[table["逾期标签"] == label].drop(columns="逾期标签").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_multi_label_rule_analysis_writes_excel(tmp_path, monkeypatch):
    class DummyMiner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, df, features):
            self.df = df
            self.features = features

        def get_report(self):
            return pd.DataFrame(
                {
                    "规则类型": ["单特征规则"],
                    "规则": ["age > 20"],
                    "覆盖率": [0.5],
                }
            )

        def get_effectiveness_matrix(self):
            return pd.DataFrame(
                {
                    "规则": ["age > 20"],
                    "短期标签": [1.8],
                }
            )

    monkeypatch.setattr(rule_analysis_module, "MultiLabelRuleMiner", DummyMiner)

    output = tmp_path / "rule_analysis.xlsx"
    result = multi_label_rule_analysis(
        df=pd.DataFrame({"age": [18, 22], "label_a": [0, 1]}),
        features=["age"],
        labels={"短期标签": "label_a"},
        output_path=str(output),
    )

    assert result == str(output)
    workbook = load_workbook(output)
    assert {"规则汇总", "有效性矩阵", "规则分类统计"}.issubset(workbook.sheetnames)
