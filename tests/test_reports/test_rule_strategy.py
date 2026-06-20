import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.report import rule_group_hit_table, rule_report_table, rule_target_analysis, rule_target_table


@pytest.fixture
def multi_target_report():
    data = pd.DataFrame(
        {
            "score": [400, 450, 500, 550, 600, 650],
            "MOB1": [8, 6, 4, 2, 0, 0],
        }
    )
    report = Rule("score < 500", name="低分拒绝").report(data, overdue="MOB1", dpds=[3, 1])
    return data, report


def test_rule_report_table_adds_total_and_maps_target_names(multi_target_report):
    _, report = multi_target_report

    result = rule_report_table(
        report,
        rule_name="低分拒绝",
        target_names={"MOB1 3+": "fpd3", "MOB1 1+": "fpd1"},
    )

    assert isinstance(result.columns, pd.MultiIndex)
    assert list(result[("规则详情", "分箱")]) == ["命中", "未命中", "合计"]
    assert ("fpd3", "坏样本率") in result.columns
    assert ("fpd1", "LIFT值") in result.columns
    assert result.loc[2, ("fpd3", "样本总数")] == 6
    assert result.loc[2, ("fpd3", "坏样本率")] == pytest.approx(0.5)


def test_rule_target_analysis_uses_current_pass_rate(multi_target_report):
    _, report = multi_target_report

    result = rule_target_analysis(
        report,
        current_pass_rate=0.9,
        rule_name="低分拒绝",
        target_names={"MOB1 3+": "fpd3", "MOB1 1+": "fpd1"},
    )

    fpd3 = result[result[("低分拒绝", "逾期指标")] == "fpd3"].iloc[0]
    assert fpd3[("坏样本率", "合计")] == pytest.approx(0.5)
    assert fpd3[("坏样本率", "未命中")] == pytest.approx(0.25)
    assert fpd3[("规则指标", "拒绝LIFT")] == pytest.approx(2.0)
    assert fpd3[("绝对比例", "逾期改善")] == pytest.approx(0.25)
    assert fpd3[("绝对比例", "通过率")] == pytest.approx(2 / 3)
    assert fpd3[("相对比例", "逾期改善")] == pytest.approx(0.5)
    assert fpd3[("相对比例", "通过率")] == pytest.approx(0.6)


def test_rule_target_table_returns_long_format(multi_target_report):
    _, report = multi_target_report

    result = rule_target_table(
        report,
        rule_name="低分拒绝",
        target_names={"MOB1 3+": "fpd3", "MOB1 1+": "fpd1"},
    )

    assert list(result.columns[:3]) == ["规则详情", "逾期指标", "命中情况"]
    assert list(result["命中情况"]) == ["命中", "未命中", "合计", "命中", "未命中", "合计"]
    assert set(result["逾期指标"]) == {"fpd1", "fpd3"}


def test_rule_group_hit_table_expands_arbitrary_groups(multi_target_report):
    data, _ = multi_target_report
    rule = Rule("score < 500", name="低分拒绝")
    group_reports = {
        "分组1": rule.report(data.iloc[:3], overdue="MOB1", dpds=[3, 1]),
        "分组2": rule.report(data.iloc[3:], overdue="MOB1", dpds=[3, 1]),
    }

    result = rule_group_hit_table(
        group_reports,
        rule_name=rule.name,
        target_names={"MOB1 3+": "fpd3", "MOB1 1+": "fpd1"},
    )

    assert isinstance(result.columns, pd.MultiIndex)
    assert list(result[("规则详情", "是否命中")]) == ["命中", "未命中", "命中", "未命中"]
    assert ("坏样本率", "分组1") in result.columns
    assert ("样本占比", "分组2") in result.columns
    assert ("LIFT指标", "分组1") in result.columns
    assert ("样本总数", "分组2") in result.columns
    assert result.loc[0, ("样本总数", "分组1")] == 2
    assert not np.isnan(result.loc[0, ("坏样本率", "分组2")])


def test_rule_strategy_supports_single_target_report():
    data = pd.DataFrame({"score": [400, 500, 600, 700], "target": [1, 1, 0, 0]})
    report = Rule("score < 550").report(data, target="target")

    result = rule_target_table(report, rule_name="低分拒绝", target_name="fpd1")

    assert list(result["逾期指标"].unique()) == ["fpd1"]
    assert result.loc[result["命中情况"] == "合计", "样本总数"].iloc[0] == 4


@pytest.mark.parametrize("current_pass_rate", [-0.1, 1.1, "0.9"])
def test_rule_target_analysis_rejects_invalid_pass_rate(multi_target_report, current_pass_rate):
    _, report = multi_target_report

    with pytest.raises(ValueError, match="current_pass_rate"):
        rule_target_analysis(report, current_pass_rate=current_pass_rate)
