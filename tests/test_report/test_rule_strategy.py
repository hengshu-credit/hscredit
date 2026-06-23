import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.report import (
    rule_group_compare,
    rule_group_hit_table,
    rule_report_table,
    rule_target_analysis,
    rule_target_table,
)


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


def test_rule_group_compare_groups_by_date_freq():
    data = pd.DataFrame(
        {
            "score": [400, 600, 450, 650],
            "放款时间": ["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-15"],
            "target": [1, 0, 1, 0],
        }
    )

    result = rule_group_compare(data, "score < 500", date_col="放款时间", freq="M", target="target", rule_name="低分拒绝")

    assert isinstance(result.columns, pd.MultiIndex)
    assert list(result[("规则详情", "是否命中")]) == ["命中", "未命中"]
    assert ("坏样本率", "2024-01") in result.columns
    assert ("样本总数", "2024-02") in result.columns
    assert result.loc[0, ("样本总数", "2024-01")] == 1


def test_rule_group_compare_groups_by_group_col_multi_label():
    data = pd.DataFrame(
        {
            "score": [400, 450, 600, 500, 650, 420],
            "MOB1": [8, 6, 0, 4, 0, 2],
            "渠道": ["A", "A", "A", "B", "B", "B"],
        }
    )

    result = rule_group_compare(
        data,
        Rule("score < 500", name="低分拒绝"),
        group_col="渠道",
        overdue="MOB1",
        dpds=[3, 1],
    )

    assert ("样本总数", "A") in result.columns
    assert ("样本总数", "B") in result.columns
    assert set(result[("规则详情", "逾期指标")]) == {"MOB1 3+", "MOB1 1+"}
    assert result.loc[0, ("规则详情", "规则名称")] == "低分拒绝"


def test_rule_group_compare_requires_exactly_one_grouping():
    data = pd.DataFrame({"score": [400, 600], "target": [1, 0]})
    with pytest.raises(ValueError):
        rule_group_compare(data, "score < 500", target="target")


def _group_order_data():
    return pd.DataFrame(
        {
            "score": [400, 600, 450, 650, 420, 610],
            "target": [1, 0, 1, 0, 1, 0],
            "渠道": ["B", "B", "C", "C", "A", "A"],
        }
    )


def _group_columns(result, level0="样本总数"):
    return list(dict.fromkeys(group for top, group in result.columns if top == level0))


def test_rule_group_compare_group_order_default_ascending():
    result = rule_group_compare(_group_order_data(), "score < 500", group_col="渠道", target="target")
    assert _group_columns(result) == ["A", "B", "C"]


def test_rule_group_compare_group_order_desc_and_appearance():
    data = _group_order_data()
    desc = rule_group_compare(data, "score < 500", group_col="渠道", target="target", group_order="desc")
    assert _group_columns(desc) == ["C", "B", "A"]

    appearance = rule_group_compare(data, "score < 500", group_col="渠道", target="target", group_order="appearance")
    assert _group_columns(appearance) == ["B", "C", "A"]


def test_rule_group_compare_group_order_explicit_sequence_appends_remaining():
    result = rule_group_compare(
        _group_order_data(), "score < 500", group_col="渠道", target="target", group_order=["C", "A"]
    )
    assert _group_columns(result) == ["C", "A", "B"]


def test_rule_group_compare_group_order_callable():
    result = rule_group_compare(
        _group_order_data(),
        "score < 500",
        group_col="渠道",
        target="target",
        group_order=lambda g: {"A": 2, "B": 0, "C": 1}[g],
    )
    assert _group_columns(result) == ["B", "C", "A"]


def test_rule_group_compare_amount_and_kwargs_passthrough():
    data = pd.DataFrame(
        {
            "score": [400, 600, 450, 650],
            "target": [1, 0, 1, 0],
            "金额": [100, 200, 300, 400],
            "渠道": ["A", "A", "B", "B"],
        }
    )

    result = rule_group_compare(data, "score < 500", group_col="渠道", target="target", amount="金额", margins=True)

    # 金额口径下 样本总数 为金额加权
    assert result.loc[0, ("样本总数", "A")] == 100
    assert result.loc[0, ("样本总数", "B")] == 300


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
