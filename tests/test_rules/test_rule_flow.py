import pandas as pd

from hscredit.core.rules import Rule, RuleFlow


def _flow_data():
    return pd.DataFrame(
        {
            "score": [420, 560, 680, 610, 450],
            "multi": [2, 8, 9, None, None],
            "channel": ["A", "A", "B", "B", "B"],
            "apply_date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-02-01", "2024-02-15", None]),
        }
    )


def test_rule_flow_serial_stops_after_first_hit_and_reports_current_denominators():
    data = _flow_data()
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="serial",
    )

    prediction = flow.predict(data)

    assert prediction["低分拒绝"].tolist() == [True, False, False, False, True]
    assert prediction["多头拒绝"].astype(object).tolist() == [pd.NA, True, True, False, pd.NA]
    assert prediction["命中规则"].tolist() == ["低分拒绝", "多头拒绝", "多头拒绝", "", "低分拒绝"]
    assert prediction["是否通过"].tolist() == [False, False, False, True, False]

    report = flow.report(data)
    first_rule = report.loc[report["规则名称"] == "低分拒绝"].iloc[0]
    second_rule = report.loc[report["规则名称"] == "多头拒绝"].iloc[0]

    assert first_rule["统计类型"] == "明细"
    assert first_rule["统计范围样本数"] == 5
    assert first_rule["当前规则样本数"] == 5
    assert first_rule["规则命中"] == 2
    assert first_rule["命中率(统计范围)"] == 0.4
    assert first_rule["命中率(当前规则)"] == 0.4
    assert first_rule["规则通过"] == 3
    assert first_rule["通过率(统计范围)"] == 0.6
    assert first_rule["通过率(当前规则)"] == 0.6

    assert second_rule["统计范围样本数"] == 5
    assert second_rule["当前规则样本数"] == 3
    assert second_rule["规则命中"] == 2
    assert second_rule["命中率(统计范围)"] == 0.4
    assert second_rule["命中率(当前规则)"] == 2 / 3
    assert second_rule["规则通过"] == 1
    assert second_rule["通过率(统计范围)"] == 0.2
    assert second_rule["通过率(当前规则)"] == 1 / 3

    overall_report = report.loc[report["统计类型"] == "整体合计"].iloc[0]
    assert overall_report["统计范围样本数"] == 5
    assert overall_report["当前规则样本数"] == 5
    assert overall_report["规则命中"] == 4
    assert overall_report["规则通过"] == 1

    summary = flow.summary(data)
    assert summary.iloc[0].to_dict() == {
        "统计类型": "整体合计",
        "样本总数": 5,
        "通过样本": 1,
        "命中样本": 4,
        "通过率": 0.2,
        "命中率": 0.8,
    }


def test_rule_flow_parallel_evaluates_all_rules_for_all_samples():
    data = _flow_data()
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="parallel",
    )

    prediction = flow.predict(data)

    assert prediction["低分拒绝"].tolist() == [True, False, False, False, True]
    assert prediction["多头拒绝"].tolist() == [False, True, True, False, False]
    assert prediction["命中规则"].tolist() == ["低分拒绝", "多头拒绝", "多头拒绝", "", "低分拒绝"]
    assert prediction["是否通过"].tolist() == [False, False, False, True, False]

    report = flow.report(data)
    detail = report[report["统计类型"] == "明细"]
    assert detail["统计范围样本数"].tolist() == [5, 5]
    assert detail["当前规则样本数"].tolist() == [5, 5]
    assert detail["规则命中"].tolist() == [2, 2]
    assert detail["规则通过"].tolist() == [3, 3]
    assert detail["命中率(统计范围)"].tolist() == [0.4, 0.4]
    assert detail["命中率(当前规则)"].tolist() == [0.4, 0.4]
    overall = report.loc[report["统计类型"] == "整体合计"].iloc[0]
    assert overall["规则命中"] == 4
    assert overall["规则通过"] == 1


def test_rule_flow_report_and_summary_support_date_and_category_groups():
    data = _flow_data()
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="serial",
    )

    report = flow.report(data, date_col="apply_date", freq="M", group_cols="channel")
    summary = flow.summary(data, date_col="apply_date", freq="M", group_cols="channel")

    assert set(["统计周期", "channel"]).issubset(report.columns)
    assert report.loc[report["统计类型"] != "整体合计", ["统计周期", "channel"]].drop_duplicates().to_dict("records") == [
        {"统计周期": "2024-01", "channel": "A"},
        {"统计周期": "2024-02", "channel": "B"},
    ]

    january_summary = summary.loc[
        (summary["统计类型"] == "分组合计") & (summary["统计周期"] == "2024-01") & (summary["channel"] == "A")
    ].iloc[0]
    assert january_summary["样本总数"] == 2
    assert january_summary["命中样本"] == 2
    assert january_summary["通过样本"] == 0
    assert january_summary["命中率"] == 1.0

    overall_summary = summary.loc[summary["统计类型"] == "整体合计"].iloc[0]
    assert overall_summary["统计周期"] == "全部"
    assert overall_summary["channel"] == "全部"
    assert overall_summary["样本总数"] == 5
    assert overall_summary["命中样本"] == 4
    assert overall_summary["通过样本"] == 1

    february_second_rule = report.loc[
        (report["统计类型"] == "明细")
        & (report["统计周期"] == "2024-02")
        & (report["channel"] == "B")
        & (report["规则名称"] == "多头拒绝")
    ].iloc[0]
    assert february_second_rule["当前规则样本数"] == 2
    assert february_second_rule["规则命中"] == 1
    assert february_second_rule["通过率(统计范围)"] == 0.5

    group_total = report.loc[
        (report["统计类型"] == "分组合计") & (report["统计周期"] == "2024-02") & (report["channel"] == "B")
    ].iloc[0]
    assert group_total["统计范围样本数"] == 2
    assert group_total["规则命中"] == 1
    assert group_total["规则通过"] == 1


def test_rule_flow_compare_accepts_rule_name_list_column_and_outputs_diff_detail():
    data = _flow_data()
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="serial",
    )
    production_hits = pd.DataFrame(
        {
            "命中规则": [
                ["低分拒绝"],
                "低分拒绝",
                ["多头拒绝"],
                [],
                ["低分拒绝"],
            ]
        },
        index=data.index,
    )

    report, detail = flow.compare(data, production_hits)

    first_rule = report.loc[report["规则名称"] == "低分拒绝"].iloc[0]
    second_rule = report.loc[report["规则名称"] == "多头拒绝"].iloc[0]
    assert first_rule["样本总数"] == 5
    assert first_rule["一致样本"] == 4
    assert first_rule["差异样本"] == 1
    assert first_rule["一致率"] == 0.8
    assert first_rule["差异率"] == 0.2
    assert second_rule["一致样本"] == 4
    assert second_rule["差异样本"] == 1

    assert detail.index.tolist() == [1]
    diff = detail.iloc[0]
    assert diff["线下命中规则"] == "多头拒绝"
    assert diff["线上命中规则"] == "低分拒绝"
    assert diff["线下独有规则"] == "多头拒绝"
    assert diff["线上独有规则"] == "低分拒绝"
    assert diff["差异规则"] == "低分拒绝|多头拒绝"
    assert not bool(diff["线下_低分拒绝"])
    assert bool(diff["线上_低分拒绝"])
    assert not bool(diff["是否一致_低分拒绝"])


def test_rule_flow_compare_accepts_hit_matrix_and_order_id_alignment():
    data = _flow_data().copy()
    data["order_id"] = ["o1", "o2", "o3", "o4", "o5"]
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="parallel",
    )
    production_hits = pd.DataFrame(
        {
            "order_id": ["o5", "o4", "o3", "o2", "o1"],
            "低分拒绝": [1, 0, 0, 0, 1],
            "多头拒绝": [0, 0, 0, 1, 0],
        }
    )

    report, detail = flow.compare(data, production_hits, order_id_col="order_id")

    first_rule = report.loc[report["规则名称"] == "低分拒绝"].iloc[0]
    second_rule = report.loc[report["规则名称"] == "多头拒绝"].iloc[0]
    assert first_rule["一致样本"] == 5
    assert first_rule["差异样本"] == 0
    assert second_rule["一致样本"] == 4
    assert second_rule["差异样本"] == 1

    assert detail["order_id"].tolist() == ["o3"]
    assert detail.iloc[0]["线下命中规则"] == "多头拒绝"
    assert detail.iloc[0]["线上命中规则"] == ""
    assert detail.iloc[0]["差异规则"] == "多头拒绝"


def test_rule_flow_compare_parses_string_hit_matrix_values():
    data = _flow_data()
    flow = RuleFlow(
        [
            Rule("score < 500", name="低分拒绝"),
            Rule("multi > 6", name="多头拒绝"),
        ],
        mode="parallel",
    )
    production_hits = pd.DataFrame(
        {
            "低分拒绝": ["1", "0", "false", "False", "true"],
            "多头拒绝": ["0", "1", "true", "False", ""],
        },
        index=data.index,
    )

    report, detail = flow.compare(data, production_hits)

    assert report["差异样本"].tolist() == [0, 0]
    assert detail.empty
