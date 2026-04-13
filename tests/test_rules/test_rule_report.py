import numpy as np
import pandas as pd

from hscredit.core.rules import Rule


def test_rule_report_uses_good_bad_distribution_denominators():
    data = pd.DataFrame(
        {
            "age": [18, 22, 30, 17],
            "target": [0, 1, 0, 1],
        }
    )

    report = Rule("age >= 21").report(data, target="target", margins=True)

    hit_row = report.loc[report["分箱"] == "命中"].iloc[0]
    miss_row = report.loc[report["分箱"] == "未命中"].iloc[0]
    total_row = report.loc[report["分箱"] == "合计"].iloc[0]

    assert hit_row["好样本占比"] == 0.5
    assert hit_row["坏样本占比"] == 0.5
    assert miss_row["好样本占比"] == 0.5
    assert miss_row["坏样本占比"] == 0.5
    assert total_row["好样本占比"] == 1.0
    assert total_row["坏样本占比"] == 1.0


def test_rule_report_zero_sample_bin_has_zero_risk_rejection_ratio():
    data = pd.DataFrame(
        {
            "age": [18, 22, 30],
            "target": [0, 1, 0],
        }
    )

    report = Rule("age >= 0").report(data, target="target")
    miss_row = report.loc[report["分箱"] == "未命中"].iloc[0]

    assert miss_row["样本总数"] == 0
    assert miss_row["样本占比"] == 0
    assert np.isfinite(miss_row["风险拒绝比"])
    assert miss_row["风险拒绝比"] == 0


def test_rule_report_multiindex_column_names_match_feature_bin_stats_style():
    data = pd.DataFrame(
        {
            "score": [450, 520, 610, 480, 700, 430],
            "MOB1": [0, 2, 4, 6, 1, 8],
        }
    )

    report = Rule("score < 500").report(data, overdue="MOB1", dpds=[5, 3, 1])

    assert isinstance(report.columns, pd.MultiIndex)
    top_level_names = list(dict.fromkeys(report.columns.get_level_values(0)))

    assert top_level_names[:4] == ["分箱详情", "MOB1 5+", "MOB1 3+", "MOB1 1+"]
    assert ("分箱详情", "规则分类") in report.columns
    assert ("MOB1 5+", "坏样本率") in report.columns
    assert ("MOB1 3+", "LIFT值") in report.columns
    assert ("MOB1 1+", "F1分数") in report.columns
    assert not any(column[1] == "指标含义" for column in report.columns)
    assert "规则详情" not in top_level_names
    assert not any("DPD" in name for name in top_level_names if isinstance(name, str))