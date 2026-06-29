import pandas as pd
import pytest

from hscredit.core.rules import Rule
from hscredit.report.swap_analysis import swap_analysis


def _reference_df():
    return pd.DataFrame(
        {
            "score": list(range(480, 520)),
            "target": [0, 1] * 20,
        }
    )


def _run_swap_analysis(swap_types):
    swap_df = pd.DataFrame(
        {
            "score": list(range(500, 500 + len(swap_types))),
            "swap_type": swap_types,
        }
    )
    return swap_analysis(
        swap_df,
        _reference_df(),
        score_col="score",
        target="target",
        max_n_bins=3,
    )


@pytest.mark.parametrize(
    "swap_types, expected_stages",
    [
        (
            ["in-in", "out-in", "out-in"],
            ["原策略通过", "新策略置出(in-out)", "原策略保留(in-in)", "新策略置入(out-in)", "新策略通过"],
        ),
        (
            ["in-in", "in-out", "in-out"],
            ["原策略通过", "新策略置出(in-out)", "原策略保留(in-in)", "新策略置入(out-in)", "新策略通过"],
        ),
        (
            ["out-out", "in-in", "in-in"],
            [
                "原策略通过",
                "新策略置出(in-out)",
                "原策略保留(in-in)",
                "新策略置入(out-in)",
                "新策略通过",
                None,
                "原策略拒绝",
                "新策略置入(out-in)",
                "原策略保留(in-in)",
                "新策略置出(in-out)",
                "现策略拒绝",
                "新策略通过",
                "全部样本",
            ],
        ),
    ],
)
def test_swap_analysis_keeps_unified_stage_output(swap_types, expected_stages):
    result = _run_swap_analysis(swap_types)
    stages = result.summary_report_count.iloc[:, 0].where(
        result.summary_report_count.iloc[:, 0].notna(), None
    ).tolist()

    assert stages == expected_stages
    assert result.get_detail_report().iloc[:, 0].tolist() == ["in-in", "in-out", "out-in", "out-out"]


def test_swap_analysis_accepts_quadrant_rule_sets():
    swap_df = pd.DataFrame(
        {
            "score": [500, 510, 520, 530],
            "bucket": ["iin", "iout", "oin", "oout"],
        }
    )

    result = swap_analysis(
        swap_df,
        _reference_df(),
        score_col="score",
        target="target",
        rules_in_in=Rule("bucket == 'iin'"),
        rules_in_out=[Rule("bucket == 'iout'")],
        rules_out_in=Rule("bucket == 'oin'"),
        rules_out_out=[Rule("bucket == 'oout'")],
        max_n_bins=3,
    )

    detail = result.get_detail_report()
    assert detail.iloc[:, 0].tolist() == ["in-in", "in-out", "out-in", "out-out"]
    assert detail.iloc[:, 2].tolist() == [1, 1, 1, 1]


def test_swap_analysis_requires_swap_type_or_rule_sets():
    swap_df = pd.DataFrame({"score": [500, 510]})

    with pytest.raises(ValueError, match="swap_df 必须包含"):
        swap_analysis(
            swap_df,
            _reference_df(),
            score_col="score",
            target="target",
            max_n_bins=3,
        )


def test_swap_analysis_prioritizes_first_hit_across_modules():
    swap_df = pd.DataFrame(
        {
            "score": [500, 510, 520, 530],
            "bucket": ["shared", "iout", "iin", "oin"],
        }
    )

    result = swap_analysis(
        swap_df,
        _reference_df(),
        score_col="score",
        target="target",
        rules_out_out=Rule("bucket == 'shared'"),
        rules_in_out=[Rule("bucket == 'shared'"), Rule("bucket == 'iout'")],
        rules_in_in=Rule("bucket == 'iin'"),
        rules_out_in=[Rule("bucket == 'shared'"), Rule("bucket == 'oin'")],
        max_n_bins=3,
    )

    detail = result.get_detail_report()
    assert detail.iloc[:, 0].tolist() == ["in-in", "in-out", "out-in", "out-out"]
    assert detail.iloc[:, 2].tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("rule_execution_mode", ["parallel", "serial"])
def test_swap_analysis_rule_set_total_does_not_duplicate(rule_execution_mode):
    swap_df = pd.DataFrame(
        {
            "score": [500, 510],
            "a": [1, 0],
            "b": [1, 1],
        }
    )

    result = swap_analysis(
        swap_df,
        _reference_df(),
        score_col="score",
        target="target",
        rules_in_out=[Rule("a == 1"), Rule("b == 1")],
        rule_execution_mode=rule_execution_mode,
        max_n_bins=3,
    )

    detail = result.get_detail_report()
    assert detail.iloc[:, 0].tolist() == ["in-in", "in-out", "out-in", "out-out"]
    assert detail.iloc[:, 2].tolist() == [0, 2, 0, 0]


def test_swap_analysis_rejects_invalid_rule_execution_mode():
    swap_df = pd.DataFrame({"score": [500], "flag": [1]})

    with pytest.raises(ValueError, match="rule_execution_mode"):
        swap_analysis(
            swap_df,
            _reference_df(),
            score_col="score",
            target="target",
            rules_in_out=Rule("flag == 1"),
            rule_execution_mode="bad-mode",
            max_n_bins=3,
        )
