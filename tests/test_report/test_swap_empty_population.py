"""置换分析没有通过人群时，风险变化不可伪装成零坏率或100%改善。"""

import numpy as np
import pandas as pd

from hscredit.core.rules import Rule
from hscredit.report import rule_swap_analysis
from hscredit.report.swap_analysis import swap_analysis


def test_swap_analyzer_all_rejected_risk_is_undefined_and_exportable_as_blank():
    reference = pd.DataFrame({"score": np.arange(40), "target": [0, 1] * 20})
    current = pd.DataFrame({"score": [1, 10, 20, 30], "swap_type": ["in-out"] * 4})
    result = swap_analysis(current, reference, score_col="score", target="target", max_n_bins=2, n_jobs=1)
    assert result.pass_rate_analysis.new_pass_rate == 0
    risk = result.risk_rejection_metrics
    assert pd.isna(risk.bad_rate_after)
    assert risk.bad_rate_improvement == risk.bad_rate_before
    assert risk.bad_rate_relative_improvement == 100
    assert pd.isna(risk.risk_rejection_rate)
    row = result.summary_report_count.set_index("阶段").loc["新策略通过"]
    assert pd.isna(row["坏样本率"])
    assert pd.isna(row["坏账率变化(相对)"])


def test_rule_swap_empty_new_pass_population_has_no_measured_bad_rate():
    data = pd.DataFrame({"score": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    bins = pd.DataFrame({"分箱标签": ["[-inf, 2.5)", "[2.5, +inf)"], "坏样本率": [0.5, 0.5]})
    result = rule_swap_analysis(
        data, score="score", target="target", bin_table=bins, rules_out=[Rule("score >= 0")], n_jobs=1
    )
    change = result["swap_result"].set_index("指标").loc["逾期率"]
    assert pd.isna(change["变化后"])
    assert pd.isna(change["相对变化"])


def test_swap_summary_uses_all_samples_as_baseline_when_no_out_out_group():
    reference = pd.DataFrame({"score": np.arange(40), "target": [0] * 20 + [1] * 20})
    current = pd.DataFrame({"score": [1, 10, 20, 30], "swap_type": ["in-in", "in-in", "in-out", "in-out"]})
    result = swap_analysis(current, reference, score_col="score", target="target", max_n_bins=2, n_jobs=1)
    row = result.summary_report_count.set_index("阶段").loc["新策略通过"]
    assert float(row["坏样本率"]) == 0
    assert float(row["LIFT"]) == 0
    assert float(row["风险改善"]) == 1
