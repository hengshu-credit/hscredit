"""已披露设计的逐项复核：不把不同定义误改为同一种指标。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.metrics import auc, compute_bin_stats, psi_table, MonitoringBaseline
from hscredit.report import feature_bin_stats


def test_auc_auto_and_fixed_probability_direction_remain_explicit():
    y, score = [0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]
    assert auc(y, score, score_direction="auto") == 1
    assert auc(y, score, score_direction="higher_risk") == 0
    assert auc(y, score, score_direction="higher_safe") == 1


def test_nonmonotonic_cumulative_table_follows_feature_order_not_bad_rate_ranking():
    table = compute_bin_stats(np.repeat([0, 1, 2], 4), [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1], round_digits=False)
    # 两端坏率1/4和1/2，从右端累计；中间全坏箱仍居中，不能把箱标签与风险排序错配。
    np.testing.assert_array_equal(table["分箱"], [0, 1, 2])
    np.testing.assert_array_equal(table["累积坏样本数"], [7, 6, 2])
    assert "非按坏率重排" in table.attrs["统计口径"]["累计方向"]


def test_overdue_grey_and_missing_policy_is_preserved_and_disclosed():
    data = pd.DataFrame({"评分": np.arange(8), "MOB1": [-1, 0, 1, 3, 8, np.nan, 20, 0]})
    table = feature_bin_stats(
        data,
        "评分",
        overdue=["MOB1"],
        dpds=[3],
        del_grey=True,
        method="uniform",
        max_n_bins=2,
        margins=True,
        round_digits=False,
        n_jobs=1,
    )
    assert table.iloc[-1]["样本总数"] == 6
    assert table.iloc[-1]["坏样本数"] == 2
    assert table.attrs["标签口径"]["逾期比较符"] == ">"
    assert table.attrs["标签口径"]["剔灰"]
    assert "负数按比较符判定" in table.attrs["标签口径"]["缺失处理"]


def test_frozen_and_joint_psi_are_labeled_and_fixed_baseline_is_reusable():
    reference = np.arange(100, dtype=float)
    actual = reference + 10
    baseline = MonitoringBaseline(max_n_bins=4).fit(reference)
    version = baseline.version_
    direct = baseline.evaluate(actual)
    batched = baseline.evaluate_batches([actual[:40], actual[40:]])
    assert batched["PSI"] == pytest.approx(direct["PSI"])
    assert baseline.version_ == version
    assert psi_table(reference, actual).attrs["分箱口径"] == "冻结基准分箱"
    assert psi_table(reference, actual, method="uniform").attrs["分箱口径"] == "两期联合分箱"
