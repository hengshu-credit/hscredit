"""跨报告入口的删灰口径回归测试。"""

import pandas as pd

from hscredit.report import feature_binning_summary, feature_group_binning_summary


def test_feature_summaries_keep_target_specific_totals_by_customer_group():
    """特征汇总或客群汇总重新合并不同 DPD 的样本基数时，本测试必须失败。"""
    data = pd.DataFrame(
        {
            "特征": list(range(12)),
            "MOB1": [0, 2, 0, 5, 0, 2, 5, 8, 0, 4, 2, 8],
            "客群": ["A"] * 6 + ["B"] * 6,
        }
    )
    metrics = ["样本总数", "坏样本数", "坏样本率"]

    _, summary = feature_binning_summary(
        data,
        feature="特征",
        methods="quantile",
        overdue="MOB1",
        dpds=[1, 3],
        del_grey=True,
        metrics=metrics,
        max_n_bins=2,
        n_jobs=1,
    )
    assert summary.loc[0, ("样本总数", "MOB1@1")] == 12
    assert summary.loc[0, ("样本总数", "MOB1@3")] == 9

    _, group_summary = feature_group_binning_summary(
        data,
        feature="特征",
        methods="quantile",
        group_col="客群",
        overdue="MOB1",
        dpds=[1, 3],
        del_grey=True,
        metrics=metrics,
        max_n_bins=2,
        n_jobs=1,
    )
    by_group = group_summary.set_index(("分箱详情", "分组"))
    assert by_group.loc["A", ("样本总数", "MOB1@1")] == 6
    assert by_group.loc["A", ("样本总数", "MOB1@3")] == 4
    assert by_group.loc["B", ("样本总数", "MOB1@1")] == 6
    assert by_group.loc["B", ("样本总数", "MOB1@3")] == 5
