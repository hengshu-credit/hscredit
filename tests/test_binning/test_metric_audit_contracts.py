"""结果表指标审计：以手算、守恒和独立聚合验证统计语义。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.metrics import add_margins, compute_bin_stats, ks_bucket


@pytest.mark.parametrize("round_digits", [False, True])
def test_unit_amounts_preserve_all_binary_statistics(round_digits):
    bins = np.repeat([0, 1, 2, -1, -2], 4)
    y = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    count = compute_bin_stats(bins, y, round_digits=round_digits)
    weighted = compute_bin_stats(
        bins, y, target_type="amount_weighted", amount=np.ones(len(y)), round_digits=round_digits
    )
    pd.testing.assert_frame_equal(count, weighted, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_amount_rejection_efficiency_uses_improvement_per_rejected_share():
    # 三箱金额各1000，坏金额100/500/900，总体坏率50%。
    # 拒绝首箱后坏率70%，改善-40%，除以1/3的拒绝占比得到-1.2。
    table = compute_bin_stats(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1],
        target_type="amount_weighted",
        amount=[900, 100, 500, 500, 100, 900],
        round_digits=False,
    )
    np.testing.assert_allclose(table["坏账改善"], [-0.4, 0, 0.4], atol=1e-12)
    np.testing.assert_allclose(table["风险拒绝比"], [-1.2, 0, 1.2], atol=1e-12)
    np.testing.assert_allclose(table["累计风险拒绝比"], [np.nan, 1.2, 1.2], atol=1e-12)


@pytest.mark.parametrize(
    "y,amount",
    [([0, 0, 0], [1, 1, 8]), ([1, 1, 1], [1, 1, 8]), ([0, 0, 1], [1, 8, 0]), ([0, 1, 1], [0, 0, 0])],
)
def test_no_effective_good_or_bad_amount_has_no_woe_or_iv(y, amount):
    table = compute_bin_stats([0, 0, 1], y, target_type="amount_weighted", amount=amount, round_digits=False)
    np.testing.assert_array_equal(table[["分档WOE值", "分档IV值", "指标IV值"]], 0)


def test_amount_raw_precision_is_retained_until_formatting():
    amounts = np.array([1.123456, 2.234567, 3.345678, 4.456789])
    raw = compute_bin_stats(
        [0, 0, 1, 1], [0, 1, 0, 1], target_type="amount_weighted", amount=amounts, round_digits=False
    )
    shown = compute_bin_stats([0, 0, 1, 1], [0, 1, 0, 1], target_type="amount_weighted", amount=amounts)
    np.testing.assert_allclose(raw["样本总数"], [amounts[:2].sum(), amounts[2:].sum()], atol=1e-14, rtol=0)
    np.testing.assert_allclose(raw["坏样本率"], [amounts[1] / amounts[:2].sum(), amounts[3] / amounts[2:].sum()])
    np.testing.assert_array_equal(shown["样本总数"], raw["样本总数"].round(2))
    np.testing.assert_array_equal(shown["坏样本率"], raw["坏样本率"].round(6))


@pytest.mark.parametrize("reverse", [False, True])
def test_ks_bucket_cumulative_rates_are_class_distributions(reverse):
    # 高风险一半包含3坏1好，全体4坏4好：累计坏/好占比应为3/4和1/4。
    scores = np.arange(8, dtype=float)
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    if reverse:
        scores = -scores
    table = ks_bucket(labels, scores, method="uniform", max_n_bins=2, n_jobs=1)
    risky = table.loc[table["坏样本率"].idxmax()]
    full = table.loc[table["坏样本率"].idxmin()]
    assert risky["累积坏样本率"] == pytest.approx(3 / 4)
    assert risky["累积好样本率"] == pytest.approx(1 / 4)
    assert full["累积坏样本率"] == pytest.approx(1)
    assert full["累积好样本率"] == pytest.approx(1)
    np.testing.assert_allclose(table["KS贡献"], abs(table["累积坏样本率"] - table["累积好样本率"]), atol=1e-10)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("offset", [0, 1e12])
def test_continuous_margins_reconstruct_overall_moments(multi, offset):
    values = np.array([1.0, 3.0, 10.0, 14.0]) + offset
    table = compute_bin_stats(
        [0, 0, 1, 1], values, target_type="continuous", bin_labels=["低", "高"], round_digits=False
    )
    if multi:
        table.columns = pd.MultiIndex.from_product([["金额"], table.columns])
    original = table.copy(deep=True)
    row = add_margins(table).iloc[-1]
    if multi:
        row = row["金额"]
    assert row["分箱标签"] == "合计"
    for name, expected in {
        "样本总数": 4,
        "目标值总和": values.sum(),
        "目标值均值": values.mean(),
        "目标值标准差": values.std(ddof=0),
        "目标值最小值": values.min(),
        "目标值最大值": values.max(),
        "累积样本数": 4,
        "累积目标值": values.sum(),
        "累积平均值": values.mean(),
        "目标值占比": 1,
        "平均LIFT值": 1,
        "累积LIFT值": 1,
        "贡献改善": 0,
        "累积贡献改善": 0,
    }.items():
        assert row[name] == pytest.approx(expected, abs=1e-9), name
    pd.testing.assert_frame_equal(table, original)
