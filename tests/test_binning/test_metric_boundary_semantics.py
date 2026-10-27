"""第二轮指标复核：无分母状态、尺度不变性与原始/截断指标。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.metrics import BinStatsAccumulator, add_margins, compute_bin_stats


@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("weighted", [False, True])
def test_single_class_bin_ks_is_zero_instead_of_perfect_separation(label, weighted):
    kwargs = dict(target_type="amount_weighted", amount=[1, 4, 2, 8]) if weighted else {}
    table = compute_bin_stats([0, 0, 1, -1], np.full(4, label), round_digits=False, **kwargs)
    np.testing.assert_array_equal(table["分档KS值"], 0)
    assert "单类" in table.attrs["统计口径"]["KS状态"]


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_amount_ks_is_invariant_to_currency_scale(scale):
    table = compute_bin_stats(
        [0, 0, 1, 1], [0, 0, 1, 1], target_type="amount_weighted", amount=np.full(4, scale), round_digits=False
    )
    np.testing.assert_array_equal(table["分档KS值"], [0, 1])


@pytest.mark.parametrize("target", [[0, 1, 0, 1], [0, 0, 0, 0]])
@pytest.mark.parametrize("weighted", [False, True])
def test_full_rejection_keeps_improvement_and_nan_efficiency(target, weighted):
    kwargs = dict(target_type="amount_weighted", amount=[1, 2, 3, 4]) if weighted else {}
    table = compute_bin_stats([0, 0, 0, 0], target, round_digits=False, **kwargs)
    expected = 1.0 if any(target) else 0.0
    np.testing.assert_array_equal(table[["坏账改善", "累积坏账改善"]], expected)
    assert table[["风险拒绝比", "累计风险拒绝比"]].isna().all().all()


def test_accumulator_shares_boundary_values_and_metadata():
    bins, y = [0, 1, 0, 1], [0, 0, 0, 0]
    actual = BinStatsAccumulator().update(bins[:2], y[:2]).update(bins[2:], y[2:]).finalize(round_digits=False)
    expected = compute_bin_stats(bins, y, round_digits=False)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs


def test_clipped_woe_table_exposes_original_information_value():
    bins, y = [0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]
    raw = compute_bin_stats(bins, y, round_digits=False)
    clipped = compute_bin_stats(bins, y, round_digits=False, woe_clip=0.1, bin_labels=["低", "高"])
    np.testing.assert_allclose(clipped["原始分档WOE值"], [-np.log(2), np.log(2)])
    np.testing.assert_allclose(clipped["原始分档IV值"], raw["分档IV值"])
    np.testing.assert_allclose(clipped["原始指标IV值"], raw["指标IV值"])
    np.testing.assert_allclose(clipped["分档IV值"], [0.1 / 3, 0.1 / 3])
    total = add_margins(clipped).iloc[-1]
    assert total["原始指标IV值"] == pytest.approx(2 * np.log(2) / 3)
    assert total["原始分档IV值"] == 0


def test_margins_are_idempotent_and_keep_per_feature_populations():
    first = compute_bin_stats([0, 0, 1, 1], [0, 1, 0, 1], bin_labels=["低", "高"], round_digits=False)
    first.insert(0, "指标名称", "甲")
    second = first.copy()
    second["指标名称"] = "乙"
    table = pd.concat([first, second], ignore_index=True)
    once = add_margins(table)
    totals = once[once["分箱标签"] == "合计"]
    assert totals["指标名称"].tolist() == ["甲", "乙"]
    assert totals["样本总数"].tolist() == [4, 4]
    pd.testing.assert_frame_equal(add_margins(once), once)


def test_long_target_margins_do_not_merge_distinct_target_populations():
    first = compute_bin_stats([0, 0, 1, 1], [0, 1, 0, 1], bin_labels=["低", "高"], round_digits=False)
    first.insert(0, "指标名称", "甲")
    first.insert(1, "逾期标签", "短期")
    second = first.copy()
    second["逾期标签"] = "长期"
    original = pd.concat([add_margins(first), add_margins(second)], ignore_index=True)
    repeated = add_margins(original)
    pd.testing.assert_frame_equal(repeated, original)
    assert repeated.loc[repeated["分箱标签"] == "合计", "样本总数"].tolist() == [4, 4]


def test_single_class_margins_do_not_invent_missing_class_share():
    table = compute_bin_stats([0, 0, 1], [0, 0, 0], bin_labels=["低", "高"], round_digits=False)
    total = add_margins(table).iloc[-1]
    assert total["好样本占比"] == 1
    assert total["坏样本占比"] == 0
    assert total["分档KS值"] == 0
    assert total["累积坏账改善"] == 0


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [-2.0, 1.0, 1.0], [1.0, -1.0, 2.0]])
def test_continuous_non_distribution_targets_keep_moments_but_not_fake_contribution(values):
    table = compute_bin_stats([0, 0, 1], values, target_type="continuous", bin_labels=["低", "高"], round_digits=False)
    assert table["目标值总和"].sum() == pytest.approx(sum(values))
    for column in ["目标值占比", "平均LIFT值", "贡献改善", "累积LIFT值", "累积贡献改善", "分档KS值"]:
        assert table[column].isna().all(), column
    total = add_margins(table).iloc[-1]
    assert total["目标值均值"] == pytest.approx(np.mean(values))
    assert total["目标值标准差"] == pytest.approx(np.std(values))
    assert pd.isna(total["目标值占比"])
