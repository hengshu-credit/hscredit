"""推荐绘图入口的公开契约回归测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hscredit.core.viz import bin_plot, ks_plot, lift_plot


def test_bin_plot_aligns_array_target_positionally_and_keeps_missing_bin():
    """非默认索引和缺失特征不能导致目标错位或样本丢失。"""
    score = pd.Series(
        [510.0, np.nan, 620.0, 680.0, 710.0, 760.0],
        index=[10, 20, 30, 40, 50, 60],
        name="评分",
    )
    target = np.array([1, 0, 1, 0, 0, 1])

    fig, table = bin_plot(score, target=target, n_bins=3, return_frame=True)

    assert table["样本总数"].sum() == len(score)
    missing = table.loc[table["分箱标签"].eq("缺失值")].iloc[0]
    assert missing["样本总数"] == 1
    assert missing["坏样本数"] == 0
    assert missing["坏样本率"] == 0
    plt.close(fig)


def test_bin_plot_returns_frame_when_drawing_on_existing_axes():
    """嵌入已有 Axes 时，return_frame=True 仍应返回绘图轴和统计表。"""
    score = pd.Series(np.arange(20, dtype=float), name="评分")
    target = pd.Series([0, 1] * 10, name="目标")
    fig, ax = plt.subplots()

    returned_ax, table = bin_plot(score, target=target, n_bins=4, ax=ax, return_frame=True)

    assert returned_ax is ax
    assert table["样本总数"].sum() == len(score)
    plt.close(fig)


def test_ks_plot_respects_explicit_positive_label_and_score_direction():
    """字符串标签必须由 pos_label 明确正类，不能按字典序静默反转。"""
    score = np.array([0.95, 0.85, 0.20, 0.10])
    target = np.array(["bad", "bad", "good", "good"])

    fig = ks_plot(
        score,
        target,
        curve="roc",
        pos_label="bad",
        score_direction="higher_risk",
    )

    assert any("AUC: 1.0000" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)


def test_ks_plot_accepts_numpy_axes_array():
    """plt.subplots 返回的 ndarray 应能直接作为 axes 参数传入。"""
    score = np.linspace(0.05, 0.95, 20)
    target = np.array([0, 1] * 10)
    fig, axes = plt.subplots(1, 2)

    returned = ks_plot(score, target, axes=axes)

    assert returned is axes
    plt.close(fig)


def test_lift_plot_rejects_more_bins_than_samples_in_chinese():
    """分箱数超过样本数时应明确拒绝，不能生成空箱和 NaN 曲线。"""
    with pytest.raises(ValueError, match="分箱数.*样本数"):
        lift_plot(
            np.array([0, 1, 0]),
            np.array([0.1, 0.9, 0.2]),
            n_bins=10,
        )


def test_ks_plot_preserves_explicit_title_and_uses_chinese_axes():
    """显式标题必须原样优先，推荐入口的轴标签应使用中文。"""
    score = np.linspace(0.05, 0.95, 20)
    target = np.array([0, 1] * 10)

    fig = ks_plot(score, target, title="模型区分能力", score_direction="higher_risk")

    assert fig._suptitle.get_text() == "模型区分能力"
    assert fig.axes[0].get_xlabel() == "累计样本占比"
    assert fig.axes[1].get_xlabel() == "假正例率"
    plt.close(fig)


def test_lift_plot_uses_chinese_default_labels():
    """推荐 Lift 入口的默认标题和轴标签应满足中文输出约定。"""
    target = np.array([0, 1] * 10)
    score = np.linspace(0.05, 0.95, 20)

    fig = lift_plot(target, score, n_bins=5)

    ax = fig.axes[0]
    assert ax.get_title() == "Lift 提升图"
    assert ax.get_xlabel() == "样本深度（累计占比）"
    assert ax.get_ylabel() == "Lift 值"
    plt.close(fig)
