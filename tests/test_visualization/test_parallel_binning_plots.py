"""分箱组合图的有序并行计算与共享统计契约。"""

from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hscredit.core.viz import binning_plots
from hscredit.core.viz.binning_plots import (
    _compute_feature_bin_stats,
    batch_bin_trend_plot,
    bin_overdues_plot,
    bin_trend_plot,
)


def _sample_data() -> pd.DataFrame:
    """构造每个分组和逾期目标都同时包含 0/1 的稳定样本。"""
    target = np.tile([0, 1], 30)
    feature = np.arange(60, dtype=float)
    return pd.DataFrame(
        {
            "分组": np.repeat(["B", "A", "C"], 20),
            "目标": target,
            "强特征": target + np.tile([0.0, 0.1, 0.2], 20),
            "弱特征": feature % 7,
            "逾期7天": np.where(target == 1, 10, 0),
            "逾期15天": np.where(np.roll(target, 1) == 1, 20, 0),
            "逾期30天": np.where(np.roll(target, 2) == 1, 40, 0),
        }
    )


def _record_parallel_batches(monkeypatch):
    """保留真实 worker 行为，只把共享执行器替换为可观测的顺序执行器。"""
    calls = []

    def recording_execute(function, tasks, *, task_labels=None, workload=None, **kwargs):
        task_list = list(tasks)
        labels = list(task_labels) if task_labels is not None else None
        calls.append(
            {
                "function": function.__name__,
                "task_count": len(task_list),
                "labels": labels,
                "workload": workload,
                "kwargs": kwargs,
            }
        )
        return [function(task) for task in task_list]

    monkeypatch.setattr(binning_plots, "parallel_execute", recording_execute, raising=False)
    return calls


def test_bin_trend_submits_overall_and_groups_once_and_keeps_panel_order(monkeypatch):
    """删除外层并行批次或打乱任务标签时必须失败。"""
    data = _sample_data()
    calls = _record_parallel_batches(monkeypatch)

    fig = bin_trend_plot(
        data,
        "强特征",
        "目标",
        dimension_cols="分组",
        method="quantile",
        shared_bins=False,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"adaptive": False},
        show_stats=False,
    )

    trend_calls = [call for call in calls if call["workload"].operation == "分箱趋势面板统计"]
    assert len(trend_calls) == 1
    assert trend_calls[0]["task_count"] == 4
    assert trend_calls[0]["labels"] == ["Overall", "A", "B", "C"]
    assert [axis.get_title().splitlines()[0] for axis in fig.axes[:4]] == ["Overall", "A", "B", "C"]
    plt.close(fig)


def test_bin_trend_max_cols_controls_layout():
    """继续硬编码三列或忽略 max_cols 时必须失败。"""
    fig = bin_trend_plot(
        _sample_data(),
        "强特征",
        "目标",
        dimension_cols="分组",
        method="quantile",
        shared_bins=False,
        max_cols=2,
        n_jobs=1,
        show_stats=False,
    )

    panel_x_positions = {round(axis.get_position().x0, 3) for axis in fig.axes[:4]}
    assert len(panel_x_positions) == 2
    plt.close(fig)


def test_bin_trend_multirow_multicolumn_layout_shares_canvas_draws(monkeypatch):
    """行列布局分别重复绘制整张画布而放大子图渲染成本时必须失败。"""
    original_draw = FigureCanvasAgg.draw
    draw_count = 0

    def counting_draw(canvas, *args, **kwargs):
        nonlocal draw_count
        draw_count += 1
        return original_draw(canvas, *args, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, "draw", counting_draw)
    rng = np.random.default_rng(20260825)
    data = pd.DataFrame(
        {
            "分组": np.repeat(["A", "B", "C", "D", "E", "F"], 40),
            "目标": np.tile([0, 1], 120),
            "强特征": rng.normal(size=240),
        }
    )
    fig = bin_trend_plot(
        data,
        "强特征",
        "目标",
        dimension_cols="分组",
        method="quantile",
        shared_bins=False,
        max_cols=3,
        n_jobs=1,
        show_stats=False,
    )

    assert draw_count <= 10
    plt.close(fig)


def test_batch_bin_trend_reuses_prepared_overall_stats(monkeypatch):
    """恢复“先排名、后绘图再拟合”的重复整体统计时必须失败。"""
    data = _sample_data()
    calls = _record_parallel_batches(monkeypatch)
    original = binning_plots._compute_feature_bin_stats
    counts = Counter()

    def counting_stats(frame, feature, target, *args, **kwargs):
        counts[feature] += 1
        return original(frame, feature, target, *args, **kwargs)

    monkeypatch.setattr(binning_plots, "_compute_feature_bin_stats", counting_stats)

    figures = batch_bin_trend_plot(
        data,
        ["强特征", "弱特征"],
        "目标",
        method="quantile",
        max_features=2,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"adaptive": False},
        show_stats=False,
    )

    batch_calls = [call for call in calls if call["workload"].operation == "批量分箱趋势统计"]
    assert len(batch_calls) == 1
    assert batch_calls[0]["labels"] == ["强特征", "弱特征"]
    assert counts == Counter({"强特征": 1, "弱特征": 1})
    assert list(figures) == ["强特征", "弱特征"]
    for figure in figures.values():
        plt.close(figure)


def test_bin_overdues_submits_targets_once_and_keeps_input_order(monkeypatch):
    """逐个串行统计或按完成先后组图时必须失败。"""
    data = _sample_data()
    calls = _record_parallel_batches(monkeypatch)
    overdue = ["逾期15天", "逾期7天", "逾期30天"]
    dpds = [15, 7, 30]

    fig = bin_overdues_plot(
        data,
        feature="强特征",
        overdue=overdue,
        dpds=dpds,
        method="quantile",
        shared_bins=False,
        max_cols=2,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"adaptive": False},
        show_stats=False,
    )

    overdue_calls = [call for call in calls if call["workload"].operation == "多逾期分箱统计"]
    assert len(overdue_calls) == 1
    assert overdue_calls[0]["labels"] == ["逾期15天 (>= 15)", "逾期7天 (>= 7)", "逾期30天 (>= 30)"]
    assert [axis.get_title() for axis in fig.axes[:3]] == overdue_calls[0]["labels"]
    plt.close(fig)


def test_feature_bin_stats_keeps_all_missing_feature_as_explicit_special_bin():
    """再次在拟合前删除特征缺失值时必须失败。"""
    data = pd.DataFrame({"特征": [np.nan] * 8, "目标": np.tile([0, 1], 4)})

    stats = _compute_feature_bin_stats(
        data,
        "特征",
        "目标",
        method="mdlp",
        special_codes=[np.nan],
    )

    assert stats["分箱"].tolist() == [-2]
    assert stats["分箱标签"].tolist() == ["特殊值"]
    assert stats["样本总数"].tolist() == [8]


def test_feature_bin_stats_maps_labels_by_bin_id():
    """按分箱表行位置映射标签导致普通箱与特殊箱错位时必须失败。"""
    data = pd.DataFrame(
        {
            "特征": [0.0, 1.0, 3.0, 5.0, 99.0],
            "目标": [0, 1, 0, 1, 0],
        }
    )

    stats = _compute_feature_bin_stats(
        data,
        "特征",
        "目标",
        rules={"特征": [2.0, 4.0]},
        special_codes=[99.0],
        user_splits_fixed=True,
    )

    assert stats["分箱"].tolist() == [0, 1, 2, -2]
    assert stats["分箱标签"].tolist() == ["[-inf, 2)", "[2, 4)", "[4, +inf)", "特殊值"]


def test_bin_trend_preserves_specific_binner_error():
    """把单类别目标异常吞成“无法计算特征”时必须失败。"""
    data = pd.DataFrame({"特征": np.arange(12, dtype=float), "目标": np.zeros(12, dtype=int)})

    with pytest.raises(ValueError, match="目标变量必须是二分类"):
        bin_trend_plot(data, "特征", "目标", method="mdlp", n_jobs=1)


def test_bin_trend_save_honors_explicit_dpi(monkeypatch, tmp_path):
    """渲染重构再次丢弃公开 dpi 参数时必须失败。"""
    saved = []

    def recording_save(figure, path, dpi=240):
        saved.append((path, dpi))

    monkeypatch.setattr(binning_plots, "save_figure", recording_save)
    figure = bin_trend_plot(
        _sample_data(),
        "强特征",
        "目标",
        method="quantile",
        n_jobs=1,
        dpi=137,
        save=tmp_path / "trend.png",
        show_stats=False,
    )

    assert saved == [(tmp_path / "trend.png", 137)]
    plt.close(figure)


def test_batch_bin_trend_save_dir_preserves_150_dpi_default(monkeypatch, tmp_path):
    """批量保存从既有 150 DPI 漂移到 save_figure 默认值时必须失败。"""
    saved = []

    def recording_save(figure, path, dpi=240):
        saved.append((path, dpi))

    monkeypatch.setattr(binning_plots, "save_figure", recording_save)
    figures = batch_bin_trend_plot(
        _sample_data(),
        ["强特征"],
        "目标",
        method="quantile",
        n_jobs=1,
        save_dir=tmp_path,
        show_stats=False,
    )

    assert saved == [(str(tmp_path / "强特征_trend.png"), 150)]
    for figure in figures.values():
        plt.close(figure)


def test_bin_overdues_rejects_empty_definitions_before_layout():
    """空逾期定义暴露 ZeroDivisionError 而非中文参数错误时必须失败。"""
    with pytest.raises(ValueError, match="overdue 和 dpds 不能为空"):
        bin_overdues_plot(
            _sample_data(),
            feature="强特征",
            overdue=[],
            dpds=[],
            n_jobs=1,
        )
