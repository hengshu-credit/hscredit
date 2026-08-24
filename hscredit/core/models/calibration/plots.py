"""概率校准的中文 Matplotlib 可视化。"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseCalibrator
from .methods import PlattCalibrator


DEFAULT_COLORS = ["#2639E9", "#F76E6C", "#FE7715", "#2E8B57", "#9370DB", "#FF6347"]


def _validated_colors(colors: Optional[List[str]]) -> List[str]:
    """返回非空配色列表。"""
    resolved = list(DEFAULT_COLORS if colors is None else colors)
    if not resolved:
        raise ValueError("colors不能为空")
    return resolved


def _setup_axis_style(ax, color: str = "#2639E9") -> None:
    """应用校准图统一坐标轴样式。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(color)
    ax.spines["left"].set_color(color)


def _plot_reliability_curve(ax, calibrator: BaseCalibrator, y_true, y_prob, label: str, color: str) -> None:
    """绘制一条经完整输入校验的可靠性曲线。"""
    probabilities, labels = calibrator._validate_fit_data(y_prob, y_true)
    predicted = []
    observed = []
    for mask in calibrator._iter_bin_masks(probabilities):
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
    ax.plot(predicted, observed, "s-", color=color, label=label)


def plot_reliability_diagram(
    calibrator: BaseCalibrator,
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    y_prob_calibrated: Optional[Union[np.ndarray, pd.Series]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    show: bool = True,
    colors: Optional[List[str]] = None,
):
    """绘制可靠性曲线、概率分布、校准指标和概率变换四联图。"""
    import matplotlib.pyplot as plt

    palette = _validated_colors(colors)
    probabilities, labels = calibrator._validate_fit_data(y_prob, y_true)
    calibrated = None
    if y_prob_calibrated is not None:
        calibrated, calibrated_labels = calibrator._validate_fit_data(y_prob_calibrated, labels)
        if not np.array_equal(calibrated_labels, labels):
            raise ValueError("校准前后标签不一致")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    reliability_ax, distribution_ax, metrics_ax, transform_ax = axes.ravel()

    _plot_reliability_curve(reliability_ax, calibrator, labels, probabilities, "校准前", palette[0])
    if calibrated is not None:
        _plot_reliability_curve(reliability_ax, calibrator, labels, calibrated, "校准后", palette[1 % len(palette)])
    reliability_ax.plot([0, 1], [0, 1], "k--", label="完美校准", alpha=0.5)
    reliability_ax.set(xlabel="平均预测概率", ylabel="实际正类比例", title="可靠性曲线")
    reliability_ax.legend(loc="lower right", frameon=False)
    reliability_ax.grid(True, alpha=0.3, linestyle="--")

    distribution_ax.hist(probabilities, bins=calibrator.n_bins, range=(0, 1), alpha=0.6, color=palette[0], label="校准前", edgecolor="white")
    if calibrated is not None:
        distribution_ax.hist(calibrated, bins=calibrator.n_bins, range=(0, 1), alpha=0.6, color=palette[1 % len(palette)], label="校准后", edgecolor="white")
    distribution_ax.set(xlabel="预测概率", ylabel="样本数", title="概率分布")
    distribution_ax.legend(frameon=False)
    distribution_ax.grid(True, alpha=0.3, linestyle="--")

    if calibrated is None:
        metrics_ax.text(0.5, 0.5, "未提供校准后概率", ha="center", va="center")
        transform_ax.text(0.5, 0.5, "未提供校准后概率", ha="center", va="center")
    else:
        original_metrics = calibrator.compute_calibration_metrics(labels, probabilities)
        calibrated_metrics = calibrator.compute_calibration_metrics(labels, calibrated)
        metric_names = ["Brier分数", "期望校准误差", "最大校准误差"]
        keys = ["brier_score", "expected_calibration_error", "max_calibration_error"]
        positions = np.arange(len(keys))
        width = 0.35
        metrics_ax.bar(positions - width / 2, [original_metrics[key] for key in keys], width, label="校准前", color=palette[0])
        metrics_ax.bar(positions + width / 2, [calibrated_metrics[key] for key in keys], width, label="校准后", color=palette[1 % len(palette)])
        metrics_ax.set_xticks(positions)
        metrics_ax.set_xticklabels(metric_names)
        metrics_ax.legend(frameon=False)
        metrics_ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        transform_ax.scatter(probabilities, calibrated, alpha=0.4, color=palette[0], s=20)
        transform_ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        transform_ax.set(xlabel="校准前概率", ylabel="校准后概率")
        transform_ax.grid(True, alpha=0.3, linestyle="--")

    metrics_ax.set_title("校准指标比较")
    transform_ax.set_title("概率映射")
    for axis in axes.ravel():
        _setup_axis_style(axis, palette[0])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_calibration_comparison(
    y_true: Union[np.ndarray, pd.Series],
    y_prob_dict: Dict[str, Union[np.ndarray, pd.Series]],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    show: bool = True,
    colors: Optional[List[str]] = None,
):
    """绘制多个模型的可靠性曲线和 Brier/ECE 指标比较。"""
    import matplotlib.pyplot as plt

    if not isinstance(y_prob_dict, dict) or not y_prob_dict:
        raise ValueError("y_prob_dict不能为空")
    palette = _validated_colors(colors)
    calibrator = PlattCalibrator(n_bins=n_bins)
    validated = {
        str(name): calibrator._validate_fit_data(probabilities, y_true)[0]
        for name, probabilities in y_prob_dict.items()
    }
    labels = np.asarray(y_true)

    fig, (reliability_ax, metrics_ax) = plt.subplots(1, 2, figsize=figsize)
    for index, (name, probabilities) in enumerate(validated.items()):
        _plot_reliability_curve(reliability_ax, calibrator, labels, probabilities, name, palette[index % len(palette)])
    reliability_ax.plot([0, 1], [0, 1], "k--", label="完美校准", alpha=0.5)
    reliability_ax.set(xlabel="平均预测概率", ylabel="实际正类比例", title="可靠性曲线")
    reliability_ax.legend(loc="lower right", frameon=False)
    reliability_ax.grid(True, alpha=0.3, linestyle="--")

    metric_names = ["Brier分数", "期望校准误差"]
    positions = np.arange(len(metric_names))
    width = 0.8 / len(validated)
    for index, (name, probabilities) in enumerate(validated.items()):
        metrics = calibrator.compute_calibration_metrics(labels, probabilities)
        metrics_ax.bar(
            positions + index * width,
            [metrics["brier_score"], metrics["expected_calibration_error"]],
            width,
            label=name,
            color=palette[index % len(palette)],
            alpha=0.8,
        )
    metrics_ax.set_xticks(positions + width * (len(validated) - 1) / 2)
    metrics_ax.set_xticklabels(metric_names)
    metrics_ax.set(ylabel="指标值（越低越好）", title="校准指标比较")
    metrics_ax.legend(frameon=False)
    metrics_ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    _setup_axis_style(reliability_ax, palette[0])
    _setup_axis_style(metrics_ax, palette[0])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
