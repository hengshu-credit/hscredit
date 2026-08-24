"""概率校准器基础契约、输入校验与校准指标。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ....utils.serialization import ArtifactSerializableMixin

if TYPE_CHECKING:
    import matplotlib


class BaseCalibrator(ArtifactSerializableMixin, BaseEstimator, ABC):
    """概率校准算法的统一基类。

    **参数**

    :param n_bins: 校准指标与可靠性曲线分箱数，必须为正整数。
    :param strategy: ``"uniform"`` 等宽分箱或 ``"quantile"`` 等频分箱。
    """

    artifact_kind = "概率校准器"

    def __init__(self, n_bins: int = 10, strategy: str = "uniform"):
        self.n_bins = n_bins
        self.strategy = strategy
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """校验可被 ``set_params`` 修改的分箱配置。"""
        if not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool) or self.n_bins < 1:
            raise ValueError("n_bins必须是大于等于1的整数")
        if self.strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy必须是'uniform'或'quantile'")

    def _bin_boundaries(self, y_prob: np.ndarray) -> np.ndarray:
        """按当前策略生成覆盖完整概率区间的去重分箱边界。"""
        self._validate_configuration()
        if self.strategy == "quantile":
            boundaries = np.quantile(y_prob, np.linspace(0, 1, self.n_bins + 1))
            boundaries = np.concatenate(([0.0], boundaries, [1.0]))
        else:
            boundaries = np.linspace(0, 1, self.n_bins + 1)
        return np.unique(np.clip(boundaries, 0.0, 1.0))

    def _iter_bin_masks(self, y_prob: np.ndarray):
        """按当前边界依次生成互斥且覆盖端点的分箱掩码。"""
        boundaries = self._bin_boundaries(y_prob)
        for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if index == 0:
                yield (y_prob >= lower) & (y_prob <= upper)
            else:
                yield (y_prob > lower) & (y_prob <= upper)

    @staticmethod
    def _validate_probabilities(y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """返回一维、非空、有限且位于 ``[0, 1]`` 的概率数组。"""
        values = np.asarray(y_prob, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("概率必须是一维非空数组")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("概率必须是[0, 1]范围内的有限数")
        return values

    @classmethod
    def _validate_fit_data(
        cls,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
        require_both_classes: bool = False,
    ):
        """校验校准训练概率与 0/1 标签并返回 NumPy 数组。"""
        probabilities = cls._validate_probabilities(y_prob)
        labels = np.asarray(y_true)
        if labels.ndim != 1 or labels.shape[0] != probabilities.shape[0]:
            raise ValueError("y_true与概率必须是一维等长数组")
        classes = np.unique(labels)
        if not set(classes).issubset({0, 1}):
            raise ValueError("校准器仅支持0/1标签")
        if require_both_classes and classes.size != 2:
            raise ValueError("校准器拟合数据必须同时包含0和1标签")
        return probabilities, labels

    @abstractmethod
    def fit(self, y_prob, y_true):
        """使用原始正类概率和 0/1 标签拟合校准映射。"""

    @abstractmethod
    def calibrate(self, y_prob):
        """把一维原始正类概率映射为一维校准概率。"""

    def transform(self, y_prob):
        """按 sklearn Transformer 风格返回一维校准概率。"""
        return self.calibrate(y_prob)

    def predict_proba(self, y_prob):
        """返回 ``[P(0), P(1)]`` 两列校准概率。"""
        positive = np.clip(self.calibrate(y_prob), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def compute_brier_score(self, y_true, y_prob) -> float:
        """计算越小越好的 Brier 分数。"""
        probabilities, labels = self._validate_fit_data(y_prob, y_true)
        return float(np.mean((labels - probabilities) ** 2))

    def compute_calibration_metrics(self, y_true, y_prob) -> Dict[str, float]:
        """计算 Brier、ECE、MCE 和样本数。"""
        self._validate_configuration()
        probabilities, labels = self._validate_fit_data(y_prob, y_true)
        errors = []
        weights = []
        for in_bin in self._iter_bin_masks(probabilities):
            proportion = float(in_bin.mean())
            if proportion > 0:
                errors.append(abs(float(probabilities[in_bin].mean()) - float(labels[in_bin].mean())))
                weights.append(proportion)
        return {
            "brier_score": self.compute_brier_score(labels, probabilities),
            "expected_calibration_error": float(np.dot(errors, weights)) if errors else 0.0,
            "max_calibration_error": max(errors, default=0.0),
            "n_samples": len(labels),
        }

    def plot_reliability_diagram(
        self,
        y_true,
        y_prob,
        y_prob_calibrated=None,
        figsize=(10, 8),
        title=None,
        show=True,
        colors=None,
    ) -> "matplotlib.figure.Figure":
        """绘制校准可靠性、概率分布、指标和概率变换四联图。"""
        from .plots import plot_reliability_diagram

        return plot_reliability_diagram(
            self,
            y_true,
            y_prob,
            y_prob_calibrated=y_prob_calibrated,
            figsize=figsize,
            title=title,
            show=show,
            colors=colors,
        )
