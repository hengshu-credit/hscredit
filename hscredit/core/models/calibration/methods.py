"""Platt、Isotonic、Beta 与 Histogram 概率校准算法。"""

from typing import Union

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from .base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    """在原始概率的 log-odds 上拟合逻辑回归校准映射。"""

    def __init__(self, n_bins: int = 10, strategy: str = "uniform", C: float = 1.0):
        super().__init__(n_bins=n_bins, strategy=strategy)
        self.C = C
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        if hasattr(self, "C") and (not np.isscalar(self.C) or not np.isfinite(self.C) or self.C <= 0):
            raise ValueError("C必须是有限正数")

    def fit(self, y_prob: Union[np.ndarray, pd.Series], y_true: Union[np.ndarray, pd.Series]):
        """拟合 Platt Scaling 并返回自身。"""
        self._validate_configuration()
        probabilities, labels = self._validate_fit_data(y_prob, y_true, require_both_classes=True)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000)
        self.lr_.fit(logit(probabilities).reshape(-1, 1), labels)
        return self

    def calibrate(self, y_prob):
        """返回 Platt 校准后的一维概率。"""
        check_is_fitted(self, "lr_")
        probabilities = np.clip(self._validate_probabilities(y_prob), 1e-15, 1 - 1e-15)
        return self.lr_.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]


class IsotonicCalibrator(BaseCalibrator):
    """使用保序回归拟合非参数概率校准映射。"""

    def __init__(self, n_bins: int = 10, strategy: str = "uniform", out_of_bounds: str = "clip"):
        super().__init__(n_bins=n_bins, strategy=strategy)
        self.out_of_bounds = out_of_bounds
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        if hasattr(self, "out_of_bounds") and self.out_of_bounds not in {"clip", "nan", "raise"}:
            raise ValueError("out_of_bounds必须是'clip'、'nan'或'raise'")

    def fit(self, y_prob, y_true):
        """拟合保序回归并返回自身。"""
        self._validate_configuration()
        probabilities, labels = self._validate_fit_data(y_prob, y_true, require_both_classes=True)
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
        self.iso_.fit(probabilities, labels)
        return self

    def calibrate(self, y_prob):
        """返回保序回归校准后的一维概率。"""
        check_is_fitted(self, "iso_")
        return self.iso_.predict(self._validate_probabilities(y_prob))


class BetaCalibrator(BaseCalibrator):
    """使用 ``log(p)`` 与 ``-log(1-p)`` 特征拟合 Beta 校准。"""

    def fit(self, y_prob, y_true):
        """拟合 Beta 校准逻辑回归并返回自身。"""
        self._validate_configuration()
        probabilities, labels = self._validate_fit_data(y_prob, y_true, require_both_classes=True)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        features = np.column_stack([np.log(probabilities), -np.log1p(-probabilities)])
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000)
        self.lr_.fit(features, labels)
        return self

    def calibrate(self, y_prob):
        """返回 Beta 校准后的一维概率。"""
        check_is_fitted(self, "lr_")
        probabilities = np.clip(self._validate_probabilities(y_prob), 1e-15, 1 - 1e-15)
        features = np.column_stack([np.log(probabilities), -np.log1p(-probabilities)])
        return self.lr_.predict_proba(features)[:, 1]


class HistogramCalibrator(BaseCalibrator):
    """使用箱内真实正类频率进行直方图概率校准。"""

    def __init__(self, n_bins: int = 10, strategy: str = "quantile"):
        super().__init__(n_bins=n_bins, strategy=strategy)

    def fit(self, y_prob, y_true):
        """拟合直方图边界和箱内频率并返回自身。"""
        self._validate_configuration()
        probabilities, labels = self._validate_fit_data(y_prob, y_true)
        if self.strategy == "quantile":
            self.bin_edges_ = np.percentile(probabilities, np.linspace(0, 100, self.n_bins + 1))
            self.bin_edges_[0] = 0.0
            self.bin_edges_[-1] = 1.0
        else:
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        self.bin_freqs_ = np.zeros(self.n_bins)
        for index in range(self.n_bins):
            if index == self.n_bins - 1:
                mask = (probabilities >= self.bin_edges_[index]) & (probabilities <= self.bin_edges_[index + 1])
            else:
                mask = (probabilities >= self.bin_edges_[index]) & (probabilities < self.bin_edges_[index + 1])
            self.bin_freqs_[index] = (
                labels[mask].mean()
                if mask.any()
                else (self.bin_edges_[index] + self.bin_edges_[index + 1]) / 2
            )
        return self

    def calibrate(self, y_prob):
        """返回输入概率所在箱的真实正类频率。"""
        check_is_fitted(self, ["bin_edges_", "bin_freqs_"])
        probabilities = self._validate_probabilities(y_prob)
        indices = np.digitize(probabilities, self.bin_edges_[1:-1])
        return self.bin_freqs_[np.clip(indices, 0, self.n_bins - 1)]
