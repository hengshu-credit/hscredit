"""模型概率校准模块.

提供多种概率校准方法，用于改善分类模型预测概率的准确性。

**核心功能**

- Platt Scaling: 使用Sigmoid函数进行参数化校准
- Isotonic Regression: 使用保序回归进行非参数校准
- Beta Calibration: 使用Beta分布进行校准
- Histogram Binning: 直方图分箱校准
- 可靠性曲线(Reliability Diagram)绘制
- Brier分数计算

**概率校准原理**

概率校准解决的是模型预测概率与真实概率不一致的问题。例如：
- 模型预测某类样本的违约概率为0.8
- 如果模型是良好校准的，那么在这类样本中实际应该有80%违约
- 如果实际只有60%违约，说明模型过于自信，需要校准

**常见校准方法对比**

| 方法 | 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| Platt Scaling | 参数化 | 简单、鲁棒、不易过拟合 | 只能拟合Sigmoid形状 | 一般情况首选 |
| Isotonic Regression | 非参数化 | 更灵活、拟合能力强 | 容易过拟合、需要更多数据 | 数据量充足时 |
| Beta Calibration | 参数化 | 适合概率已接近0/1的模型 | 计算复杂 | 特殊场景 |
| Histogram Binning | 非参数化 | 简单直观 | 需要大量数据、边界不稳定 | 大数据集 |

**依赖**
- scipy
- sklearn

**引用**

- Platt Scaling：Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines.*
  Advances in Large Margin Classifiers.
- Isotonic Regression 校准：Zadrozny, B., & Elkan, C. (2002). *Transforming Classifier
  Scores into Accurate Multiclass Probability Estimates.* KDD 2002.
- Beta Calibration：Kull, M., Silva Filho, T., & Flach, P. (2017). *Beta calibration.*
  AISTATS 2017. https://arxiv.org/abs/1710.08628
- Brier 分数：Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of
  Probability.* Monthly Weather Review.
- sklearn 校准指南：https://scikit-learn.org/stable/modules/calibration.html

**参考样例**
>>> from hscredit.core.models import XGBoost
>>> from hscredit.core.models.evaluation import ProbabilityCalibrator
>>>
>>> # 训练基础模型
>>> model = XGBoost()
>>> model.fit(X_train, y_train)
>>>
>>> # 创建校准器
>>> calibrator = ProbabilityCalibrator(method='isotonic', model=model, calib_ratio=None)
>>>
>>> # 拟合校准器
>>> calibrator.fit(X_calib, y_calib)
>>>
>>> # 预测校准后的概率
>>> proba_calibrated = calibrator.predict_proba(X_test)
>>>
>>> # 绘制可靠性曲线
>>> calibrator.plot_reliability_diagram(X_test, y_test)
"""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from ....utils.serialization import ArtifactSerializableMixin

if TYPE_CHECKING:
    import matplotlib


class BaseCalibrator(ArtifactSerializableMixin, BaseEstimator, ABC):
    """概率校准器基类.

    所有概率校准器的抽象基类，定义统一接口。

    **参数**

    :param n_bins: 可靠性曲线的分箱数，默认10
    :param strategy: 分箱策略，默认'uniform'
        - 'uniform': 等宽分箱
        - 'quantile': 等频分箱
    """

    artifact_kind = "概率校准器"

    def __init__(self, n_bins: int = 10, strategy: str = "uniform"):
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError("n_bins必须是大于等于1的整数")
        if strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy必须是'uniform'或'quantile'")
        self.n_bins = n_bins
        self.strategy = strategy

    def _bin_boundaries(self, y_prob: np.ndarray) -> np.ndarray:
        """按配置生成覆盖完整概率区间的去重分箱边界。"""
        if self.strategy == "quantile":
            boundaries = np.quantile(y_prob, np.linspace(0, 1, self.n_bins + 1))
            boundaries = np.concatenate(([0.0], boundaries, [1.0]))
        else:
            boundaries = np.linspace(0, 1, self.n_bins + 1)
        return np.unique(np.clip(boundaries, 0.0, 1.0))

    def _iter_bin_masks(self, y_prob: np.ndarray):
        boundaries = self._bin_boundaries(y_prob)
        for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if index == 0:
                mask = (y_prob >= lower) & (y_prob <= upper)
            else:
                mask = (y_prob > lower) & (y_prob <= upper)
            yield mask

    @staticmethod
    def _validate_probabilities(y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校验并返回一维有限概率。"""
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """校验校准训练概率与0/1标签。"""
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
    def fit(
        self,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "BaseCalibrator":
        """拟合校准器.

        :param y_prob: 原始预测概率（正类概率）
        :param y_true: 真实标签
        :return: self
        """
        pass

    @abstractmethod
    def calibrate(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校准概率.

        :param y_prob: 原始预测概率
        :return: 校准后的概率
        """
        pass

    def transform(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """按 sklearn Transformer 约定校准正类概率."""
        return self.calibrate(y_prob)

    def predict_proba(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """返回两列格式的校准概率 ``[P(0), P(1)]``."""
        positive = np.clip(self.calibrate(y_prob), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = "target",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备数据，支持两种传参风格.

        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名
        :return: (X, y) 处理后的数据
        """
        # scorecardpipeline风格：从X中提取target
        if y is None:
            if isinstance(X, pd.DataFrame) and target in X.columns:
                y = X[target].values
                X = X.drop(columns=[target]).values
            else:
                raise ValueError(f"y为None时，X必须是包含'{target}'列的DataFrame")
        else:
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values

        return X, y

    def compute_brier_score(self, y_true: Union[np.ndarray, pd.Series], y_prob: Union[np.ndarray, pd.Series]) -> float:
        """计算Brier分数.

        Brier分数是衡量概率校准的指标，范围[0, 1]，越小越好。
        Brier = mean((y_true - y_prob)^2)

        :param y_true: 真实标签
        :param y_prob: 预测概率
        :return: Brier分数
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true)
        return np.mean((y_true - y_prob) ** 2)

    def compute_calibration_metrics(
        self, y_true: Union[np.ndarray, pd.Series], y_prob: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """计算校准相关指标.

        :param y_true: 真实标签
        :param y_prob: 预测概率
        :return: 包含校准指标的字典
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true)

        # Brier分数
        brier_score = self.compute_brier_score(y_true, y_prob)

        # 可靠性曲线的平均绝对误差(MAE)
        mae = 0.0
        bin_masks = list(self._iter_bin_masks(y_prob))
        for in_bin in bin_masks:
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = y_prob[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                mae += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

        # Expected Calibration Error (ECE)
        ece = mae

        # Maximum Calibration Error (MCE)
        mce = 0.0
        for in_bin in bin_masks:
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = y_prob[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence - avg_accuracy))

        return {
            "brier_score": brier_score,
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
            "n_samples": len(y_true),
        }

    def plot_reliability_diagram(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        y_prob_calibrated: Optional[Union[np.ndarray, pd.Series]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True,
        colors: Optional[List[str]] = None,
    ) -> "matplotlib.figure.Figure":
        """绘制可靠性曲线.

        可靠性曲线显示预测概率与实际频率之间的关系。
        完美校准的模型应该在对角线上。

        :param y_true: 真实标签
        :param y_prob: 原始预测概率
        :param y_prob_calibrated: 校准后的概率，可选
        :param figsize: 图表大小，默认(10, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
        :return: matplotlib Figure对象
        """
        import matplotlib.pyplot as plt

        # hscredit默认配色
        if colors is None:
            colors = ["#2639E9", "#F76E6C", "#FE7715"]

        y_prob, y_true = self._validate_fit_data(y_prob, y_true)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 辅助函数：设置坐标轴样式
        def _setup_axis_style(ax, color="#2639E9"):
            ax.spines["top"].set_color(color)
            ax.spines["bottom"].set_color(color)
            ax.spines["right"].set_color(color)
            ax.spines["left"].set_color(color)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 1. 可靠性曲线
        ax1 = axes[0, 0]
        self._plot_reliability_curve(ax1, y_true, y_prob, "Original", colors[0])
        if y_prob_calibrated is not None:
            y_prob_calibrated = np.asarray(y_prob_calibrated)
            self._plot_reliability_curve(ax1, y_true, y_prob_calibrated, "Calibrated", colors[1])
        ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.5)
        ax1.set_xlabel("Mean Predicted Probability", fontweight="bold")
        ax1.set_ylabel("Fraction of Positives", fontweight="bold")
        ax1.set_title("Reliability Diagram", fontweight="bold")
        ax1.legend(loc="lower right", frameon=False)
        ax1.grid(True, alpha=0.3, linestyle="--")
        _setup_axis_style(ax1)

        # 2. 概率分布直方图
        ax2 = axes[0, 1]
        ax2.hist(
            y_prob, bins=self.n_bins, range=(0, 1), alpha=0.6, color=colors[0], label="Original", edgecolor="white"
        )
        if y_prob_calibrated is not None:
            ax2.hist(
                y_prob_calibrated,
                bins=self.n_bins,
                range=(0, 1),
                alpha=0.6,
                color=colors[1],
                label="Calibrated",
                edgecolor="white",
            )
        ax2.set_xlabel("Predicted Probability", fontweight="bold")
        ax2.set_ylabel("Count", fontweight="bold")
        ax2.set_title("Probability Distribution", fontweight="bold")
        ax2.legend(frameon=False)
        ax2.grid(True, alpha=0.3, linestyle="--")
        _setup_axis_style(ax2)

        # 3. 校准前后对比
        ax3 = axes[1, 0]
        metrics_orig = self.compute_calibration_metrics(y_true, y_prob)
        if y_prob_calibrated is not None:
            metrics_calib = self.compute_calibration_metrics(y_true, y_prob_calibrated)
            metrics_names = ["Brier Score", "ECE", "MCE"]
            x = np.arange(len(metrics_names))
            width = 0.35
            ax3.bar(
                x - width / 2,
                [
                    metrics_orig["brier_score"],
                    metrics_orig["expected_calibration_error"],
                    metrics_orig["max_calibration_error"],
                ],
                width,
                label="Original",
                color=colors[0],
                alpha=0.8,
            )
            ax3.bar(
                x + width / 2,
                [
                    metrics_calib["brier_score"],
                    metrics_calib["expected_calibration_error"],
                    metrics_calib["max_calibration_error"],
                ],
                width,
                label="Calibrated",
                color=colors[1],
                alpha=0.8,
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics_names)
            ax3.set_ylabel("Score", fontweight="bold")
            ax3.set_title("Calibration Metrics Comparison", fontweight="bold")
            ax3.legend(frameon=False)
            ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
        else:
            ax3.text(0.5, 0.5, "No Calibrated Data", ha="center", va="center")
            ax3.set_title("Calibration Metrics Comparison", fontweight="bold")
        _setup_axis_style(ax3)

        # 4. 预测概率变化散点图
        ax4 = axes[1, 1]
        if y_prob_calibrated is not None:
            ax4.scatter(y_prob, y_prob_calibrated, alpha=0.4, color=colors[0], s=20)
            ax4.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax4.set_xlabel("Original Probability", fontweight="bold")
            ax4.set_ylabel("Calibrated Probability", fontweight="bold")
            ax4.set_title("Probability Transformation", fontweight="bold")
            ax4.grid(True, alpha=0.3, linestyle="--")
        else:
            ax4.text(0.5, 0.5, "No Calibrated Data", ha="center", va="center")
            ax4.set_title("Probability Transformation", fontweight="bold")
        _setup_axis_style(ax4)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def _plot_reliability_curve(self, ax, y_true: np.ndarray, y_prob: np.ndarray, label: str, color: str):
        """绘制单条可靠性曲线."""
        bin_centers = []
        bin_accuracies = []

        for in_bin in self._iter_bin_masks(y_prob):
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = y_prob[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                bin_centers.append(avg_confidence)
                bin_accuracies.append(avg_accuracy)

        ax.plot(bin_centers, bin_accuracies, "s-", color=color, label=label)


class PlattCalibrator(BaseCalibrator):
    """Platt Scaling 概率校准器.

    在原始概率的 log-odds 上拟合 Sigmoid：
    ``p_calibrated = sigmoid(A * logit(p_original) + B)``。

    这是最常见的校准方法，适用于大多数情况。

    **参数**

    :param n_bins: 可靠性曲线的分箱数，默认10
    :param strategy: 分箱策略，默认'uniform'
    :param C: 逻辑回归正则化强度，默认1.0

    **参考样例**

    >>> calibrator = PlattCalibrator()
    >>> calibrator.fit(y_prob, y_true)
    >>> proba_calib = calibrator.calibrate(y_prob_test)
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform", C: float = 1.0):
        super().__init__(n_bins=n_bins, strategy=strategy)
        self.C = C

    def fit(
        self,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "PlattCalibrator":
        """拟合Platt Scaling参数.

        :param y_true: 真实标签
        :param y_prob: 原始预测概率
        :return: self
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true, require_both_classes=True)

        # 确保概率在(0, 1)范围内，避免log(0)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

        # Platt Scaling 在线性预测分数（此处为概率的 log-odds）上拟合 sigmoid。
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000)
        self.lr_.fit(logit(y_prob).reshape(-1, 1), y_true)

        return self

    def calibrate(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校准概率.

        :param y_prob: 原始预测概率
        :return: 校准后的概率
        """
        check_is_fitted(self, "lr_")
        y_prob = self._validate_probabilities(y_prob)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

        return self.lr_.predict_proba(logit(y_prob).reshape(-1, 1))[:, 1]


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic Regression 概率校准器.

    使用保序回归进行非参数概率校准。
    比Platt Scaling更灵活，但需要更多数据，容易过拟合。

    **参数**

    :param n_bins: 可靠性曲线的分箱数，默认10
    :param strategy: 分箱策略，默认'uniform'
    :param out_of_bounds: 边界外处理方式，默认'clip'
        - 'clip': 裁剪到拟合范围
        - 'nan': 返回NaN

    **参考样例**

    >>> calibrator = IsotonicCalibrator()
    >>> calibrator.fit(y_prob, y_true)
    >>> proba_calib = calibrator.calibrate(y_prob_test)
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform", out_of_bounds: str = "clip"):
        super().__init__(n_bins=n_bins, strategy=strategy)
        self.out_of_bounds = out_of_bounds

    def fit(
        self,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "IsotonicCalibrator":
        """拟合保序回归.

        :param y_true: 真实标签
        :param y_prob: 原始预测概率
        :return: self
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true, require_both_classes=True)

        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
        self.iso_.fit(y_prob, y_true)

        return self

    def calibrate(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校准概率.

        :param y_prob: 原始预测概率
        :return: 校准后的概率
        """
        check_is_fitted(self, "iso_")
        y_prob = self._validate_probabilities(y_prob)
        return self.iso_.predict(y_prob)


class BetaCalibrator(BaseCalibrator):
    """Beta Calibration 概率校准器.

    使用标准 Beta Calibration 特征 ``log(p)`` 与 ``-log(1-p)`` 拟合
    逻辑回归，适合处理已经接近0或1的概率。

    **参数**

    :param n_bins: 可靠性曲线的分箱数，默认10
    :param strategy: 分箱策略，默认'uniform'

    **参考样例**

    >>> calibrator = BetaCalibrator()
    >>> calibrator.fit(y_prob, y_true)
    >>> proba_calib = calibrator.calibrate(y_prob_test)
    """

    def fit(
        self,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "BetaCalibrator":
        """拟合Beta Calibration参数.

        :param y_true: 真实标签
        :param y_prob: 原始预测概率
        :return: self
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true, require_both_classes=True)

        # 确保概率在(0, 1)范围内
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

        features = np.column_stack([np.log(y_prob), -np.log1p(-y_prob)])
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000)
        self.lr_.fit(features, y_true)

        return self

    def calibrate(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校准概率.

        :param y_prob: 原始预测概率
        :return: 校准后的概率
        """
        check_is_fitted(self, "lr_")
        y_prob = self._validate_probabilities(y_prob)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

        features = np.column_stack([np.log(y_prob), -np.log1p(-y_prob)])
        return self.lr_.predict_proba(features)[:, 1]


class HistogramCalibrator(BaseCalibrator):
    """直方图分箱概率校准器.

    将概率分成若干区间，在每个区间内用真实频率替换预测概率。

    **参数**

    :param n_bins: 分箱数，默认10
    :param strategy: 分箱策略，默认'quantile'
        - 'uniform': 等宽分箱
        - 'quantile': 等频分箱

    **参考样例**

    >>> calibrator = HistogramCalibrator(n_bins=10)
    >>> calibrator.fit(y_prob, y_true)
    >>> proba_calib = calibrator.calibrate(y_prob_test)
    """

    def __init__(self, n_bins: int = 10, strategy: str = "quantile"):
        super().__init__(n_bins=n_bins, strategy=strategy)

    def fit(
        self,
        y_prob: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "HistogramCalibrator":
        """拟合直方图校准器.

        :param y_true: 真实标签
        :param y_prob: 原始预测概率
        :return: self
        """
        y_prob, y_true = self._validate_fit_data(y_prob, y_true)

        # 创建分箱边界
        if self.strategy == "quantile":
            # 等频分箱
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges_ = np.percentile(y_prob, quantiles)
            self.bin_edges_[0] = 0.0
            self.bin_edges_[-1] = 1.0
        else:
            # 等宽分箱
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)

        # 计算每个箱的真实频率
        self.bin_freqs_ = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                in_bin = (y_prob >= self.bin_edges_[i]) & (y_prob <= self.bin_edges_[i + 1])
            else:
                in_bin = (y_prob >= self.bin_edges_[i]) & (y_prob < self.bin_edges_[i + 1])

            if in_bin.sum() > 0:
                self.bin_freqs_[i] = y_true[in_bin].mean()
            else:
                # 空箱使用相邻箱的平均
                self.bin_freqs_[i] = (self.bin_edges_[i] + self.bin_edges_[i + 1]) / 2

        return self

    def calibrate(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """校准概率.

        :param y_prob: 原始预测概率
        :return: 校准后的概率
        """
        check_is_fitted(self, ["bin_edges_", "bin_freqs_"])
        y_prob = self._validate_probabilities(y_prob)

        # 找到每个概率所属的箱
        bin_indices = np.digitize(y_prob, self.bin_edges_[1:-1])

        # 用箱的真实频率替换
        return self.bin_freqs_[np.clip(bin_indices, 0, self.n_bins - 1)]


class ProbabilityCalibrator(ArtifactSerializableMixin, BaseEstimator, ClassifierMixin):
    """概率校准器 - 统一入口.

    提供统一的概率校准接口，支持多种校准方法。

    **参数**

    :param method: 校准方法，默认'platt'
        - 'platt': Platt Scaling (Sigmoid校准)
        - 'isotonic': 保序回归校准
        - 'beta': Beta分布校准
        - 'histogram': 直方图分箱校准
    :param calib_ratio: 用于校准的数据比例，默认0.2
        - 从训练集中划分出calib_ratio的数据用于校准
        - 如果为None，需要使用独立的校准集
    :param n_bins: 可靠性曲线的分箱数，默认10
    :param random_state: 随机种子，默认None
    :param target: 目标列名，默认'target'
        - 用于从DataFrame中提取目标变量
    :param **kwargs: 传递给具体校准器的参数

    **属性**

    :ivar calibrator_: 底层校准器实例
    :ivar is_fitted_: 是否已拟合
    :ivar calib_metrics_: 校准前后的指标对比

    **参考样例**

    >>> # 方式1：使用独立校准集
    >>> calibrator = ProbabilityCalibrator(method='isotonic', model=model, calib_ratio=None)
    >>> calibrator.fit(X_calib, y_calib)
    >>> proba_calib = calibrator.predict_proba(X_test)

    >>> # 方式2：自动划分校准集
    >>> calibrator = ProbabilityCalibrator(method='platt', model=model, calib_ratio=0.2)
    >>> calibrator.fit(X_train, y_train)  # 自动划分20%用于校准
    >>> proba_calib = calibrator.predict_proba(X_test)

    >>> # 方式3：scorecardpipeline风格
    >>> calibrator = ProbabilityCalibrator(method='platt', model=model, calib_ratio=None)
    >>> calibrator.fit(df_calib)  # df_calib包含target列
    >>> proba_calib = calibrator.predict_proba(df_test)

    >>> # 评估校准效果
    >>> calibrator.plot_reliability_diagram(X_test, y_test)
    >>> metrics = calibrator.get_calibration_metrics()
    """

    CALIB_METHODS = {
        "platt": PlattCalibrator,
        "sigmoid": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
        "beta": BetaCalibrator,
        "histogram": HistogramCalibrator,
    }
    artifact_kind = "概率校准模型"

    def __init__(
        self,
        method: str = "platt",
        calib_ratio: Optional[float] = 0.2,
        n_bins: int = 10,
        random_state: Optional[int] = None,
        target: str = "target",
        model: Optional[Any] = None,
        calibrator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if method not in self.CALIB_METHODS:
            raise ValueError(f"不支持的校准方法: {method}，可选: {list(self.CALIB_METHODS.keys())}")

        self.method = method
        self.calib_ratio = calib_ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.target = target
        self.model = model
        self.calibrator_params = calibrator_params
        self.kwargs = {**(calibrator_params or {}), **kwargs}

        # 创建校准器
        calib_class = self.CALIB_METHODS[method]
        self.calibrator_ = calib_class(n_bins=n_bins, **self.kwargs)

        # 属性
        self.model_ = None
        self.is_fitted_ = False
        self.calib_metrics_ = {}
        self.target_ = target

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        model=None,
        target: Optional[str] = None,
        **fit_params,
    ) -> "ProbabilityCalibrator":
        """拟合校准器.

        支持两种传参风格：

        **sklearn风格**::

            calibrator = ProbabilityCalibrator(model=model, calib_ratio=None)
            calibrator.fit(X_calib, y_calib)

        **scorecardpipeline风格**::

            calibrator = ProbabilityCalibrator(model=model, calib_ratio=None)
            calibrator.fit(df_calib)  # df_calib包含target列

        :param model: 已训练的基础模型
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名，默认使用初始化时设置的target
        :param fit_params: 其他参数
        :return: self
        """
        # 兼容旧调用 ``fit(model, X, y)``。在该调用中，参数会映射为
        # X=模型、y=特征、model=标签。
        if hasattr(X, "predict_proba"):
            legacy_model = X
            legacy_X = y
            legacy_y = model
            X, y, model = legacy_X, legacy_y, legacy_model

        model = model if model is not None else self.model
        if model is None or not hasattr(model, "predict_proba"):
            raise ValueError("必须提供实现 predict_proba 方法的基础模型")

        # 使用初始化时设置的target
        target = target or self.target
        self.target_ = target

        # 处理两种传参风格
        X, y = self._prepare_data(X, y, target)

        self.classes_ = np.asarray(getattr(model, "classes_", np.unique(y)))
        if len(self.classes_) != 2:
            raise ValueError("概率校准目前仅支持二分类模型")
        self.positive_class_ = self.classes_[1]

        # 如果需要划分校准集
        if self.calib_ratio is not None and self.calib_ratio > 0:
            if not 0 < self.calib_ratio < 1:
                raise ValueError("calib_ratio必须在(0, 1)范围内，或设为None使用独立校准集")
            indices = np.arange(len(y))
            train_indices, calib_indices = train_test_split(
                indices, test_size=self.calib_ratio, random_state=self.random_state, stratify=y
            )

            def take(values, idx):
                return values.iloc[idx] if hasattr(values, "iloc") else np.asarray(values)[idx]

            X_train, y_train = take(X, train_indices), take(y, train_indices)
            X, y = take(X, calib_indices), take(y, calib_indices)
            try:
                fitted_model = clone(model)
            except Exception:
                fitted_model = copy.deepcopy(model)

            model_fit_params = {}
            for name, value in fit_params.items():
                if hasattr(value, "__len__") and len(value) == len(indices):
                    model_fit_params[name] = take(value, train_indices)
                else:
                    model_fit_params[name] = value
            fitted_model.fit(X_train, y_train, **model_fit_params)
            self.model_ = fitted_model
        else:
            self.model_ = model

        y_binary = (np.asarray(y) == self.positive_class_).astype(int)

        # 获取模型预测概率
        y_prob = self._get_model_proba(X)

        # 拟合校准器
        self.calibrator_.fit(y_prob, y_binary)

        # 计算校准前后的指标
        self.calib_metrics_["original"] = self.calibrator_.compute_calibration_metrics(y_binary, y_prob)
        y_prob_calib = self.calibrator_.calibrate(y_prob)
        self.calib_metrics_["calibrated"] = self.calibrator_.compute_calibration_metrics(y_binary, y_prob_calib)

        self.is_fitted_ = True
        if hasattr(X, "shape") and len(X.shape) == 2:
            self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测校准后的概率.

        :param X: 特征矩阵
        :return: 两列概率数组，shape ``(n_samples, 2)``
        """
        check_is_fitted(self, "classes_")

        # 获取原始概率
        y_prob = self._get_model_proba(X)

        # 校准
        positive = np.clip(self.calibrator_.calibrate(y_prob), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """预测类别标签.

        :param X: 特征矩阵
        :param threshold: 分类阈值，默认0.5
        :return: 预测类别
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold必须在[0, 1]范围内")
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= threshold, self.classes_[1], self.classes_[0])

    def _get_model_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """获取模型预测概率."""
        if hasattr(self.model_, "predict_proba"):
            proba = np.asarray(self.model_.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] == 2:
                classes = np.asarray(getattr(self.model_, "classes_", self.classes_))
                matches = np.flatnonzero(classes == self.positive_class_)
                if len(matches) != 1:
                    raise ValueError(f"基础模型概率列中不存在正类标签: {self.positive_class_!r}")
                positive_index = int(matches[0])
                return proba[:, positive_index]
            return proba
        else:
            raise ValueError("模型必须实现predict_proba方法")

    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]], target: str):
        """准备数据，支持两种传参风格."""
        # scorecardpipeline风格：从X中提取target
        if y is None:
            if isinstance(X, pd.DataFrame) and target in X.columns:
                y = X[target].to_numpy()
                X = X.drop(columns=[target])
            else:
                raise ValueError(f"y为None时，X必须是包含'{target}'列的DataFrame")
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        else:
            y = np.asarray(y)

        return X, y

    def get_calibration_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取校准前后的指标对比.

        :return: 包含校准前后指标的字典
        """
        check_is_fitted(self, "classes_")
        return self.calib_metrics_

    def calibration_report(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: Optional[str] = None,
    ) -> pd.DataFrame:
        """生成校准前后指标对比报告.

        :return: 包含指标、校准前、校准后、改善值和改善率的中文 DataFrame
        """
        check_is_fitted(self, "classes_")
        X, y = self._prepare_data(X, y, target or self.target_)
        y_binary = (np.asarray(y) == self.positive_class_).astype(int)
        original = self.calibrator_.compute_calibration_metrics(
            y_binary,
            self._get_model_proba(X),
        )
        calibrated = self.calibrator_.compute_calibration_metrics(
            y_binary,
            self.predict_proba(X)[:, 1],
        )
        labels = {
            "brier_score": "Brier分数",
            "expected_calibration_error": "期望校准误差(ECE)",
            "max_calibration_error": "最大校准误差(MCE)",
        }
        rows = []
        for key, label in labels.items():
            before = float(original[key])
            after = float(calibrated[key])
            improvement = before - after
            rows.append(
                {
                    "指标": label,
                    "校准前": before,
                    "校准后": after,
                    "改善值": improvement,
                    "改善率": improvement / before if before != 0 else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def report(self, X, y=None, target: Optional[str] = None) -> pd.DataFrame:
        """``calibration_report`` 的统一报告入口."""
        return self.calibration_report(X, y=y, target=target)

    def plot_reliability_diagram(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = "target",
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True,
    ) -> "matplotlib.figure.Figure":
        """绘制可靠性曲线.

        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名（scorecardpipeline风格使用）
        :param figsize: 图表大小，默认(10, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :return: matplotlib Figure对象
        """
        check_is_fitted(self, "classes_")

        # 处理两种传参风格
        X, y = self._prepare_data(X, y, target)
        y_binary = (np.asarray(y) == self.positive_class_).astype(int)

        # 获取原始和校准后的概率
        y_prob_orig = self._get_model_proba(X)
        y_prob_calib = self.predict_proba(X)[:, 1]

        return self.calibrator_.plot_reliability_diagram(
            y_binary, y_prob_orig, y_prob_calib, figsize=figsize, title=title, show=show
        )

    def calibrate_proba(self, y_prob: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """直接校准概率（不通过模型）.

        :param y_prob: 原始概率
        :return: 校准后的概率
        """
        check_is_fitted(self, "classes_")
        return self.calibrator_.calibrate(y_prob)


class CalibratedModel(ArtifactSerializableMixin, BaseEstimator, ClassifierMixin):
    """已校准模型包装器.

    将基础模型和校准器组合在一起，提供统一的预测接口。

    **参数**

    :param base_model: 基础模型
    :param calibrator: 概率校准器实例

    **参考样例**

    >>> from hscredit.core.models import XGBoost
    >>> from hscredit.core.models.evaluation import ProbabilityCalibrator, CalibratedModel
    >>>
    >>> # 训练基础模型
    >>> model = XGBoost()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # 创建校准器并拟合
    >>> calibrator = ProbabilityCalibrator(method='platt', model=model, calib_ratio=None)
    >>> calibrator.fit(X_calib, y_calib)
    >>>
    >>> # 包装为已校准模型
    >>> calibrated_model = CalibratedModel(model, calibrator)
    >>> proba = calibrated_model.predict_proba(X_test)
    """

    artifact_kind = "概率校准模型"

    def __init__(self, base_model, calibrator: ProbabilityCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(base_model, "classes_", [0, 1]))

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测校准后的正类（坏样本）概率。

        将基础模型输出的原始概率经已拟合的校准器映射为更准确的概率。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :return: 两列校准概率数组，shape ``(n_samples, 2)``
        """
        raw = np.asarray(self.base_model.predict_proba(X))
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("基础模型必须返回两列二分类概率")
        base_classes = np.asarray(getattr(self.base_model, "classes_", self.classes_))
        positive_matches = np.flatnonzero(base_classes == self.classes_[1])
        if len(positive_matches) != 1:
            raise ValueError(f"基础模型概率列中不存在正类标签: {self.classes_[1]!r}")
        positive = self.calibrator.calibrator_.calibrate(raw[:, int(positive_matches[0])])
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """基于校准后概率预测类别标签。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :param threshold: 判正阈值，校准概率 ``>= threshold`` 记为 1，默认为 ``0.5``
        :return: 0/1 类别数组
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold必须在[0, 1]范围内")
        positive = self.predict_proba(X)[:, 1]
        return np.where(positive >= threshold, self.classes_[1], self.classes_[0])

    def predict_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """将校准后概率线性映射为 0–1000 的风险评分（概率越低分越高）。

        采用 ``score = (1 - p) * 1000`` 的简易映射；若需标准 log-odds 评分卡刻度，
        请改用 :class:`~hscredit.core.models.scorecard.ScoreCard` 或
        :class:`~hscredit.core.models.scorecard.score_transformer.StandardScoreTransformer`。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :return: 0–1000 区间的风险评分数组
        """
        proba = self.predict_proba(X)[:, 1]
        return (1 - proba) * 1000

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """评估校准后模型的区分度与校准度。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :param y: 真实标签（0/1）
        :return: 含 ``AUC`` / ``KS``（区分度）与 ``Brier``（校准度，越小越好）的指标字典

        **参考样例**

        >>> calibrated.evaluate(X_test, y_test)
        {'AUC': ..., 'KS': ..., 'Brier': ...}
        """
        from ...metrics.classification import ks, auc

        y_proba = self.predict_proba(X)[:, 1]
        y = np.asarray(y)
        unknown_labels = set(np.unique(y)) - set(self.classes_)
        if unknown_labels:
            raise ValueError(f"y包含模型未见过的标签: {sorted(unknown_labels, key=str)}")
        y_binary = (y == self.classes_[1]).astype(int)

        return {
            "AUC": auc(y_binary, y_proba),
            "KS": ks(y_binary, y_proba),
            "Brier": self.calibrator.calibrator_.compute_brier_score(y_binary, y_proba),
        }


def plot_calibration_comparison(
    y_true: Union[np.ndarray, pd.Series],
    y_prob_dict: Dict[str, Union[np.ndarray, pd.Series]],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    show: bool = True,
    colors: Optional[List[str]] = None,
) -> "matplotlib.figure.Figure":
    """绘制多个模型的校准对比图.

    :param y_true: 真实标签
    :param y_prob_dict: 各模型的预测概率字典，如 {'Model A': proba_a, 'Model B': proba_b}
    :param n_bins: 分箱数，默认10
    :param figsize: 图表大小，默认(12, 5)
    :param title: 图表标题，可选
    :param show: 是否显示图表，默认True
    :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
    :return: matplotlib Figure对象

    **参考样例**

    >>> plot_calibration_comparison(
    ...     y_test,
    ...     {'Original': proba_orig, 'Platt': proba_platt, 'Isotonic': proba_iso},
    ...     title='Calibration Comparison'
    ... )
    """
    import matplotlib.pyplot as plt

    # hscredit默认配色
    if colors is None:
        colors = ["#2639E9", "#F76E6C", "#FE7715", "#2E8B57", "#9370DB", "#FF6347"]

    y_true = np.asarray(y_true)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 辅助函数：设置坐标轴样式
    def _setup_axis_style(ax, color="#2639E9"):
        ax.spines["top"].set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["right"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 1. 可靠性曲线
    ax1 = axes[0]
    color_list = colors[: len(y_prob_dict)]

    for (name, y_prob), color in zip(y_prob_dict.items(), color_list):
        y_prob = np.asarray(y_prob)
        # 使用PlattCalibrator实例来调用方法
        calibrator = PlattCalibrator(n_bins=n_bins)
        calibrator._plot_reliability_curve(ax1, y_true, y_prob, name, color)

    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.5)
    ax1.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax1.set_ylabel("Fraction of Positives", fontweight="bold")
    ax1.set_title("Reliability Diagram", fontweight="bold")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(True, alpha=0.3, linestyle="--")
    _setup_axis_style(ax1)

    # 2. 指标对比
    ax2 = axes[1]
    metrics_names = ["Brier", "ECE"]
    x = np.arange(len(metrics_names))
    width = 0.8 / len(y_prob_dict)

    for i, (name, y_prob) in enumerate(y_prob_dict.items()):
        y_prob = np.asarray(y_prob)
        # 使用PlattCalibrator实例来调用方法
        calibrator = PlattCalibrator(n_bins=n_bins)
        metrics = calibrator.compute_calibration_metrics(y_true, y_prob)
        values = [metrics["brier_score"], metrics["expected_calibration_error"]]
        ax2.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], alpha=0.8)

    ax2.set_xticks(x + width * (len(y_prob_dict) - 1) / 2)
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylabel("Score (lower is better)", fontweight="bold")
    ax2.set_title("Calibration Metrics Comparison", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    _setup_axis_style(ax2)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def calibrate_model(
    model,
    X_calib: Union[np.ndarray, pd.DataFrame],
    y_calib: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "platt",
    target: str = "target",
    **kwargs,
) -> ProbabilityCalibrator:
    """便捷函数：创建并拟合概率校准器.

    :param model: 已训练的基础模型
    :param X_calib: 校准集特征或包含target的DataFrame
    :param y_calib: 校准集标签，可选
    :param method: 校准方法，默认'platt'
    :param target: 目标列名（scorecardpipeline风格使用）
    :param kwargs: 其他参数
    :return: 拟合好的ProbabilityCalibrator

    **参考样例**

    >>> calibrator = calibrate_model(model, X_calib, y_calib, method='isotonic')
    >>> proba_calib = calibrator.predict_proba(X_test)
    """
    kwargs.setdefault("calib_ratio", None)
    calibrator = ProbabilityCalibrator(method=method, model=model, **kwargs)
    calibrator.fit(X_calib, y_calib, target=target)
    return calibrator
