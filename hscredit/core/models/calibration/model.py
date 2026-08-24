"""概率校准模型包装、便捷入口与兼容评估。"""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ....utils.serialization import ArtifactSerializableMixin
from ..base import _evaluate_binary_predictions
from .base import BaseCalibrator
from .methods import BetaCalibrator, HistogramCalibrator, IsotonicCalibrator, PlattCalibrator

if TYPE_CHECKING:
    import matplotlib


def _class_index(classes: np.ndarray, label: Any) -> int:
    """返回二分类标签的唯一概率列位置。"""
    matches = np.flatnonzero(np.asarray(classes) == label)
    if len(matches) != 1:
        raise ValueError(f"基础模型概率列中不存在唯一正类标签: {label!r}")
    return int(matches[0])


def _assemble_binary_probabilities(positive: np.ndarray, classes: np.ndarray, positive_class: Any) -> np.ndarray:
    """按 ``classes_`` 的真实顺序组装二分类概率矩阵。"""
    positive_index = _class_index(classes, positive_class)
    result = np.empty((len(positive), 2), dtype=float)
    result[:, positive_index] = positive
    result[:, 1 - positive_index] = 1.0 - positive
    return result


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
    :param positive_class: 正类标签；None 时使用基础模型 ``classes_[1]``。
    :param calibrator_params: 传递给具体校准算法的可克隆参数字典。

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
        positive_class: Optional[Any] = None,
        calibrator_params: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.calib_ratio = calib_ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.target = target
        self.model = model
        self.positive_class = positive_class
        self.calibrator_params = calibrator_params
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """校验构造参数和经 ``set_params`` 修改后的状态。"""
        if self.method not in self.CALIB_METHODS:
            raise ValueError(f"不支持的校准方法: {self.method}，可选: {list(self.CALIB_METHODS.keys())}")
        if not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool) or self.n_bins < 1:
            raise ValueError("n_bins必须是大于等于1的整数")
        if self.calib_ratio is not None:
            if (
                isinstance(self.calib_ratio, bool)
                or not np.isscalar(self.calib_ratio)
                or not np.isfinite(self.calib_ratio)
                or not 0 < float(self.calib_ratio) < 1
            ):
                raise ValueError("calib_ratio必须在(0, 1)范围内，或设为None使用独立校准集")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target必须是非空字符串")
        if self.calibrator_params is not None and not isinstance(self.calibrator_params, dict):
            raise ValueError("calibrator_params必须是字典或None")

    def _make_calibrator(self) -> BaseCalibrator:
        """根据当前参数创建全新的底层校准算法。"""
        calibrator_class = self.CALIB_METHODS[self.method]
        return calibrator_class(n_bins=self.n_bins, **dict(self.calibrator_params or {}))

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
        self._validate_parameters()

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
        self.positive_class_ = self.classes_[1] if self.positive_class is None else self.positive_class
        if not np.any(self.classes_ == self.positive_class_):
            raise ValueError(f"positive_class={self.positive_class_!r} 不在模型类别 {self.classes_.tolist()!r} 中")

        # 如果需要划分校准集
        if self.calib_ratio is not None:
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
            except (TypeError, RuntimeError):
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

        # 每次 fit 都按当前参数重建底层校准器，避免 set_params 后复用旧算法或拟合状态。
        self.calibrator_ = self._make_calibrator()
        self.calibrator_.fit(y_prob, y_binary)

        # 计算校准前后的指标
        self.calib_metrics_ = {}
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
        return _assemble_binary_probabilities(positive, self.classes_, self.positive_class_)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """预测类别标签.

        :param X: 特征矩阵
        :param threshold: 分类阈值，默认0.5
        :return: 预测类别
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold必须在[0, 1]范围内")
        positive_index = _class_index(self.classes_, self.positive_class_)
        proba = self.predict_proba(X)[:, positive_index]
        negative_class = self.classes_[1 - positive_index]
        return np.where(proba >= threshold, self.positive_class_, negative_class)

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
            self.predict_proba(X)[:, _class_index(self.classes_, self.positive_class_)],
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
        target: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True,
        colors: Optional[List[str]] = None,
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
        X, y = self._prepare_data(X, y, target or self.target_)
        y_binary = (np.asarray(y) == self.positive_class_).astype(int)

        # 获取原始和校准后的概率
        y_prob_orig = self._get_model_proba(X)
        y_prob_calib = self.predict_proba(X)[:, _class_index(self.classes_, self.positive_class_)]

        return self.calibrator_.plot_reliability_diagram(
            y_binary, y_prob_orig, y_prob_calib, figsize=figsize, title=title, show=show, colors=colors
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
    >>> from hscredit.core.models.calibration import ProbabilityCalibrator, CalibratedModel
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
        check_is_fitted(calibrator, ["classes_", "positive_class_", "calibrator_"])
        self.classes_ = np.asarray(getattr(base_model, "classes_", []))
        if self.classes_.shape != (2,):
            raise ValueError("基础模型必须提供两个类别的classes_")
        if not np.any(self.classes_ == calibrator.positive_class_):
            raise ValueError(f"基础模型类别与校准器正类 {calibrator.positive_class_!r} 不兼容")
        self.positive_class_ = calibrator.positive_class_

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
        positive_index = _class_index(base_classes, self.positive_class_)
        positive = np.clip(self.calibrator.calibrate_proba(raw[:, positive_index]), 0.0, 1.0)
        return _assemble_binary_probabilities(positive, self.classes_, self.positive_class_)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """基于校准后概率预测类别标签。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :param threshold: 判正阈值，校准概率 ``>= threshold`` 记为 1，默认为 ``0.5``
        :return: 0/1 类别数组
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold必须在[0, 1]范围内")
        positive_index = _class_index(self.classes_, self.positive_class_)
        positive = self.predict_proba(X)[:, positive_index]
        return np.where(positive >= threshold, self.positive_class_, self.classes_[1 - positive_index])

    def predict_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """将校准后概率线性映射为 0–1000 的风险评分（概率越低分越高）。

        采用 ``score = (1 - p) * 1000`` 的简易映射；若需标准 log-odds 评分卡刻度，
        请改用 :class:`~hscredit.core.models.scorecard.ScoreCard` 或
        :class:`~hscredit.core.models.scorecard.score_transformer.StandardScoreTransformer`。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :return: 0–1000 区间的风险评分数组
        """
        proba = self.predict_proba(X)[:, _class_index(self.classes_, self.positive_class_)]
        return (1 - proba) * 1000

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """评估校准后模型的区分度与校准度。

        :param X: 特征矩阵，DataFrame 或 ndarray
        :param y: 真实标签（0/1）
        :param sample_weight: 样本权重；KS 与 Lift 不支持时发出一次中文警告。
        :param metrics: 指标名列表；None 时返回 AUC、KS 和 Brier。
        :return: 含 ``AUC`` / ``KS`` （区分度）与 ``Brier`` （校准度）的指标字典

        **参考样例**

        >>> calibrated.evaluate(X_test, y_test)
        {'AUC': ..., 'KS': ..., 'Brier': ...}
        """
        probabilities = self.predict_proba(X)
        positive_index = _class_index(self.classes_, self.positive_class_)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(probabilities):
            raise ValueError("y必须是一维且与评估样本等长")
        unknown_labels = set(np.unique(labels)) - set(self.classes_)
        if unknown_labels:
            raise ValueError(f"y包含模型未见过的标签: {sorted(unknown_labels, key=str)}")
        binary_labels = (labels == self.positive_class_).astype(int)
        binary_predictions = (self.predict(X) == self.positive_class_).astype(int)
        return _evaluate_binary_predictions(
            binary_labels,
            probabilities[:, positive_index],
            binary_predictions,
            metrics=["auc", "ks", "brier"] if metrics is None else list(metrics),
            sample_weight=sample_weight,
        )


from .plots import plot_calibration_comparison


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
