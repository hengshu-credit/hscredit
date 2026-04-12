"""
损失函数和评估指标基类

定义统一的接口，确保所有损失函数和评估指标在不同框架间的一致性。
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np


class BaseLoss(ABC):
    """损失函数基类。

    所有自定义损失函数都应该继承此类，并实现以下方法:
    - __call__: 计算损失值
    - gradient: 计算梯度（一阶导数）
    - hessian: 计算二阶导数（可选，用于XGBoost等需要二阶导的框架）

    :param name: 损失函数名称，默认为"custom_loss"
    """
    
    def __init__(self, name: str = "custom_loss"):
        self.name = name
    
    @abstractmethod
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算损失值。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测值（概率或logits）, shape (n_samples,)
        :return: 损失值
        """
        pass
    
    @abstractmethod
    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算梯度（一阶导数）。

        :param y_true: 真实标签
        :param y_pred: 预测值
        :return: 梯度数组, shape (n_samples,)
        """
        pass
    
    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Optional[np.ndarray]:
        """计算二阶导数（可选）。

        某些框架如XGBoost需要二阶导数，如果不需要可以返回None

        :param y_true: 真实标签
        :param y_pred: 预测值
        :return: 二阶导数数组, shape (n_samples,), 或None
        """
        return None
    
    def to_xgboost(self) -> Callable:
        """转换为XGBoost格式的损失函数。

        :return: XGBoost可用的损失函数
        """
        def xgb_loss(preds: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
            labels = dtrain.get_label()
            grad = self.gradient(labels, preds)
            hess = self.hessian(labels, preds)
            if hess is None:
                # 如果没有二阶导，使用近似值
                hess = np.ones_like(grad) * 0.5
            return grad, hess

        return xgb_loss

    def to_lightgbm(self) -> Callable:
        """转换为LightGBM格式的损失函数。

        :return: LightGBM可用的损失函数
        """
        def lgb_loss(y_true: np.ndarray, y_pred: np.ndarray):
            grad = self.gradient(y_true, y_pred)
            hess = self.hessian(y_true, y_pred)
            if hess is None:
                hess = np.ones_like(grad) * 0.5
            return grad, hess

        return lgb_loss

    def to_catboost(self) -> Callable:
        """转换为CatBoost格式的损失函数。

        :return: CatBoost可用的损失函数
        """
        def catboost_loss(approxes, target, weight):
            # CatBoost使用不同的接口
            approx = approxes[0]
            grad = self.gradient(target, approx)
            hess = self.hessian(target, approx)
            if hess is None:
                hess = np.ones_like(grad) * 0.5
            return grad, hess

        return catboost_loss

    def to_ngboost(self):
        """转换为NGBoost格式的Score类（仅支持 Bernoulli 二分类）。

        NGBoost 使用自然梯度 + 概率分布框架，自定义 loss 需要实现 Score 子类。
        本方法通过链式法则将 ``dL/dp``（BaseLoss.gradient 的输出）转换为
        ``dL/d(logit)``（NGBoost 需要的分布参数梯度）::

            dL/d(logit) = dL/dp × dp/d(logit) = dL/dp × p × (1 - p)

        :return: NGBoost Score 子类（未实例化），可直接传给 ``NGBClassifier(Score=...)``

        **参考样例**

        >>> from ngboost import NGBClassifier
        >>> from hscredit.core.models.losses import ExpectedProfitLoss
        >>>
        >>> loss = ExpectedProfitLoss(revenue=100, default_cost=1000)
        >>> model = NGBClassifier(
        ...     Score=loss.to_ngboost(),
        ...     n_estimators=500,
        ...     learning_rate=0.01
        ... )
        >>> model.fit(X_train, y_train)

        **注意**

        - 仅支持 ``Dist=Bernoulli``（NGBoost 默认二分类分布）
        - ``score()`` 使用标准 BCE 作为监控指标
        - ``d_score()`` 使用自定义 loss 的梯度驱动参数更新
        """
        try:
            from ngboost.scores import Score as _NGBScore
        except ImportError:
            raise ImportError(
                "NGBoost未安装，请使用 pip install ngboost 安装"
            )

        loss_obj = self

        class _CustomNGBoostScore(_NGBScore):
            """由 BaseLoss 自动生成的 NGBoost Score 类。"""

            def score(self, Y):
                """计算每个样本的损失值（用于监控 / 早停）。

                :param Y: 真实标签, shape (n_samples,)
                :return: 每样本损失, shape (n_samples,)
                """
                p = np.clip(self.prob, 1e-7, 1 - 1e-7)
                Y = np.asarray(Y, dtype=float)
                # 使用标准 BCE 作为监控指标
                return -(Y * np.log(p) + (1 - Y) * np.log(1 - p))

            def d_score(self, Y):
                """计算损失对 logit 参数的导数（驱动自然梯度更新）。

                通过链式法则: dL/d(logit) = dL/dp × p(1-p)

                :param Y: 真实标签, shape (n_samples,)
                :return: 梯度, shape (1, n_samples)
                """
                p = np.clip(self.prob, 1e-7, 1 - 1e-7)
                Y = np.asarray(Y, dtype=float)

                # 自定义 loss 的 dL/dp
                grad_p = loss_obj.gradient(Y, p)

                # 链式法则: dp/d(logit) = p(1-p)
                grad_logit = grad_p * p * (1 - p)

                # NGBoost 要求 shape = (n_params, n_samples), Bernoulli n_params=1
                return grad_logit.reshape(1, -1)

        # 设置可读名称
        _CustomNGBoostScore.__name__ = f"NGBoost_{loss_obj.name}"
        _CustomNGBoostScore.__qualname__ = f"NGBoost_{loss_obj.name}"

        return _CustomNGBoostScore


class BaseMetric(ABC):
    """评估指标基类。

    所有自定义评估指标都应该继承此类，并实现__call__方法。

    :param name: 指标名称，默认为"custom_metric"
    :param greater_is_better: 是否越大越好，默认为True
    """
    
    def __init__(self, name: str = "custom_metric", greater_is_better: bool = True):
        self.name = name
        self.greater_is_better = greater_is_better
    
    @abstractmethod
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算评估指标。

        :param y_true: 真实标签
        :param y_pred: 预测值
        :return: 指标值
        """
        pass

    def to_xgboost(self) -> Callable:
        """转换为XGBoost格式的评估指标。

        :return: XGBoost可用的评估指标
        """
        def xgb_metric(preds: np.ndarray, dtrain) -> Tuple[str, float]:
            labels = dtrain.get_label()
            value = self(labels, preds)
            return self.name, value

        return xgb_metric

    def to_lightgbm(self) -> Callable:
        """转换为LightGBM格式的评估指标。

        :return: LightGBM可用的评估指标
        """
        def lgb_metric(y_true: np.ndarray, y_pred: np.ndarray):
            value = self(y_true, y_pred)
            return self.name, value, self.greater_is_better

        return lgb_metric

    def to_catboost(self) -> Callable:
        """转换为CatBoost格式的评估指标。

        :return: CatBoost可用的评估指标
        """
        class CatBoostMetricWrapper:
            def __init__(self, metric_obj):
                self.metric_obj = metric_obj

            def evaluate(self, approxes, target, weight):
                assert len(approxes) == 1
                assert len(target) == len(approxes[0])

                preds = approxes[0]
                value = self.metric_obj(target, preds)

                return self.metric_obj.name, value, []

            def get_final_error(self, error, weight):
                return error

            def is_max_optimal(self):
                return self.metric_obj.greater_is_better

        return CatBoostMetricWrapper(self)
