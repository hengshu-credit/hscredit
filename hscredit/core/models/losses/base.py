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
