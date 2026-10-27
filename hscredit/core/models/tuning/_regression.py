"""回归搜索指标：连续预测与独立评估权重，不套用分类概率契约。"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..losses.base import BaseLoss, BaseMetric
from ._metrics import Metric


class RegressionMetric(Metric):
    """ModelTuner 内部的回归指标适配。"""

    BUILTIN_METRICS = {
        "rmse": {"direction": "minimize"},
        "mse": {"direction": "minimize"},
        "mae": {"direction": "minimize"},
        "r2": {"direction": "maximize"},
    }
    LABELS = {"rmse": "均方根误差", "mse": "均方误差", "mae": "平均绝对误差", "r2": "决定系数"}

    def __init__(self, metric, name=None, direction=None):
        if isinstance(metric, (BaseLoss, BaseMetric)):
            raise ValueError("回归调参请使用 rmse、mse、mae、r2 或连续预测函数；BaseLoss/BaseMetric 为分类概率协议")
        super().__init__(
            metric,
            name=name or self.LABELS.get(metric.lower() if isinstance(metric, str) else None),
            direction=direction,
        )

    def __call__(self, y_true, y_pred, y_train=None, y_train_pred=None, sample_weight=None, train_sample_weight=None):
        from .._contracts import validate_sample_weight

        y, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        if y.ndim != 1 or prediction.shape != y.shape or not len(y):
            raise ValueError("回归评估要求标签与预测是非空、等长一维数组")
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("回归标签与预测必须是有限数值")
        weight = validate_sample_weight(sample_weight, len(y))
        if weight is not None:
            self.validate_evaluation_weight()
        if self._is_builtin:
            key = self.metric.lower()
            function = {
                "mse": mean_squared_error,
                "rmse": mean_squared_error,
                "mae": mean_absolute_error,
                "r2": r2_score,
            }[key]
            value = function(y, prediction, sample_weight=weight)
            if key == "rmse":
                value = np.sqrt(value)
        else:
            value = (
                self.scorer(y, prediction, sample_weight=weight) if weight is not None else self.scorer(y, prediction)
            )
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值") from exc
        if not np.isfinite(value):
            raise ValueError(f"指标 {self.name} 返回了非有限数值")
        return value
