"""将普通概率评估函数转换为统一指标，无需编写指标子类。"""

from copy import deepcopy
from inspect import Parameter, signature

import numpy as np

from .base import BaseMetric, _validate_binary_inputs


class CallableMetric(BaseMetric):
    """包装返回标量的概率评估函数。

    **参数**

    :param function: ``function(y_true, probability, **kwargs) -> float``。
        第一项为0/1标签，第二项为坏样本概率；不接受 sklearn scorer 或框架回调协议。
    :param name: 中文显示名称，默认“自定义指标”。多指标时请使用不同名称。
    :param greater_is_better: 必须明确指定 True（最大化）或 False（最小化）。
    :param kwargs: 固定业务参数，例如阈值或收益参数；创建时复制。
        样本权重请在评价时通过 ``sample_weight`` 提供，不可绑定整个数据集。

    **属性**

    ``direction``、``evaluate``、``to_scorer`` 和框架适配方法继承自 BaseMetric。

    **参考样例**

    >>> import numpy as np
    >>> def error(y, p):
    ...     return float(np.mean((y - p) ** 2))
    >>> metric = CallableMetric(error, name="概率均方误差", greater_is_better=False)
    >>> round(metric([0, 1], [0.1, 0.9]), 2)
    0.01
    """

    def __init__(self, function, *, name="自定义指标", greater_is_better, **kwargs):
        if not callable(function):
            raise TypeError("评估函数必须可调用，签名为 (真实标签, 坏样本概率) -> 标量")
        if not isinstance(greater_is_better, (bool, np.bool_)):
            raise ValueError("greater_is_better 必须是 True 或 False")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("指标名称必须是非空字符串")
        if "sample_weight" in kwargs:
            raise ValueError("样本权重不能固定绑定到指标，请在 evaluate 时传入 sample_weight")
        try:
            parameters = signature(function).parameters
            # bind_partial 允许 sample_weight 等必需参数在调用时提供，同时拒绝误写参数。
            signature(function).bind_partial(None, None, **kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError("评估函数需接收 (y_true, probability)，且固定参数必须与函数签名一致") from exc
        if any(key in parameters for key in ("estimator", "X")) and hasattr(function, "_score_func"):
            raise TypeError("请提供概率评估函数；sklearn scorer 请用于 cross_val_score，不可作为普通指标")
        super().__init__(name, bool(greater_is_better))
        self.function = function
        self.kwargs = deepcopy(kwargs)
        self.supports_sample_weight = "sample_weight" in parameters or any(
            item.kind == Parameter.VAR_KEYWORD for item in parameters.values()
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        """校验并计算原始指标值，不按方向改变正负号。

        :param y_true: 0/1真实标签，长度为样本数。
        :param y_pred: 坏样本概率；可为一维或两列 ``predict_proba``。
        :param sample_weight: 可选非负有限权重，函数须支持该参数。
        :return: 单个有限浮点数；列表、框架三元组、缺失值会报中文错误。

        >>> metric = make_metric(lambda y, p: np.mean((y-p)**2), greater_is_better=False)
        >>> metric([0, 1], [0.2, 0.8]) < 0.05
        True
        """
        y, probability, weight = _validate_binary_inputs(y_true, y_pred, sample_weight)
        kwargs = dict(self.kwargs)
        if weight is not None:
            if not self.supports_sample_weight:
                raise ValueError(f"指标 {self.name} 的函数没有 sample_weight 参数，不能进行加权评价")
            kwargs["sample_weight"] = weight
        result = self.function(y, probability, **kwargs)
        if isinstance(result, (bool, np.bool_)) or np.ndim(result) != 0:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值，不能返回标签、列表或框架回调元组")
        try:
            value = float(result)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值") from exc
        if not np.isfinite(value):
            raise ValueError(f"指标 {self.name} 返回了非有限值")
        return value


def make_metric(function, *, name="自定义指标", greater_is_better, **kwargs):
    """用一行代码定义离线、训练及调参共用的自定义指标。

    :param function: ``(y_true, probability, **kwargs) -> float`` 的函数。
        输入是坏样本概率，不能传 sklearn scorer 或 XGBoost/LightGBM 回调。
    :param name: 显示名称，默认“自定义指标”。
    :param greater_is_better: True 表示越大越好，False 表示越小越好；必须显式提供。
    :param kwargs: 转交函数的固定业务参数，不能包含样本数组权重。
    :return: CallableMetric，可直接传 ``ModelTuner(metric=...)`` 或
        ``LightGBM(eval_metric=...)``；CV 使用返回对象的 ``to_scorer()``。

    **参考样例**

    >>> from sklearn.metrics import log_loss
    >>> metric = make_metric(log_loss, name="交叉熵", greater_is_better=False, labels=[0, 1])
    >>> metric.direction
    'minimize'
    >>> round(metric([0, 1], [0.1, 0.9]), 3)
    0.105
    """
    return CallableMetric(function, name=name, greater_is_better=greater_is_better, **kwargs)
