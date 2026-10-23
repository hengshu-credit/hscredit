"""单次自动留出中的逐行金融参数路由，不按数组长度猜测外部评估集身份。"""

import copy
import numpy as np
from .base import BaseLoss, LossMetric


def bound_sample_params(loss):
    """列出损失明确声明的逐行金额或金融参数，标量配置不参与切分。"""
    if not isinstance(loss, BaseLoss):
        return {}
    result = {}
    for attribute, parameter in (
        ("amounts_", "amounts"),
        ("lgd", "lgd"),
        ("ead", "ead"),
        ("rate", "rate"),
        ("cost", "cost"),
    ):
        value = getattr(loss, attribute, None)
        if value is not None and np.ndim(value) > 0:
            result[parameter] = np.asarray(value)
    return result


def slice_sample_loss(value, indices, n_samples):
    """按已确定的位置切分独立副本；指标绑定相应验证部分，训练对象不被修改。"""
    if isinstance(value, (list, tuple)):
        return type(value)(slice_sample_loss(item, indices, n_samples) for item in value)
    loss = value.loss if isinstance(value, LossMetric) else value
    parameters = bound_sample_params(loss)
    if not parameters:
        return value
    if not callable(getattr(loss, "set_sample_params", None)):
        raise ValueError("绑定逐行金融参数的损失需要实现 set_sample_params 才能自动切分")
    for name, array in parameters.items():
        if array.shape != (n_samples,):
            raise ValueError(f"金融参数 {name} 必须与原始训练输入等长，不能猜测其对应的数据集")
    cloned = copy.deepcopy(value)
    target = cloned.loss if isinstance(cloned, LossMetric) else cloned
    target.set_sample_params(**{name: array[indices].copy() for name, array in parameters.items()})
    return cloned


def has_bound_sample_params(value):
    if isinstance(value, (list, tuple)):
        return any(has_bound_sample_params(item) for item in value)
    return bool(bound_sample_params(value.loss if isinstance(value, LossMetric) else value))
