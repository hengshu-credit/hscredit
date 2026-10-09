"""二分类损失的概率输入校验和解析导数。"""

from numbers import Real

import numpy as np


def binary_inputs(y_true, y_pred):
    """校验一维二分类标签及概率；端点仅用于数值稳定性截断。"""
    try:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("标签与预测概率必须是可转换为数值的数组。") from exc
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not y.size:
        raise ValueError("标签与预测概率必须是长度相同的非空一维数组。")
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("标签必须为 0（好样本）或 1（坏样本）。")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("预测值必须是 [0, 1] 内的有限坏样本概率。")
    return y, np.clip(p, 1e-7, 1 - 1e-7)


def nonnegative(**parameters):
    """校验非负有限标量参数。"""
    for name, value in parameters.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"参数 {name} 必须是非负有限实数，不能使用布尔值。")
        try:
            valid = bool(np.isfinite(value)) and value >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"参数 {name} 必须是非负有限数值。")


def unit_interval(**parameters):
    """校验 [0, 1] 内有限参数。"""
    nonnegative(**parameters)
    for name, value in parameters.items():
        if value > 1:
            raise ValueError(f"参数 {name} 必须在 [0, 1] 范围内。")


def positive(**parameters):
    """校验正有限参数。"""
    nonnegative(**parameters)
    for name, value in parameters.items():
        if value == 0:
            raise ValueError(f"参数 {name} 必须大于 0。")


def bce_terms(y, p):
    """返回逐样本 BCE、相对概率的一阶导和二阶导（不除以样本数）。"""
    value = -y * np.log(p) - (1 - y) * np.log1p(-p)
    grad = -y / p + (1 - y) / (1 - p)
    hess = y / p**2 + (1 - y) / (1 - p) ** 2
    return value, grad, hess


def focal_terms(pt, gamma):
    """返回 - (1-pt)^gamma log(pt) 及其相对 pt 的精确导数。"""
    q = 1 - pt
    log_pt = np.log(pt)
    value = -(q**gamma) * log_pt
    grad = gamma * q ** (gamma - 1) * log_pt - q**gamma / pt
    hess = -gamma * (gamma - 1) * q ** (gamma - 2) * log_pt + 2 * gamma * q ** (gamma - 1) / pt + q**gamma / pt**2
    return value, grad, hess
