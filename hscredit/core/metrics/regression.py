"""回归指标计算.

提供回归模型评估的核心指标。

**参考样例**

>>> from hscredit.core.metrics import mse, mae, rmse, r2
>>> import numpy as np
>>> np.random.seed(42)
>>> y_true = np.random.randn(100) * 10 + 50
>>> y_pred = y_true + np.random.randn(100) * 2
>>> print(f"MSE={mse(y_true, y_pred):.2f}, MAE={mae(y_true, y_pred):.2f}, RMSE={rmse(y_true, y_pred):.2f}, R2={r2(y_true, y_pred):.4f}")
"""

import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mse(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算均方误差 (Mean Squared Error).

    MSE = (1/n) * Σ(y_true - y_pred)²，对大误差更敏感。

    **参数**

    :param y_true: 真实值（目标变量）
    :param y_pred: 预测值
    :return: MSE值，非负浮点数

    **参考样例**

    >>> from hscredit.core.metrics import mse
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 2.2, 2.9, 4.1]
    >>> mse(y_true, y_pred)
    0.0175
    """
    return mean_squared_error(y_true, y_pred)


def mae(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算平均绝对误差 (Mean Absolute Error).

    MAE = (1/n) * Σ|y_true - y_pred|，对异常值鲁棒。

    **参数**

    :param y_true: 真实值（目标变量）
    :param y_pred: 预测值
    :return: MAE值，非负浮点数

    **参考样例**

    >>> from hscredit.core.metrics import mae
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 2.2, 2.9, 4.1]
    >>> mae(y_true, y_pred)
    0.125
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: Union[np.ndarray, pd.Series],
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算均方根误差 (Root Mean Squared Error).

    RMSE = sqrt(MSE)，与目标变量单位一致，便于解释。

    **参数**

    :param y_true: 真实值（目标变量）
    :param y_pred: 预测值
    :return: RMSE值，非负浮点数

    **参考样例**

    >>> from hscredit.core.metrics import rmse
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 2.2, 2.9, 4.1]
    >>> rmse(y_true, y_pred)
    0.132...
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true: Union[np.ndarray, pd.Series],
       y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算决定系数 (R-squared).

    R² = 1 - SS_res / SS_tot，其中SS_res为残差平方和，SS_tot为总平方和。
    取值范围通常为[0, 1]，越接近1表示模型拟合效果越好。

    **参数**

    :param y_true: 真实值（目标变量）
    :param y_pred: 预测值
    :return: R²值，通常在[0, 1]范围内（可为负如果模型比均值预测更差）

    **参考样例**

    >>> from hscredit.core.metrics import r2
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 2.2, 2.9, 4.1]
    >>> r2(y_true, y_pred)
    0.988...
    """
    return r2_score(y_true, y_pred)
