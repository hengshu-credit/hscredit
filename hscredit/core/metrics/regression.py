"""回归指标计算.

提供回归模型评估的核心指标。

主要指标:
- mse: 均方误差 (Mean Squared Error)
- mae: 平均绝对误差 (Mean Absolute Error)
- rmse: 均方根误差 (Root Mean Squared Error)
- r2: 决定系数 (R-squared)
"""

import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mse(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算均方误差 (Mean Squared Error).

    MSE = (1/n) * Σ(y_true - y_pred)²

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MSE值
    """
    return mean_squared_error(y_true, y_pred)


def mae(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算平均绝对误差 (Mean Absolute Error).

    MAE = (1/n) * Σ|y_true - y_pred|

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: Union[np.ndarray, pd.Series],
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算均方根误差 (Root Mean Squared Error).

    RMSE = sqrt(MSE)

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true: Union[np.ndarray, pd.Series],
       y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算决定系数 (R-squared).

    R² = 1 - SS_res / SS_tot

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: R²值，取值范围[0, 1]
    """
    return r2_score(y_true, y_pred)
