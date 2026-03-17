"""回归指标计算.

提供回归模型评估的核心指标。

主要指标:
- MSE (Mean Squared Error): 均方误差
- MAE (Mean Absolute Error): 平均绝对误差
- RMSE (Root Mean Squared Error): 均方根误差
- R2 (R-squared): 决定系数
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def MSE(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算Mean Squared Error (均方误差).

    MSE = (1/n) * Σ(y_true - y_pred)²

    MSE值越小表示模型预测越准确。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MSE值
    """
    return mean_squared_error(y_true, y_pred)


def MAE(y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算Mean Absolute Error (平均绝对误差).

    MAE = (1/n) * Σ|y_true - y_pred|

    MAE值越小表示模型预测越准确，对异常值不如MSE敏感。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def RMSE(y_true: Union[np.ndarray, pd.Series],
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算Root Mean Squared Error (均方根误差).

    RMSE = sqrt(MSE) = sqrt((1/n) * Σ(y_true - y_pred)²)

    RMSE与目标变量的量纲相同，便于理解。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def R2(y_true: Union[np.ndarray, pd.Series],
       y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算R-squared (决定系数).

    R² = 1 - SS_res / SS_tot

    其中:
    - SS_res = Σ(y_true - y_pred)² (残差平方和)
    - SS_tot = Σ(y_true - y_mean)² (总平方和)

    R²取值范围为[0, 1]，越接近1表示模型拟合越好。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: R²值，取值范围[0, 1]
    """
    return r2_score(y_true, y_pred)
