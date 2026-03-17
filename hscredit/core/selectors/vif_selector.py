"""VIF筛选器.

使用方差膨胀因子（VIF）检测和移除多重共线性特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_vif_single(x: np.ndarray, idx: int) -> float:
    """计算单个特征的VIF值。

    :param x: 特征矩阵
    :param idx: 特征索引
    :return: VIF值
    """
    n_features = x.shape[1]
    if n_features <= 1:
        return np.inf
    
    # 获取其他特征
    mask = np.ones(n_features, dtype=bool)
    mask[idx] = False
    x_other = x[:, mask]
    x_target = x[:, idx]
    
    # 处理缺失值
    valid = ~(np.isnan(x_target) | np.any(np.isnan(x_other), axis=1))
    if valid.sum() < 2:
        return np.inf
    
    x_other = x_other[valid]
    x_target = x_target[valid]
    
    # 线性回归
    lr = LinearRegression(fit_intercept=False)
    lr.fit(x_other, x_target)
    y_pred = lr.predict(x_other)
    
    # 计算VIF
    ss_res = np.sum((x_target - y_pred) ** 2)
    ss_tot = np.sum((x_target - np.mean(x_target)) ** 2)
    
    if ss_tot == 0:
        return np.inf
    
    r2 = 1 - ss_res / ss_tot
    vif = 1 / (1 - r2) if r2 < 1 else np.inf
    
    return vif


class VIFSelector(BaseFeatureSelector):
    """VIF筛选器.

    使用方差膨胀因子（VIF）检测多重共线性。
    VIF值越高，表示特征与其他特征的多重共线性越严重。
    在金融风控中，通常认为VIF > 4存在多重共线性问题。

    **参数**

    :param threshold: VIF阈值，默认为4.0
        - 4.0: 移除VIF值超过4的特征
        - 范围: 正数
    :param missing: 缺失值填充值，默认为-1
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import VIFSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [1, 2, 3, 4, 5],  # 与a完全相关
        ...     'c': [5, 4, 3, 2, 1]
        ... })
        >>> selector = VIFSelector(threshold=4.0)
        >>> selector.fit(X)
        >>> print(selector.select_columns_)
        ['c']
    """

    def __init__(
        self,
        threshold: float = 4.0,
        missing: float = -1.0,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.missing = missing
        self.method_name = 'VIF筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合VIF筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 填充缺失值
        x_filled = X.fillna(self.missing).values

        # 计算VIF
        n_features = x_filled.shape[1]
        if self.n_jobs == 1:
            vif_values = np.array([
                _compute_vif_single(x_filled, i) 
                for i in range(n_features)
            ])
        else:
            vif_values = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_vif_single)(x_filled, i)
                    for i in range(n_features)
                )
            )

        self.scores_ = pd.Series(vif_values, index=X.columns)

        # 选择VIF低于阈值的特征
        selected_mask = vif_values < self.threshold
        self.select_columns = X.columns[selected_mask].tolist()
        self._drop_reason = f'VIF值 >= {self.threshold}'
