"""方差筛选器.

移除低方差特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold

from .base import BaseFeatureSelector


class VarianceSelector(BaseFeatureSelector):
    """方差筛选器.

    移除方差低于阈值的特征。
    常用于移除常量特征或近似常量特征。

    **参数**

    :param threshold: 方差阈值，默认为0.0
        - 0.0: 移除常量特征（方差为0）
        - 其他值: 移除方差小于该值的特征

    **示例**

    ::

        >>> from hscredit.core.selection import VarianceSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X)
        >>> print(selector.select_columns_)
        ['a', 'c']
    """

    def __init__(
        self,
        threshold: float = 0.0,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude)
        self.method_name = '方差筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合方差筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 计算方差
        if hasattr(X, 'toarray'):
            # 稀疏矩阵
            from sklearn.utils.sparsefuncs import mean_variance_axis
            _, var = mean_variance_axis(X, axis=0)
            self.scores_ = pd.Series(var, index=X.columns)
        else:
            self.scores_ = pd.Series(X.var(ddof=0).values, index=X.columns)

        # 根据阈值筛选
        if self.threshold == 0:
            # 使用峰值差避免数值精度问题
            if hasattr(X, 'toarray'):
                from sklearn.utils.sparsefuncs import min_max_axis
                _, mins, maxs = min_max_axis(X, axis=0)
                peak_to_peak = maxs - mins
            else:
                peak_to_peak = X.max() - X.min()

            scores = np.minimum(
                self.scores_.fillna(0).values,
                peak_to_peak.fillna(0).values
            )
            self.scores_ = pd.Series(scores, index=X.columns)

        # 选择方差大于阈值的特征
        selected_mask = self.scores_ > self.threshold
        self.select_columns = X.columns[selected_mask].tolist()
        self._drop_reason = f'方差 <= {self.threshold}'
