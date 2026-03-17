"""相关性筛选器.

移除与其它特征高度相关的特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class CorrSelector(BaseFeatureSelector):
    """相关性筛选器.

    移除与其它特征相关性高于阈值的特征。
    使用加权贪心策略，保留权重高的特征。

    **参数**

    :param threshold: 相关系数阈值，默认为0.7
        - 0.7: 移除与其它特征相关性超过0.7的特征
        - 范围: 0-1之间的浮点数
    :param method: 相关性计算方法，默认为'pearson'
        - 'pearson': 皮尔逊相关系数
        - 'spearman': 斯皮尔曼等级相关系数
        - 'kendall': 肯德尔相关系数
    :param weights: 特征权重，用于决定保留哪个特征，默认为None

    **示例**

    ::

        >>> from hscredit.core.selection import CorrSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [1, 2, 3, 4, 5],  # 与a完全相关
        ...     'c': [5, 4, 3, 2, 1]   # 与a负相关
        ... })
        >>> selector = CorrSelector(threshold=0.9)
        >>> selector.fit(X)
        >>> print(selector.select_columns_)
        ['a', 'c']
    """

    def __init__(
        self,
        threshold: float = 0.7,
        method: str = 'pearson',
        weights: Optional[Union[pd.Series, List[float]]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude)
        self.method = method
        self.weights = weights
        self.method_name = '相关性筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合相关性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 处理权重
        n_features = X.shape[1]
        if self.weights is None:
            weight_arr = np.ones(n_features)
        elif isinstance(self.weights, pd.Series):
            weight_arr = np.zeros(n_features)
            for i, col in enumerate(X.columns):
                if col in self.weights.index:
                    weight_arr[i] = self.weights[col]
        else:
            weight_arr = np.array(self.weights)

        # 按权重排序（权重高的特征优先级高）
        sort_idx = np.argsort(weight_arr)[::-1]
        X_sorted = X.iloc[:, sort_idx]
        sorted_names = X_sorted.columns.tolist()

        # 计算相关矩阵
        corr_matrix = X_sorted.corr(method=self.method).abs()

        # 找出高度相关的特征对
        drops = set()
        upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr = np.where((corr_matrix.values > self.threshold) & upper)

        for i, j in zip(high_corr[0], high_corr[1]):
            # 保留权重高的，剔除权重低的
            if weight_arr[i] >= weight_arr[j]:
                drops.add(j)
            else:
                drops.add(i)

        # 获取保留的特征
        keep_idx = [idx for idx in range(n_features) if idx not in drops]
        self.select_columns = [sorted_names[idx] for idx in keep_idx]

        # 保存scores
        self.scores_ = corr_matrix.max(axis=1)
        self._drop_reason = f'与其它特征相关系数 >= {self.threshold}'
