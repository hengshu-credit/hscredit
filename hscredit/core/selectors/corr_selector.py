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
        include: Optional[List[str]] = None,
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
        self.selected_features_ = [sorted_names[idx] for idx in keep_idx]

        # 保存scores
        self.scores_ = corr_matrix.max(axis=1)

        # 构建详细的dropped_记录，包含相关性信息
        if len(drops) > 0:
            dropped_cols = [sorted_names[idx] for idx in drops]
            # 找到每个被剔除特征的最大相关系数及对应的特征
            max_corr_values = []
            max_corr_features = []
            for idx in drops:
                col_name = sorted_names[idx]
                # 获取该特征与其他特征的相关系数
                corr_values = corr_matrix.loc[col_name, :].copy()
                corr_values[col_name] = 0  # 排除自身
                max_corr = corr_values.max()
                max_corr_feat = corr_values.idxmax()
                max_corr_values.append(max_corr)
                max_corr_features.append(max_corr_feat)

            self.dropped_ = pd.DataFrame({
                '特征': dropped_cols,
                '剔除原因': [f'与{max_corr_features[i]}相关系数({max_corr_values[i]:.4f}) >= 阈值({self.threshold})' for i in range(len(dropped_cols))],
                '最大相关系数': max_corr_values,
                '相关特征': max_corr_features,
                '阈值': [self.threshold] * len(dropped_cols),
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', '最大相关系数', '相关特征', '阈值'])
