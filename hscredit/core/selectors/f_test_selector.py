"""F检验筛选器.

使用F检验（ANOVA）进行特征选择。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile

from .base import BaseFeatureSelector


class FTestSelector(BaseFeatureSelector):
    """F检验筛选器.

    使用F检验（ANOVA）评估特征与目标变量的相关性。
    适用于分类问题。

    F值解释:
    - 值越大: 特征与目标变量越相关

    **参数**

    :param threshold: F值阈值，默认为0.0
    :param k: 保留的特征数，默认为'all'
    :param percentile: 保留的特征百分比，默认为None
    :param target: 目标变量列名，默认为'target'

    **示例**

    ::

        >>> from hscredit.core.selection import FTestSelector
        >>> selector = FTestSelector(k=10)
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        k: Union[int, str] = 'all',
        percentile: Optional[int] = None,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude)
        self.k = k
        self.percentile = percentile
        self.method_name = 'F检验筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合F检验筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 处理类别变量
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X_encoded[col] = pd.factorize(X[col])[0]

        # 计算F值
        f_scores, p_values = f_classif(X_encoded.values, y)

        # 处理NaN
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        
        self.scores_ = pd.Series(f_scores, index=X.columns)

        # 选择特征
        if self.percentile is not None:
            # 使用百分比
            selector = SelectPercentile(percentile=self.percentile)
            selector.fit(X_encoded.values, y)
            selected_mask = selector.get_support()
        elif isinstance(self.k, int):
            # 保留top-k
            top_k = min(self.k, len(X.columns))
            top_indices = np.argsort(f_scores)[-top_k:]
            selected_mask = np.zeros(len(X.columns), dtype=bool)
            selected_mask[top_indices] = True
        else:
            # 使用阈值
            selected_mask = f_scores >= self.threshold

        self.select_columns = X.columns[selected_mask].tolist()
        self._drop_reason = f'F值 < {self.threshold}'
