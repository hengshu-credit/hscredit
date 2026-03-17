"""卡方筛选器.

使用卡方检验进行特征选择。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest

from .base import BaseFeatureSelector


class Chi2Selector(BaseFeatureSelector):
    """卡方筛选器.

    使用卡方检验评估特征与目标变量的独立性。
    适用于分类问题和非负特征。

    卡方值解释:
    - 值越大: 特征与目标变量越相关

    **参数**

    :param threshold: 得分阈值，默认为0.0
    :param k: 保留的特征数，默认为'all'
    :param target: 目标变量列名，默认为'target'

    **示例**

    ::

        >>> from hscredit.core.selection import Chi2Selector
        >>> selector = Chi2Selector(k=10)
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        k: Union[int, str] = 'all',
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude)
        self.k = k
        self.method_name = '卡方检验筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合卡方筛选器。

        :param X: 输入特征DataFrame（需要非负值）
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 确保非负
        X_pos = X.copy()
        for col in X_pos.columns:
            if X_pos[col].dtype == 'object':
                X_pos[col] = pd.factorize(X_pos[col])[0]
        
        X_array = X_pos.values
        X_array = np.maximum(X_array, 0)

        # 计算卡方得分
        chi2_scores, p_values = chi2(X_array, y)

        self.scores_ = pd.Series(chi2_scores, index=X.columns)

        # 选择特征
        if isinstance(self.k, int):
            # 保留top-k
            top_k = min(self.k, len(X.columns))
            top_indices = np.argsort(chi2_scores)[-top_k:]
            selected_cols = X.columns[top_indices].tolist()
        else:
            # 使用阈值
            selected_mask = chi2_scores >= self.threshold
            selected_cols = X.columns[selected_mask].tolist()

        self.select_columns = selected_cols
        self._drop_reason = f'卡方得分 < {self.threshold}'
