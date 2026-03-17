"""互信息筛选器.

使用互信息进行特征选择。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .base import BaseFeatureSelector


class MutualInfoSelector(BaseFeatureSelector):
    """互信息筛选器.

    使用互信息（Mutual Information）评估特征与目标变量的相关性。
    互信息可以捕捉非线性关系。

    互信息值解释:
    - 0: 特征与目标完全独立
    - 值越大: 特征与目标的依赖关系越强

    **参数**

    :param threshold: 互信息阈值，默认为0.0
    :param n_neighbors: 邻居数，用于估计互信息，默认为3
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import MutualInfoSelector
        >>> selector = MutualInfoSelector(threshold=0.1)
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.method_name = '互信息筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合互信息筛选器。

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

        # 计算互信息
        mi_scores = mutual_info_classif(
            X_encoded.values,
            y,
            discrete_features=False,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        self.scores_ = pd.Series(mi_scores, index=X.columns)

        # 选择互信息大于阈值的特征
        selected_mask = mi_scores >= self.threshold
        self.select_columns = X.columns[selected_mask].tolist()
        self._drop_reason = f'互信息值 < {self.threshold}'
