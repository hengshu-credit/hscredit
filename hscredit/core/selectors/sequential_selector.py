"""逐步特征筛选器.

使用逐步选择（前向/后向）进行特征筛选。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector as SklearnSFS
from sklearn.base import clone

from .base import BaseFeatureSelector


class SequentialFeatureSelector(BaseFeatureSelector):
    """逐步特征筛选器.

    使用前向或后向逐步选择选择最优特征子集。
    前向选择：从空集开始，逐步添加最有价值的特征
    后向消除：从所有特征开始，逐步剔除最无价值的特征

    **参数**

    :param estimator: 评估器
    :param n_features_to_select: 保留的特征数，默认为'auto'
        - 'auto': 保留一半特征
        - 整数: 保留的特征数量
        - 浮点数: 保留的特征比例
    :param direction: 方向，默认为'forward'
        - 'forward': 前向选择
        - 'backward': 后向消除
    :param scoring: 评分指标，默认为None
    :param cv: 交叉验证折数，默认为5
    :param target: 目标变量列名，默认为'target'

    **示例**

    ::

        >>> from hscredit.core.selection import SequentialFeatureSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> selector = SequentialFeatureSelector(
        ...     RandomForestClassifier(n_estimators=100),
        ...     n_features_to_select=10,
        ...     direction='forward'
        ... )
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: Union[int, float, str] = 'auto',
        direction: str = 'forward',
        scoring: Optional[str] = None,
        cv: int = 5,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=n_features_to_select, include=include, exclude=exclude, n_jobs=n_jobs)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.method_name = '逐步筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合逐步筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 使用sklearn的SequentialFeatureSelector
        sfs = SklearnSFS(
            estimator=clone(self.estimator),
            n_features_to_select=self.n_features_to_select,
            direction=self.direction,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        sfs.fit(X, y)

        # 获取选中特征
        selected_mask = sfs.get_support()
        self.select_columns = X.columns[selected_mask].tolist()
        self.scores_ = pd.Series(
            selected_mask.astype(int),
            index=X.columns
        )
        self._drop_reason = '未选中'
