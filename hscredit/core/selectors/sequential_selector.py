"""逐步特征筛选器.

使用前向逐步选择或后向逐步消除搜索最优特征子集。
前向选择从空集开始逐步添加最有价值的特征；
后向消除从全特征集开始逐步剔除最无价值的特征。
基于 sklearn.feature_selection.SequentialFeatureSelector 实现。

**参考样例**

>>> from hscredit.core.selectors import SequentialFeatureSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])  # 10个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = SequentialFeatureSelector(
...     RandomForestClassifier(n_estimators=50, random_state=42),
...     n_features_to_select=5,  # 选择5个最优特征
...     direction='forward',    # 前向选择（从空集开始逐步加入）
...     cv=3                     # 3折交叉验证评估
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
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

    **参考样例**

    ::

        >>> from hscredit.core.selectors import SequentialFeatureSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = SequentialFeatureSelector(
        ...     RandomForestClassifier(n_estimators=50, random_state=42),
        ...     n_features_to_select=5,
        ...     direction='forward',
        ...     cv=3
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)
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
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target, threshold=n_features_to_select, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
        )
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
        # 基于交叉验证评分评估特征子集，无需 importance_getter
        sfs = SklearnSFS(
            estimator=clone(self.estimator),
            n_features_to_select=self.n_features_to_select,
            direction=self.direction,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        sfs.fit(X, y)

        # 获取选中特征
        selected_mask = sfs.get_support()
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.scores_ = pd.Series(
            selected_mask.astype(int),
            index=X.columns
        )
        self._drop_reason = '未选中'
