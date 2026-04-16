"""递归特征消除筛选器.

递归特征消除（Recursive Feature Elimination）通过递归方式逐步剔除
最不重要的特征，直到达到目标数量。适用于任何有 feature_importances_
或 coef_ 属性的模型。基于 sklearn.feature_selection.RFE 实现。

**参考样例**

>>> from hscredit.core.selectors import RFESelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
>>> y = np.random.randint(0, 2, 200)
>>> selector = RFESelector(
...     RandomForestClassifier(n_estimators=100, random_state=42),
...     n_features_to_select=5
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE as SklearnRFE

from .base import BaseFeatureSelector


class RFESelector(BaseFeatureSelector):
    """递归特征消除筛选器.

    通过递归方式逐步剔除最不重要的特征。
    适用于任何有feature_importances_或coef_属性的模型。

    **参数**

    :param estimator: 评估器
    :param n_features_to_select: 保留的特征数，默认为10
        - 整数: 保留的特征数量
        - 浮点数: 保留的特征比例
    :param step: 每次剔除的特征数，默认为1
    :param target: 目标变量列名，默认为'target'

    **参考样例**

    ::

        >>> from hscredit.core.selectors import RFESelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = RFESelector(
        ...     RandomForestClassifier(n_estimators=100, random_state=42),
        ...     n_features_to_select=5
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: Union[int, float] = 10,
        step: int = 1,
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
        self.step = step
        self.method_name = 'RFE筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合RFE筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 使用sklearn的RFE
        rfe = SklearnRFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step
        )
        rfe.fit(X, y)

        # 获取选中特征
        selected_mask = rfe.support_
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 获取特征排名（越小越重要）
        self.scores_ = pd.Series(
            rfe.ranking_,
            index=X.columns
        )
        self._drop_reason = 'RFE排名较低'
