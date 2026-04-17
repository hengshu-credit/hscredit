"""正则表达式筛选器.

按特征名称正则表达式筛选特征。

**参考样例**

>>> from hscredit.core.selectors import RegexSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'income_1': [1, 2, 3],
...     'income_2': [4, 5, 6],
...     'age': [1, 2, 3]
... })
>>> # 选择以income开头的特征
>>> selector = RegexSelector(pattern='^income')
>>> selector.fit(X)
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class RegexSelector(BaseFeatureSelector):
    """正则表达式筛选器.

    按特征名称的正则表达式匹配筛选特征。

    **参数**

    :param pattern: 正则表达式模式
    :param exclude: 是否排除匹配的特征，默认为False

    **参考样例**

    ::

        >>> from hscredit.core.selectors import RegexSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'income_1': [1, 2, 3],
        ...     'income_2': [4, 5, 6],
        ...     'age': [1, 2, 3]
        ... })
        >>> # 选择以income开头的特征
        >>> selector = RegexSelector(pattern='^income')
        >>> selector.fit(X)
    """

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target, include=include, exclude=exclude,
            force_drop=force_drop, n_jobs=n_jobs,
        )
        self.pattern = pattern
        self.flags = flags
        self.method_name = '正则筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合正则筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 正则匹配
        matches = X.columns.str.contains(self.pattern, regex=True, flags=self.flags)
        
        if self.exclude:
            selected_cols = X.columns[~matches].tolist()
            self.scores_ = (~matches).astype(int)
        else:
            selected_cols = X.columns[matches].tolist()
            self.scores_ = matches.astype(int)

        self.selected_features_ = selected_cols
        self._drop_reason = f'特征名不匹配正则表达式: {self.pattern}'
