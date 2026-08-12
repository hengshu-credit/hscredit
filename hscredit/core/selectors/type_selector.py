"""类型筛选器.

按数据类型筛选特征。

**参考样例**

>>> from hscredit.core.selectors import TypeSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'a': [1, 2, 3],
...     'b': ['x', 'y', 'z'],
...     'c': [1.0, 2.0, 3.0]
... })
>>> # 仅保留数值类型
>>> selector = TypeSelector(dtype_include='number')
>>> selector.fit(X)
"""

from typing import Union, List, Optional, Type, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


def _matches_dtype_feature(task):
    """判断单列是否满足类型包含/排除条件。"""
    feature, series, dtype_include, dtype_exclude = task
    if dtype_include is None and dtype_exclude is None:
        return feature, True
    selected = series.to_frame().select_dtypes(
        include=dtype_include,
        exclude=dtype_exclude,
    )
    return feature, feature in selected.columns


class TypeSelector(BaseFeatureSelector):
    """类型筛选器.

    按数据类型筛选特征。
    可以按包含类型或排除类型进行筛选。

    **参数**

    :param dtype_include: 包含的数据类型，默认为None
        - numpy.number: 所有数值类型
        - 'object': 所有对象类型
        - 'category': 类别类型
    :param dtype_exclude: 排除的数据类型，默认为None

    **参考样例**

    ::

        >>> from hscredit.core.selectors import TypeSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': ['x', 'y', 'z'],
        ...     'c': [1.0, 2.0, 3.0]
        ... })
        >>> # 仅保留数值类型
        >>> selector = TypeSelector(dtype_include='number')
        >>> selector.fit(X)
    """

    def __init__(
        self,
        dtype_include: Optional[Union[str, Type, List[str]]] = None,
        dtype_exclude: Optional[Union[str, Type, List[str]]] = None,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.method_name = "类型筛选"

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合类型筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        if self.dtype_include is None and self.dtype_exclude is None:
            selected_cols = list(X.columns)
        else:
            selected_cols = X.select_dtypes(
                include=self.dtype_include,
                exclude=self.dtype_exclude,
            ).columns.tolist()

        self.dtypes_ = X.dtypes
        self.scores_ = pd.Series(
            [1 if c in selected_cols else 0 for c in X.columns],
            index=X.columns,
        )
        self.selected_features_ = selected_cols
        self._drop_reason = "数据类型不匹配"
