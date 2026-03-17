"""类型筛选器.

按数据类型筛选特征。
"""

from typing import Union, List, Optional, Type
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


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

    **示例**

    ::

        >>> from hscredit.core.selection import TypeSelector
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
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(include=include, exclude=exclude)
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.method_name = '类型筛选'

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

        # 选择符合类型的列
        if self.dtype_include is not None or self.dtype_exclude is not None:
            selected_cols = X.select_dtypes(
                include=self.dtype_include,
                exclude=self.dtype_exclude
            ).columns.tolist()
        else:
            selected_cols = X.columns.tolist()

        self.scores_ = X.dtypes
        self.selected_features_ = selected_cols
        self._drop_reason = '数据类型不匹配'
