"""基数筛选器.

移除基数（唯一值数量）过高的类别型特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class CardinalitySelector(BaseFeatureSelector):
    """基数筛选器.

    移除基数高于阈值的类别型特征。
    高基数特征可能导致过拟合和计算问题。

    **参数**

    :param threshold: 基数阈值，默认为10
        - 10: 移除唯一值数量超过10的类别型特征
    :param dropna: 是否将NaN视为独立类别，默认为True

    **示例**

    ::

        >>> from hscredit.core.selection import CardinalitySelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'city': ['北京', '上海', '广州', '北京', '深圳'],
        ...     'id': [1, 2, 3, 4, 5],  # 高基数
        ... })
        >>> selector = CardinalitySelector(threshold=4)
        >>> selector.fit(X)
    """

    def __init__(
        self,
        threshold: int = 10,
        dropna: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude)
        self.dropna = dropna
        self.method_name = '基数筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合基数筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 计算各特征的基数
        cardinalities = X.nunique(dropna=self.dropna)
        self.scores_ = cardinalities

        # 选择基数低于阈值的特征
        selected_mask = cardinalities <= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f'唯一值数量 > {self.threshold}'
