"""单一值筛选器.

移除单一值（众数）占比过高的特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_mode_ratio(series: pd.Series, dropna: bool = True) -> float:
    """计算众数占比。

    :param series: 输入序列
    :param dropna: 是否排除缺失值
    :return: 众数占比
    """
    if len(series) == 0:
        return 1.0
    
    summary = series.value_counts(dropna=dropna)
    if len(summary) == 0:
        return 1.0
    
    return summary.iloc[0] / len(series)


class ModeSelector(BaseFeatureSelector):
    """单一值筛选器.

    移除众数占比高于阈值的特征。
    用于过滤掉取值过于集中、区分度低的特征。

    **参数**

    :param threshold: 单一值占比阈值，默认为0.95
        - 0.95: 移除单一值占比超过95%的特征
        - 范围: 0-1之间的浮点数
    :param dropna: 是否将NaN视为独立类别，默认为True
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import ModeSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'a': [1, 1, 1, 1, 2],
        ...     'b': [1, 2, 3, 4, 5],
        ...     'c': [1, 1, 1, 1, 1]
        ... })
        >>> selector = ModeSelector(threshold=0.8)
        >>> selector.fit(X)
        >>> print(selector.select_columns_)
        ['a', 'b']
    """

    def __init__(
        self,
        threshold: float = 0.95,
        dropna: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.dropna = dropna
        self.method_name = '单一值筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合单一值筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 计算各特征的众数占比
        if self.n_jobs == 1:
            mode_ratios = X.apply(
                lambda col: _compute_mode_ratio(col, self.dropna)
            )
        else:
            mode_ratios = pd.Series(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_mode_ratio)(X[col], self.dropna)
                    for col in X.columns
                ),
                index=X.columns
            )

        self.scores_ = mode_ratios

        # 选择众数占比低于阈值的特征
        selected_mask = mode_ratios < self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f'单一值占比 >= {self.threshold:.2%}'
