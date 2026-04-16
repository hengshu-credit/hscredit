"""缺失值筛选器.

移除缺失率过高的特征。

**参考样例**

>>> from hscredit.core.selectors import NullSelector
>>> import pandas as pd
>>> import numpy as np
>>> X = pd.DataFrame({
...     'a': [1, 2, np.nan, 4, 5],
...     'b': [1, 2, 3, 4, 5],
...     'c': [np.nan, np.nan, np.nan, np.nan, np.nan]
... })
>>> selector = NullSelector(threshold=0.5)
>>> selector.fit(X)
>>> print(selector.selected_features_)
['a', 'b']
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class NullSelector(BaseFeatureSelector):
    """缺失率筛选器.

    移除缺失率高于阈值的特征。
    用于过滤掉数据质量较差的特征。

    **参数**

    :param threshold: 缺失率阈值，默认为0.95
        - 0.95: 移除缺失率超过95%的特征
        - 范围: 0-1之间的浮点数

    **参考样例**

    ::

        >>> from hscredit.core.selectors import NullSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 4, 5],
        ...     'b': [1, 2, 3, 4, 5],
        ...     'c': [np.nan, np.nan, np.nan, np.nan, np.nan]
        ... })
        >>> selector = NullSelector(threshold=0.5)
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['a', 'b']
    """

    def __init__(
        self,
        threshold: float = 0.95,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target, threshold=threshold, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
        )
        self.method_name = '缺失率筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合缺失率筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        # 计算缺失率
        null_counts = X.isnull().sum()
        null_rates = null_counts / len(X)
        self.scores_ = null_rates

        # 选择缺失率低于阈值的特征
        selected_mask = null_rates < self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含缺失率数值
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame({
                '特征': dropped_cols,
                '剔除原因': [f'缺失率({null_rates[col]:.2%}) >= 阈值({self.threshold:.2%})' for col in dropped_cols],
                '缺失率': [null_rates[col] for col in dropped_cols],
                '阈值': [self.threshold] * len(dropped_cols),
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', '缺失率', '阈值'])
