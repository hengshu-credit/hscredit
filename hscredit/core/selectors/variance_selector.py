"""方差筛选器.

移除低方差特征。

**参考样例**

>>> from hscredit.core.selectors import VarianceSelector
>>> import pandas as pd
>>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})  # b为常量特征，方差为0
>>> selector = VarianceSelector(threshold=0.1)  # 移除方差<0.1的特征
>>> selector.fit(X)
>>> print(selector.selected_features_)
['a', 'c']
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold

from .base import BaseFeatureSelector


def _compute_variance_feature(task):
    """计算单列总体方差与峰值差。"""
    feature, series = task
    return feature, series.var(ddof=0), series.max() - series.min()


class VarianceSelector(BaseFeatureSelector):
    """方差筛选器.

    移除方差低于阈值的特征。
    常用于移除常量特征或近似常量特征。

    **参数**

    :param threshold: 方差阈值，默认为0.0
        - 0.0: 移除常量特征（方差为0）
        - 其他值: 移除方差小于该值的特征

    **参考样例**

    ::

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['a', 'c']

    **注意**

    方差筛选为无监督方法，不使用标签 ``y``；方差受量纲影响，不同尺度特征建议先标准化
    再比较，否则大量纲特征会因方差天然偏大而被保留。

    **引用**

    对齐 sklearn ``VarianceThreshold``：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    """

    def __init__(
        self,
        threshold: float = 0.0,
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
            threshold=threshold,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.method_name = "方差筛选"

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合方差筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        self.scores_ = X.var(axis=0, ddof=0, numeric_only=False).reindex(X.columns)
        peak_to_peak = (X.max(axis=0) - X.min(axis=0)).reindex(X.columns)

        # 根据阈值筛选
        if self.threshold == 0:
            scores = np.minimum(self.scores_.fillna(0).values, peak_to_peak.fillna(0).values)
            self.scores_ = pd.Series(scores, index=X.columns)

        # 选择方差大于阈值的特征
        selected_mask = self.scores_ > self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含方差值
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [f"方差({self.scores_[col]:.6f}) <= 阈值({self.threshold})" for col in dropped_cols],
                    "方差": [self.scores_[col] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "方差", "阈值"])
