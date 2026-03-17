"""LIFT筛选器.

使用LIFT值进行特征筛选。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_lift_single(x: np.ndarray, y: np.ndarray) -> float:
    """计算单个特征的LIFT值。

    :param x: 特征值数组
    :param y: 目标变量数组
    :return: LIFT值
    """
    if len(np.unique(x)) <= 1:
        return 1.0

    base_bad_rate = np.mean(y)
    if base_bad_rate == 0 or base_bad_rate == 1:
        return 1.0

    lift_scores = []
    for v in np.unique(x):
        mask = x == v
        if np.sum(mask) == 0:
            continue
        bad_rate = np.mean(y[mask])
        if bad_rate > 0:
            lift_scores.append(bad_rate / base_bad_rate)

    return np.nanmax(lift_scores) if len(lift_scores) > 0 else 1.0


class LiftSelector(BaseFeatureSelector):
    """LIFT筛选器.

    使用LIFT值筛选特征。
    LIFT衡量特征对目标群体的提升程度。

    LIFT值解释:
    - LIFT = 1: 特征无提升能力
    - LIFT > 1: 特征有提升能力，值越大提升能力越强
    - LIFT < 1: 特征反而降低响应率

    **参数**

    :param threshold: LIFT阈值，默认为1.0
        - 1.0: 仅保留LIFT值大于1的特征
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import LiftSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'income': [5000, 8000, 12000, 2000, 15000],
        ...     'age': [25, 35, 45, 55, 23],
        ... })
        >>> y = pd.Series([0, 0, 1, 0, 1])
        >>> selector = LiftSelector(threshold=1.0)
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.method_name = 'LIFT筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合LIFT筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        y = np.asarray(y)

        # 计算LIFT值
        if self.n_jobs == 1:
            lift_values = np.array([
                _compute_lift_single(X[col].values, y)
                for col in X.columns
            ])
        else:
            lift_values = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_lift_single)(X[col].values, y)
                    for col in X.columns
                )
            )

        self.scores_ = pd.Series(lift_values, index=X.columns)

        # 选择LIFT值大于等于阈值的特征
        selected_mask = lift_values >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f'LIFT值 < {self.threshold}'
