"""PSI稳定性筛选器.

使用群体稳定性指标（PSI）筛选特征。

**参考样例**

>>> from hscredit.core.selectors import PSISelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = PSISelector(threshold=0.25, n_splits=5)  # 筛选PSI<0.25的稳定特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_psi_single(expected: np.ndarray, actual: np.ndarray) -> float:
    """计算单个特征的PSI值。

    :param expected: 期望值数组（训练集）
    :param actual: 实际值数组（测试集）
    :return: PSI值
    """
    # 获取所有唯一值
    all_values = np.unique(np.concatenate([expected, actual]))
    
    if len(all_values) <= 1:
        return 0.0

    # 计算分位数区间
    bins = np.percentile(expected, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    
    if len(bins) < 2:
        return 0.0

    # 统计各区间占比
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    # 避免除零
    expected_counts = np.maximum(expected_counts, 1)
    actual_counts = np.maximum(actual_counts, 1)

    expected_rates = expected_counts / expected_counts.sum()
    actual_rates = actual_counts / actual_counts.sum()

    # 计算PSI
    psi = np.sum(
        (actual_rates - expected_rates) * 
        np.log(actual_rates / expected_rates)
    )

    return psi


class PSISelector(BaseFeatureSelector):
    """PSI筛选器.

    使用群体稳定性指标（Population Stability Index）筛选特征。
    PSI衡量特征在不同样本间的分布稳定性。
    常用于跨时间验证和oot验证。

    PSI值解释:
    - < 0.1: 特征稳定性好
    - 0.1 - 0.25: 特征有轻微变化，需要关注
    - > 0.25: 特征分布变化显著，需要处理

    **参数**

    :param threshold: PSI阈值，默认为0.25
        - 0.25: 移除PSI值超过0.25的特征
    :param n_splits: 交叉验证折数，用于计算PSI
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **参考样例**

    ::

        >>> from hscredit.core.selectors import PSISelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 1000))
        >>> selector = PSISelector(threshold=0.25, n_splits=5)
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        threshold: float = 0.25,
        n_splits: int = 5,
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
        self.n_splits = n_splits
        self.method_name = 'PSI筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合PSI筛选器。

        使用交叉验证，将数据分为训练集和验证集，计算PSI。

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

        # 使用交叉验证计算PSI
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        psi_values = np.zeros(len(X.columns))
        
        for train_idx, test_idx in kfold.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            for i, col in enumerate(X.columns):
                psi = _compute_psi_single(
                    X_train[col].values,
                    X_test[col].values
                )
                psi_values[i] += psi

        psi_values /= self.n_splits
        self.scores_ = pd.Series(psi_values, index=X.columns)

        # 选择PSI值小于阈值的特征（PSI越小越稳定）
        selected_mask = psi_values < self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含PSI值
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame({
                '特征': dropped_cols,
                '剔除原因': [f'PSI值({self.scores_[col]:.4f}) >= 阈值({self.threshold})' for col in dropped_cols],
                'PSI值': [self.scores_[col] for col in dropped_cols],
                '阈值': [self.threshold] * len(dropped_cols),
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', 'PSI值', '阈值'])
