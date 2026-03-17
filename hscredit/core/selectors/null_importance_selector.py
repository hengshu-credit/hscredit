"""零重要性筛选器（Permutation Importance）.

使用置换重要性识别真正有价值的特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv
from sklearn.base import clone
from sklearn.utils import check_random_state

from .base import BaseFeatureSelector


class NullImportanceSelector(BaseFeatureSelector):
    """零重要性筛选器.

    使用置换重要性（Permutation Importance）识别真正有价值的特征。
    通过多次shuffle目标变量，计算特征真实重要性与随机情况下的比值。

    **参数**

    :param estimator: 评估器
    :param threshold: 阈值，默认为1.0
        - >1: 保留真实重要性/shuffle重要性比值 > threshold的特征
    :param cv: 交叉验证折数，默认为5
    :param n_runs: 置换次数，默认为5
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'

    **示例**

    ::

        >>> from hscredit.core.selection import NullImportanceSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> selector = NullImportanceSelector(
        ...     RandomForestClassifier(n_estimators=100),
        ...     threshold=1.0
        ... )
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        estimator,
        threshold: float = 1.0,
        cv: int = 5,
        n_runs: int = 5,
        random_state: Optional[int] = 42,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.estimator = estimator
        self.cv = cv
        self.n_runs = n_runs
        self.random_state = random_state
        self.method_name = '零重要性筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合零重要性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        
        rng = check_random_state(self.random_state)
        cv = check_cv(self.cv, y, classifier=True)
        
        n_samples, n_features = X.shape
        n_splits = cv.get_n_splits()
        
        # 计算shuffle后的特征重要性
        null_importances = np.zeros((n_features, n_splits * self.n_runs))
        
        for run in range(self.n_runs):
            # Shuffle目标变量
            idx = np.arange(n_samples)
            rng.shuffle(idx)
            y_shuffled = y[idx]
            
            for fold_idx, (train_idx, _) in enumerate(cv.split(y_shuffled, y_shuffled)):
                model = clone(self.estimator)
                model.fit(X.iloc[train_idx], y_shuffled[train_idx])
                null_importance = model.feature_importances_
                null_importances[:, n_splits * run + fold_idx] = null_importance
        
        # 计算真实的特征重要性
        actual_importances = np.zeros((n_features, n_splits * self.n_runs))
        
        for run in range(self.n_runs):
            idx = np.arange(n_samples)
            rng.shuffle(idx)
            X_shuffled = X.iloc[idx]
            y_shuffled = y[idx]
            
            for fold_idx, (train_idx, _) in enumerate(cv.split(y_shuffled, y_shuffled)):
                model = clone(self.estimator)
                model.fit(X_shuffled.iloc[train_idx], y_shuffled[train_idx])
                actual_importance = model.feature_importances_
                actual_importances[:, n_splits * run + fold_idx] = actual_importance
        
        # 计算得分: 真实重要性/shuffle后重要性的均值
        scores = np.zeros(n_features)
        for i in range(n_features):
            actual_mean = np.mean(actual_importances[i, :])
            null_mean = np.mean(null_importances[i, :])
            if null_mean > 0:
                scores[i] = actual_mean / null_mean
            else:
                scores[i] = actual_mean if actual_mean > 0 else 0

        self.scores_ = pd.Series(scores, index=X.columns)

        # 筛选
        selected_mask = scores > self.threshold
        self.select_columns = X.columns[selected_mask].tolist()
        self._drop_reason = f'零重要性得分 <= {self.threshold}'
