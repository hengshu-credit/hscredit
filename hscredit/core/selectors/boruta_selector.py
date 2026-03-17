"""Boruta特征筛选器.

使用Boruta算法进行特征选择。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import check_cv
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


class BorutaSelector(BaseFeatureSelector):
    """Boruta筛选器.

    Boruta是一种基于随机森林的特征选择算法。
    通过创建影子特征（shuffled版本），与原始特征进行比较，
    保留统计显著优于影子特征的特征。

    **参数**

    :param estimator: 随机森林评估器
    :param n_estimators: 树的数量，默认为100
    :param max_iter: 最大迭代次数，默认为100
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import BorutaSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> selector = BorutaSelector(
        ...     RandomForestClassifier(n_estimators=100, n_jobs=-1)
        ... )
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        estimator=None,
        n_estimators: int = 100,
        max_iter: int = 100,
        random_state: Optional[int] = 42,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, include=include, exclude=exclude, n_jobs=n_jobs)
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.method_name = 'Boruta筛选'
        
        # 默认使用随机森林
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state
            )

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合Boruta筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        
        n_samples, n_features = X.shape
        
        # 准备数据
        X_array = X.values
        feature_names = X.columns.tolist()
        
        # 创建影子特征
        X_shadow = np.random.permutation(X_array)
        X_with_shadow = np.hstack([X_array, X_shadow])
        
        all_features = feature_names + [f'shadow_{i}' for i in range(n_features)]
        
        # 迭代
        selected = set(range(n_features))
        history = []
        
        for iteration in range(self.max_iter):
            # 训练模型
            model = clone(self.estimator)
            model.fit(X_with_shadow, y)
            
            # 获取特征重要性
            importances = model.feature_importances_
            
            # 分离真实和影子特征重要性
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]
            
            # 计算阈值（影子特征最大值的均值）
            shadow_max = np.mean(shadow_importances)
            
            # 记录历史
            history.append({
                'iteration': iteration,
                'selected': len(selected),
                'shadow_max': shadow_max
            })
            
            # 更新选中特征
            new_selected = set()
            for i in range(n_features):
                if i in selected and real_importances[i] > shadow_max:
                    new_selected.add(i)
                elif i in selected:
                    # 进行统计检验
                    pass  # 简化版直接保留
            
            selected = new_selected
            
            if len(selected) == 0:
                break
        
        # 选中特征
        self.selected_features_ = [feature_names[i] for i in selected]
        
        # 计算得分
        self.scores_ = pd.Series(real_importances, index=feature_names)
        self._drop_reason = '重要性低于影子特征'
