"""基于sklearn的风控模型.

提供RandomForest、ExtraTrees、GradientBoosting等模型的统一封装。

**参考样例**

>>> from hscredit.core.models import RandomForestRiskModel, GradientBoostingRiskModel
>>> model = RandomForestRiskModel(n_estimators=100, max_depth=10)  # 随机森林模型
>>> model.fit(X_train, y_train)  # 训练模型
>>> proba = model.predict_proba(X_test)  # 预测概率
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.utils.validation import check_is_fitted

from ..base import BaseRiskModel


class SklearnRiskModel(BaseRiskModel):
    """基于sklearn的模型基类.

    封装sklearn分类器，提供统一的接口。
    """

    def __init__(
        self,
        estimator_class,
        objective: str = 'binary',
        eval_metric: Union[str, List[str], None] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=None,  # sklearn不支持早停
            validation_fraction=validation_fraction,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )
        self._estimator_class = estimator_class

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'SklearnRiskModel':
        """训练模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: fit(X) 在init中指定target
        """
        # 准备数据（支持从X中提取target）
        X, y, sample_weight = self._prepare_data(X, y, sample_weight, extract_target=True)

        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        # _prepare_data 已在内部设置 feature_names_in_（DataFrame 或人工命名）
        self.classes_ = np.unique(y)

        # 构建参数
        params = self.kwargs.copy()
        params['random_state'] = self.random_state
        params['verbose'] = self.verbose
        # GradientBoosting不支持n_jobs参数
        if self._estimator_class != GradientBoostingClassifier:
            params['n_jobs'] = self.n_jobs

        # 创建模型
        self._model = self._estimator_class(**params)

        # 训练
        if sample_weight is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)

        # 保存评估结果
        self._evals_result = {}
        if eval_set:
            for i, (X_val, y_val) in enumerate(eval_set):
                scores = self.evaluate(X_val, y_val)
                self._evals_result[f'validation_{i}'] = scores

        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict_proba(X)

    def get_feature_importances(self, importance_type: str = 'gain') -> pd.Series:
        """获取特征重要性."""
        check_is_fitted(self, '_is_fitted')

        importances = self._model.feature_importances_

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=self.feature_names_in_,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性属性 (兼容sklearn风格).

        直接在包装类上暴露重要性，兼容sklearn RFE/SFS等组件的 importance_getter。
        """
        check_is_fitted(self, '_is_fitted')
        # 直接从内部模型获取，避免缓存逻辑在clone后出错
        return self._model.feature_importances_


class RandomForestRiskModel(SklearnRiskModel):
    """随机森林风控模型.

    基于sklearn的RandomForestClassifier封装。

    **参数**

    :param n_estimators: 树的数量，默认100
    :param max_depth: 树最大深度，默认None
    :param min_samples_split: 节点分裂最小样本数，默认2
    :param min_samples_leaf: 叶子节点最小样本数，默认1
    :param max_features: 最大特征数，默认'sqrt'
    :param bootstrap: 是否使用自助采样，默认True
    :param class_weight: 类别权重，默认None
    :param criterion: 分裂标准，默认'gini'
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1
    :param verbose: 是否输出详细信息，默认False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = True,
        class_weight: Optional[Union[str, Dict]] = None,
        criterion: str = 'gini',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            estimator_class=RandomForestClassifier,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.criterion = criterion

        # 更新kwargs
        self.kwargs.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'class_weight': class_weight,
            'criterion': criterion,
        })


class ExtraTreesRiskModel(SklearnRiskModel):
    """极端随机树风控模型.

    基于sklearn的ExtraTreesClassifier封装。
    比普通随机森林随机性更强，训练更快。

    **参数**

    :param n_estimators: 树的数量，默认100
    :param max_depth: 树最大深度，默认None
    :param min_samples_split: 节点分裂最小样本数，默认2
    :param min_samples_leaf: 叶子节点最小样本数，默认1
    :param max_features: 最大特征数，默认'sqrt'
    :param bootstrap: 是否使用自助采样，默认False
    :param class_weight: 类别权重，默认None
    :param criterion: 分裂标准，默认'gini'
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1
    :param verbose: 是否输出详细信息，默认False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = False,
        class_weight: Optional[Union[str, Dict]] = None,
        criterion: str = 'gini',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            estimator_class=ExtraTreesClassifier,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.criterion = criterion

        # 更新kwargs
        self.kwargs.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'class_weight': class_weight,
            'criterion': criterion,
        })


class GradientBoostingRiskModel(SklearnRiskModel):
    """梯度提升树风控模型.

    基于sklearn的GradientBoostingClassifier封装。

    **参数**

    :param n_estimators: 树的数量，默认100
    :param learning_rate: 学习率，默认0.1
    :param max_depth: 树最大深度，默认3
    :param min_samples_split: 节点分裂最小样本数，默认2
    :param min_samples_leaf: 叶子节点最小样本数，默认1
    :param subsample: 样本采样比例，默认1.0
    :param max_features: 最大特征数，默认None
    :param criterion: 分裂标准，默认'friedman_mse'
    :param random_state: 随机种子，默认None
    :param verbose: 是否输出详细信息，默认False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        subsample: float = 1.0,
        max_features: Optional[Union[str, int, float]] = None,
        criterion: str = 'friedman_mse',
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            estimator_class=GradientBoostingClassifier,
            random_state=random_state,
            n_jobs=1,  # GBT不支持n_jobs
            verbose=verbose,
            **kwargs
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.criterion = criterion
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        # 更新kwargs
        self.kwargs.update({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'max_features': max_features,
            'criterion': criterion,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
        })

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'GradientBoostingRiskModel':
        """训练模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: fit(X) 在init中指定target
        """
        result = super().fit(X, y, sample_weight, eval_set, **fit_params)

        # 保存训练过程中的损失
        if hasattr(self._model, 'train_score_'):
            self._evals_result['train'] = {'loss': self._model.train_score_}
        if hasattr(self._model, 'validation_score_') and self._model.validation_score_:
            self._evals_result['validation'] = {'loss': self._model.validation_score_}

        # 最佳迭代次数
        if hasattr(self._model, 'n_estimators_'):
            self._best_iteration = self._model.n_estimators_

        return result
