"""Boruta特征筛选器.

使用Boruta算法进行特征选择。

**参考样例**

>>> from hscredit.core.selectors import BorutaSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])  # 10个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = BorutaSelector(
...     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)  # Boruta需传入基模型
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseFeatureSelector, get_feature_importances


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

    **参考样例**

    ::

        >>> from hscredit.core.selectors import BorutaSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = BorutaSelector(
        ...     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    Boruta 是 all-relevant（保留所有相关特征）而非 minimal-optimal 的方法，倾向于多保留
    特征；计算量为 ``max_iter × n_estimators``，特征数多时较慢。

    **引用**

    Kursa, M. B., & Rudnicki, W. R. (2010). *Feature Selection with the Boruta
    Package.* Journal of Statistical Software, 36(11).
    https://doi.org/10.18637/jss.v036.i11
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
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target=target, include=include, exclude=exclude,
            force_drop=force_drop, n_jobs=n_jobs,
            binner=binner, binning_params=binning_params,
            parallel_backend=parallel_backend, parallel_config=parallel_config,
        )
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.method_name = 'Boruta筛选'
        
        # 默认使用随机森林

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合Boruta筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # 准备数据：编码类别变量并填充缺失值（基模型如随机森林不接受 object/NaN），
        # 保持与 chi2/mutual_info/f_test 等筛选器对原始信贷数据的鲁棒性一致
        X_prepared = X.copy()
        for col in X_prepared.columns:
            if X_prepared[col].dtype == 'object':
                X_prepared[col] = pd.factorize(X_prepared[col])[0]
        if X_prepared.isna().any().any():
            X_prepared = X_prepared.fillna(X_prepared.median(numeric_only=True)).fillna(0)
        X_array = X_prepared.values
        feature_names = X.columns.tolist()

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            base_estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        else:
            base_estimator = self.estimator
        base_estimator = self._clone_estimator_for_parallel(base_estimator)

        # 迭代
        selected = set(range(n_features))
        history = []
        real_importances = np.zeros(n_features)

        for iteration in range(self.max_iter):
            # 每轮重新生成影子特征
            X_shadow = rng.permutation(X_array)
            X_with_shadow = np.hstack([X_array, X_shadow])

            # 训练模型
            model = clone(base_estimator)
            with self._estimator_parallel_context():
                model.fit(X_with_shadow, y)

            # 获取特征重要性（兼容所有模型类型）
            importances = get_feature_importances(model)

            # 分离真实和影子特征重要性
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # 阈值：影子特征的最大重要性
            shadow_max = np.max(shadow_importances) if len(shadow_importances) > 0 else 0.0

            # 记录历史
            history.append({
                'iteration': iteration,
                'selected': len(selected),
                'shadow_max': shadow_max
            })

            # 更新选中特征：简化版，只保留重要性高于影子特征最大值的特征
            selected = {
                index
                for index in selected
                if real_importances[index] > shadow_max
            }

            if len(selected) == 0:
                break

        # 选中特征
        self.selected_features_ = [feature_names[i] for i in sorted(selected)]

        # 计算得分
        self.scores_ = pd.Series(real_importances, index=feature_names)
        self._drop_reason = '重要性低于影子特征'
