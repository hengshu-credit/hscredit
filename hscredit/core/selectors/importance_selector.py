"""特征重要性筛选器.

使用模型特征重要性筛选特征。

**参考样例**

>>> from hscredit.core.selectors import FeatureImportanceSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = np.random.randint(0, 2, 100)  # 目标变量
>>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
>>> selector = FeatureImportanceSelector(rf, threshold=0.1)  # 保留重要性>0.1的特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from operator import attrgetter
from typing import Union, List, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
from .base import BaseFeatureSelector, get_feature_importances


class FeatureImportanceSelector(BaseFeatureSelector):
    """特征重要性筛选器.

    使用模型的特征重要性进行筛选。
    支持任意有feature_importances_或coef_属性的模型，
    以及原生xgboost/lightgbm/catboost模型。

    **参数**

    :param estimator: 评估器
        - 树模型: RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier
        - 线性模型: LogisticRegression, LinearSVC 等
        - hscredit模型: XGBoostRiskModel, LightGBMRiskModel, CatBoostRiskModel 等
    :param threshold: 重要性阈值或保留特征数
        - 浮点数: 保留重要性 >= threshold的特征
        - 整数: 保留top-k个特征
    :param importance_getter: 重要性获取方式，默认为'auto'
    :param target: 目标变量列名，默认为'target'

    **参考样例**

    ::

        >>> from hscredit.core.selectors import FeatureImportanceSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = np.random.randint(0, 2, 100)
        >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> selector = FeatureImportanceSelector(rf, threshold=0.1)
        >>> selector.fit(X, y)

    **注意**

    重要性来源依模型而定（树模型为 ``feature_importances_``，线性模型为 ``|coef_|``），
    见 :func:`~hscredit.core.selectors.base.get_feature_importances`；不同来源不可直接横向
    比较。基于 impurity 的树重要性对高基数特征有偏，必要时配合
    :class:`NullImportanceSelector` 校正。

    **引用**

    嵌入式重要性筛选对齐 sklearn ``SelectFromModel``：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
    """

    method_name = '特征重要性筛选'

    def __init__(
        self,
        estimator,
        threshold: Union[float, int] = 0.0,
        importance_getter: Union[str, Callable[[Any], Any]] = 'auto',
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
            target=target, threshold=threshold, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
            binner=binner, binning_params=binning_params,
            parallel_backend=parallel_backend, parallel_config=parallel_config,
        )
        self.estimator = estimator
        self.importance_getter = importance_getter

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合特征重要性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        if isinstance(self.threshold, (bool, np.bool_)):
            raise ValueError("threshold 不能是布尔值")
        if isinstance(self.threshold, (int, np.integer)) and not isinstance(self.threshold, (bool, np.bool_)):
            if int(self.threshold) <= 0:
                raise ValueError("top-k 特征数必须大于 0")

        # 克隆并训练模型；调用者 estimator 保持不变。
        model = self._clone_estimator_for_parallel(self.estimator)
        with self._estimator_parallel_context():
            model.fit(X, y)

        # 获取特征重要性（兼容所有模型类型）
        if self.importance_getter == "auto":
            importances = np.asarray(get_feature_importances(model), dtype=float)
        elif callable(self.importance_getter):
            importances = np.asarray(self.importance_getter(model), dtype=float)
        elif isinstance(self.importance_getter, str):
            try:
                importances = np.asarray(attrgetter(self.importance_getter)(model), dtype=float)
            except AttributeError as exc:
                raise ValueError(f"importance_getter 无法读取属性 '{self.importance_getter}'") from exc
            if importances.ndim > 1:
                importances = np.linalg.norm(importances, axis=0)
            else:
                importances = np.abs(importances)
        else:
            raise ValueError("importance_getter 必须为 'auto'、属性路径或可调用对象")
        importances = np.ravel(importances)
        if importances.shape[0] != X.shape[1]:
            raise ValueError(f"重要性数量 {importances.shape[0]} 与特征数 {X.shape[1]} 不一致")
        self.scores_ = pd.Series(importances, index=X.columns)

        # 根据阈值筛选
        if isinstance(self.threshold, (int, np.integer)) and not isinstance(self.threshold, (bool, np.bool_)):
            # 保留top-k
            top_k = min(int(self.threshold), len(X.columns))
            ranking = np.argsort(-importances, kind="stable")
            selected_mask = np.zeros(len(X.columns), dtype=bool)
            selected_mask[ranking[:top_k]] = True
            selected_cols = X.columns[selected_mask].tolist()
        else:
            # 保留重要性 >= threshold
            selected_mask = importances >= self.threshold
            selected_cols = X.columns[selected_mask].tolist()

        self.selected_features_ = selected_cols
        self._drop_reason = f'特征重要性 < {self.threshold}'
