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

from typing import Union, List, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.base import clone

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
    """

    def __init__(
        self,
        estimator,
        threshold: Union[float, int] = 0.0,
        importance_getter: str = 'auto',
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
        self.estimator = estimator
        self.importance_getter = importance_getter
        self.method_name = '特征重要性筛选'

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

        # 克隆并训练模型
        model = clone(self.estimator)
        model.fit(X, y)

        # 获取特征重要性（兼容所有模型类型）
        importances = get_feature_importances(model)
        self.scores_ = pd.Series(importances, index=X.columns)

        # 根据阈值筛选
        if isinstance(self.threshold, int):
            # 保留top-k
            top_k = min(self.threshold, len(X.columns))
            selected_idx = np.argsort(importances)[-top_k:]
            selected_cols = X.columns[selected_idx].tolist()
        else:
            # 保留重要性 >= threshold
            selected_mask = importances >= self.threshold
            selected_cols = X.columns[selected_mask].tolist()

        self.selected_features_ = selected_cols
        self._drop_reason = f'特征重要性 < {self.threshold}'
