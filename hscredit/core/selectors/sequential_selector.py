"""逐步特征筛选器.

使用前向逐步选择或后向逐步消除搜索最优特征子集。
前向选择从空集开始逐步添加最有价值的特征；
后向消除从全特征集开始逐步剔除最无价值的特征。
基于 sklearn.feature_selection.SequentialFeatureSelector 实现。

**参考样例**

>>> from hscredit.core.selectors import SequentialFeatureSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])  # 10个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = SequentialFeatureSelector(
...     RandomForestClassifier(n_estimators=50, random_state=42),
...     n_features_to_select=5,  # 选择5个最优特征
...     direction='forward',    # 前向选择（从空集开始逐步加入）
...     cv=3                     # 3折交叉验证评估
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from .base import BaseFeatureSelector
from ...utils.parallel import _current_parallel_budget


def _evaluate_sequential_candidate(task):
    """评估当前轮的一个候选子集。"""
    ordinal, candidate, estimator, X, y, selected, direction, scoring, cv = task
    if direction == 'forward':
        features = selected + [candidate]
    else:
        features = [feature for feature in selected if feature != candidate]
    model = clone(estimator)
    if 'n_jobs' in model.get_params(deep=False):
        model.set_params(n_jobs=_current_parallel_budget().available)
    score = cross_val_score(
        model,
        X[features],
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
    ).mean()
    return ordinal, candidate, score


class SequentialFeatureSelector(BaseFeatureSelector):
    """逐步特征筛选器.

    使用前向或后向逐步选择选择最优特征子集。
    前向选择：从空集开始，逐步添加最有价值的特征
    后向消除：从所有特征开始，逐步剔除最无价值的特征

    **参数**

    :param estimator: 评估器
    :param n_features_to_select: 保留的特征数，默认为'auto'
        - 'auto': 保留一半特征
        - 整数: 保留的特征数量
        - 浮点数: 保留的特征比例
    :param direction: 方向，默认为'forward'
        - 'forward': 前向选择
        - 'backward': 后向消除
    :param scoring: 评分指标，默认为None
    :param cv: 交叉验证折数，默认为5
    :param target: 目标变量列名，默认为'target'

    **参考样例**

    ::

        >>> from hscredit.core.selectors import SequentialFeatureSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = SequentialFeatureSelector(
        ...     RandomForestClassifier(n_estimators=50, random_state=42),
        ...     n_features_to_select=5,
        ...     direction='forward',
        ...     cv=3
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    与 :class:`RFESelector` 不同，本类基于交叉验证评分而非模型权重逐个增删特征，更稳健但
    更耗时（约 ``n_features × cv`` 次拟合）；与 :class:`StepwiseSelector`（基于 AIC/BIC/KS 等
    统计准则、面向逻辑回归）适用场景亦不同。

    **引用**

    对齐 sklearn ``SequentialFeatureSelector``：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: Union[int, float, str] = 'auto',
        direction: str = 'forward',
        scoring: Optional[str] = None,
        cv: int = 5,
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
            target=target, threshold=n_features_to_select, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
            binner=binner, binning_params=binning_params,
            parallel_backend=parallel_backend, parallel_config=parallel_config,
        )
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.method_name = '逐步筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合逐步筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        n_features = X.shape[1]
        if self.n_features_to_select == 'auto':
            n_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, float):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError("n_features_to_select 为浮点数时必须在 (0, 1] 范围内")
            n_to_select = int(n_features * self.n_features_to_select)
        else:
            n_to_select = int(self.n_features_to_select)
        if not 0 < n_to_select <= n_features:
            raise ValueError("n_features_to_select 必须在有效特征数量范围内")
        if self.direction not in ('forward', 'backward'):
            raise ValueError("direction 必须为 'forward' 或 'backward'")

        selected = [] if self.direction == 'forward' else X.columns.tolist()
        self.selection_history_ = []
        while (
            len(selected) < n_to_select
            if self.direction == 'forward'
            else len(selected) > n_to_select
        ):
            candidates = (
                [feature for feature in X.columns if feature not in selected]
                if self.direction == 'forward'
                else list(selected)
            )
            tasks = [
                (
                    ordinal,
                    candidate,
                    self.estimator,
                    X,
                    np.asarray(y),
                    list(selected),
                    self.direction,
                    self.scoring,
                    self.cv,
                )
                for ordinal, candidate in enumerate(candidates)
            ]
            results = self._parallel_execute(
                _evaluate_sequential_candidate,
                tasks,
                task_labels=candidates,
                has_parallel_children=True,
            )
            # Ordered results + argmax preserve sklearn's first-candidate tie break.
            best_position = int(np.argmax([score for _, _, score in results]))
            _, best_feature, best_score = results[best_position]
            if self.direction == 'forward':
                selected.append(best_feature)
                action = 'add'
            else:
                selected.remove(best_feature)
                action = 'remove'
            self.selection_history_.append(
                {'轮次': len(self.selection_history_) + 1, '动作': action, '特征': best_feature, '得分': best_score}
            )

        selected_mask = X.columns.isin(selected)
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.scores_ = pd.Series(
            selected_mask.astype(int),
            index=X.columns
        )
        self._drop_reason = '未选中'
