"""递归特征消除筛选器.

递归特征消除（Recursive Feature Elimination）通过递归方式逐步剔除
最不重要的特征，直到达到目标数量。适用于任何有 feature_importances_
或 coef_ 属性的模型。基于 sklearn.feature_selection.RFE 实现。

**参考样例**

>>> from hscredit.core.selectors import RFESelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])  # 10个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = RFESelector(
...     RandomForestClassifier(n_estimators=100, random_state=42),
...     n_features_to_select=5  # 递归消除至剩余5个特征
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from .base import BaseFeatureSelector, get_feature_importances
from ._selection_history import initialize_history, record_event


class RFESelector(BaseFeatureSelector):
    """递归特征消除筛选器.

    通过递归方式逐步剔除最不重要的特征。
    适用于任何有feature_importances_或coef_属性的模型。

    **参数**

    :param estimator: 评估器
    :param n_features_to_select: 保留的特征数，默认为10
        - 整数: 保留的特征数量
        - 浮点数: 保留的特征比例
        数量预算包含 include；强制保留字段始终参与候选模型拟合。
    :param step: 每次剔除的特征数，默认为1
    :param target: 目标变量列名，默认为'target'

    **参考样例**

    ::

        >>> from hscredit.core.selectors import RFESelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = RFESelector(
        ...     RandomForestClassifier(n_estimators=100, random_state=42),
        ...     n_features_to_select=5
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **引用**

    递归特征消除（RFE）出自 Guyon, I. et al. (2002). *Gene Selection for Cancer
    Classification using Support Vector Machines.* Machine Learning, 46.
    https://doi.org/10.1023/A:1012487302797 ；实现对齐 sklearn ``RFE``
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    """

    method_name = "RFE筛选"

    def __init__(
        self,
        estimator,
        n_features_to_select: Union[int, float] = 10,
        step: int = 1,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
        report_history: str = "summary",
        max_report_events: int = 10000,
        max_report_bytes: int = 8 * 1024 * 1024,
    ):
        super().__init__(
            target=target,
            threshold=n_features_to_select,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.report_history = report_history
        self.max_report_events = max_report_events
        self.max_report_bytes = max_report_bytes

    def _included_features_participate_in_selection(self) -> bool:
        """固定保留变量参与每轮模型，数量包含在总保留预算内。"""
        return True

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合RFE筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        initialize_history(self)
        value = self.n_features_to_select
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("n_features_to_select 不能是布尔值")
        if isinstance(value, (int, np.integer)) and value > 0:
            budget = min(int(value), X.shape[1])
        elif isinstance(value, (float, np.floating)) and 0 < value <= 1:
            budget = max(1, int(X.shape[1] * value))
        else:
            raise ValueError("n_features_to_select 必须为正整数或 (0, 1] 范围内的比例")
        forced = set(self.include_).intersection(X.columns)
        if len(forced) > budget:
            raise ValueError("include 特征数不能超过 n_features_to_select 总预算")
        if isinstance(self.step, (bool, np.bool_)):
            raise ValueError("step 不能是布尔值")
        if isinstance(self.step, (int, np.integer)) and self.step > 0:
            step = int(self.step)
        elif isinstance(self.step, (float, np.floating)) and 0 < self.step < 1:
            step = max(1, int(self.step * X.shape[1]))
        else:
            raise ValueError("step 必须为正整数或 (0, 1) 范围内的比例")

        active = X.columns.tolist()
        ranking = pd.Series(1, index=X.columns, dtype=int)
        fixed_budget = len(forced) == budget
        self.selection_reasons_ = pd.Series(index=X.columns, dtype=object)
        if fixed_budget:
            active = [feature for feature in X.columns if feature in forced]
            ranking = ranking.astype(float)
            ranking.loc[~ranking.index.isin(active)] = np.nan
            self.selection_input_features_ = list(active)
            for feature in X.columns:
                if feature not in forced:
                    self.selection_reasons_[feature] = "总数量预算已由强制保留字段占满，未参与递归筛选"
                    record_event(
                        self,
                        {
                            "轮次": 0,
                            "特征": feature,
                            "动作": "预算剔除",
                            "指标名称": None,
                            "指标值": np.nan,
                            "有效阈值": None,
                            "是否有效": False,
                            "原因": self.selection_reasons_[feature],
                            "补充信息": {"总保留数量预算": budget},
                        },
                    )
        self.elimination_round_ = pd.Series(0, index=X.columns, dtype=int)
        self.decision_importances_ = pd.Series(np.nan, index=X.columns, dtype=float)
        iteration = 0
        while True:
            iteration += 1
            model = self._clone_estimator_for_parallel(self.estimator)
            with self._estimator_parallel_context():
                model.fit(X[active], y)
            importances = np.asarray(get_feature_importances(model), dtype=float).reshape(-1)
            if importances.size != len(active) or not np.isfinite(importances).all():
                raise ValueError("RFE 模型必须返回与当前特征数相等的有限重要性")
            current = pd.Series(importances, index=active)
            final = len(active) <= budget
            removable = [name for name in active if name not in forced]
            # 与 sklearn RFE 一样按重要性平方排序；保留稳定的输入顺序平局规则。
            order = np.argsort(np.square(current.reindex(removable).to_numpy()), kind="stable")
            count = min(step, len(active) - budget)
            removed = [] if final else [removable[index] for index in order[:count]]
            for feature in active:
                decided = final or feature in removed
                record_event(
                    self,
                    {
                        "轮次": iteration,
                        "特征": feature,
                        "动作": "保留" if final else ("剔除" if feature in removed else "候选评估"),
                        "指标名称": "模型特征重要性",
                        "指标值": float(current[feature]),
                        "有效阈值": None,
                        "是否有效": True,
                        "原因": "强制保留" if feature in forced else "递归重要性排名与总数量预算",
                        "补充信息": {"总保留数量预算": budget},
                    },
                    diagnostic=not decided,
                )
                if decided:
                    self.decision_importances_[feature] = current[feature]
            if final:
                self.estimator_ = model
                break
            self.elimination_round_.loc[removed] = iteration
            active = [feature for feature in active if feature not in removed]
            ranking.loc[~ranking.index.isin(active)] += 1

        self.selected_features_ = active
        self.ranking_ = ranking
        self.scores_ = ranking.copy()
        self.score_name_ = "RFE排名"
        self.score_direction_ = "越小越好"
        self.effective_threshold_ = budget
        self.selection_stopping_reason_ = "达到包含强制保留字段的总数量预算"
        self.n_iterations_ = iteration
        self.importance_history_ = pd.DataFrame(self.selection_events_)
        self._drop_reason = (
            "总数量预算已由强制保留字段占满，未参与递归筛选" if fixed_budget else "递归消除后未进入总保留数量预算"
        )
