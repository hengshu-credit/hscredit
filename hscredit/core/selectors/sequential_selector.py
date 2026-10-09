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
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv, cross_val_score

from .base import BaseFeatureSelector, _set_estimator_parallel_budget
from ._selection_history import event_bytes, initialize_history, record_event
from ...utils.parallel import ParallelWorkload, _current_parallel_budget


def _evaluate_sequential_candidate(task):
    """评估当前轮的一个候选子集。"""
    ordinal, candidate, estimator, X, y, selected, direction, scoring, cv = task
    if direction == "forward":
        features = selected + [candidate]
    else:
        features = [feature for feature in selected if feature != candidate]
    model = clone(estimator)
    _set_estimator_parallel_budget(model, _current_parallel_budget().available)
    try:
        fold_scores = np.asarray(
            cross_val_score(
                model,
                X[features],
                y,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                error_score=np.nan,
            ),
            dtype=float,
        )
        valid = bool(fold_scores.size and np.isfinite(fold_scores).all())
        score = float(fold_scores.mean()) if valid else np.nan
        reason = "" if valid else "至少一折评分非有限值，候选不可参与择优"
        return ordinal, candidate, score, fold_scores.tolist(), reason
    except Exception as exc:
        return ordinal, candidate, np.nan, [], f"候选交叉验证失败：{type(exc).__name__}: {str(exc)[:500]}"


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
        数量预算包含 include；强制保留字段始终作为候选模型的条件变量。
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

    method_name = "逐步筛选"

    def __init__(
        self,
        estimator,
        n_features_to_select: Union[int, float, str] = "auto",
        direction: str = "forward",
        scoring: Optional[str] = None,
        cv: int = 5,
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
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.report_history = report_history
        self.max_report_events = max_report_events
        self.max_report_bytes = max_report_bytes

    def _included_features_participate_in_selection(self) -> bool:
        """固定保留变量参与候选模型，并计入总数量预算。"""
        return True

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
        initialize_history(self)

        n_features = X.shape[1]
        if self.n_features_to_select == "auto":
            n_to_select = max(1, n_features // 2)
        elif isinstance(self.n_features_to_select, (bool, np.bool_)):
            raise ValueError("n_features_to_select 不能是布尔值")
        elif isinstance(self.n_features_to_select, (float, np.floating)):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError("n_features_to_select 为浮点数时必须在 (0, 1] 范围内")
            n_to_select = max(1, int(n_features * self.n_features_to_select))
        elif isinstance(self.n_features_to_select, (int, np.integer)):
            n_to_select = int(self.n_features_to_select)
        else:
            raise ValueError("n_features_to_select 必须为 'auto'、正整数或 (0, 1] 比例")
        if not 0 < n_to_select <= n_features:
            raise ValueError("n_features_to_select 必须在有效特征数量范围内")
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction 必须为 'forward' 或 'backward'")

        forced = [feature for feature in X.columns if feature in self.include_]
        if len(forced) > n_to_select:
            raise ValueError("include 特征数不能超过 n_features_to_select 总预算")
        fixed_budget = len(forced) == n_to_select
        selected = list(forced) if self.direction == "forward" or fixed_budget else X.columns.tolist()
        needs_search = len(selected) != n_to_select
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        # 固定同一组拆分，避免随机 splitter 每个候选重新抽样而失去可比性。
        splits = list(cv.split(X, y)) if needs_search else []
        if needs_search and not splits:
            raise ValueError("交叉验证至少需要一个有效拆分")
        self.n_cv_splits_ = len(splits)
        self.selection_reasons_ = pd.Series(index=X.columns, dtype=object)
        if not needs_search:
            self.selection_input_features_ = []
            for feature in X.columns:
                self.selection_reasons_[feature] = (
                    "强制保留"
                    if feature in forced
                    else (
                        "总数量预算已由强制保留字段占满，未执行交叉验证搜索"
                        if fixed_budget
                        else "总数量预算保留全部字段，无需交叉验证搜索"
                    )
                )
        self.selection_history_ = []
        self.selection_history_bytes_ = 0
        self.selection_history_truncated_ = 0
        self.candidate_scores_ = pd.Series(np.nan, index=X.columns, dtype=float)
        self.candidate_failures_ = 0
        self.score_name_ = "加入该特征后子集CV评分" if self.direction == "forward" else "移除该特征后子集CV评分"
        self.score_direction_ = "候选子集评分越大越好"
        self.effective_threshold_ = n_to_select
        iteration = 0
        while len(selected) < n_to_select if self.direction == "forward" else len(selected) > n_to_select:
            candidates = (
                [feature for feature in X.columns if feature not in selected]
                if self.direction == "forward"
                else [feature for feature in selected if feature not in forced]
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
                    splits,
                )
                for ordinal, candidate in enumerate(candidates)
            ]
            results = self._parallel_execute(
                _evaluate_sequential_candidate,
                tasks,
                task_labels=candidates,
                has_parallel_children=True,
                default_backend="loky",
                workload=ParallelWorkload(
                    task_count=len(tasks),
                    rows=len(X),
                    columns=max(1, len(selected) + 1),
                    data_bytes=int(X.memory_usage(deep=True).sum()),
                    cost_per_item=max(10.0, float(len(splits)) * 10.0),
                    capability="process_safe",
                    has_parallel_children=True,
                    operation="逐步筛选候选交叉验证",
                ),
            )
            iteration += 1
            valid_results = []
            for result in results:
                _, feature, score, folds, reason = result
                valid = bool(np.isfinite(score))
                self.candidate_scores_[feature] = score
                self.candidate_failures_ += int(not valid)
                event = {
                    "轮次": iteration,
                    "特征": feature,
                    "动作": "候选评估",
                    "指标名称": self.score_name_,
                    "指标值": score,
                    "有效阈值": None,
                    "是否有效": valid,
                    "原因": reason,
                    "补充信息": {"总保留数量预算": n_to_select},
                }
                if self.report_history == "full":
                    subset = (
                        selected + [feature]
                        if self.direction == "forward"
                        else [name for name in selected if name != feature]
                    )
                    event["补充信息"].update({"候选子集": subset, "折评分": folds})
                record_event(self, event, diagnostic=valid)
                if valid:
                    valid_results.append(result)
            if not valid_results:
                reasons = "; ".join(result[4] for result in results[:3])
                raise ValueError(f"第 {iteration} 轮没有有效的交叉验证候选：{reasons}")
            # 有序候选和严格比较保留输入顺序平局规则，NaN 不参与择优。
            _, best_feature, best_score, _, _ = max(valid_results, key=lambda item: item[2])
            if self.direction == "forward":
                selected.append(best_feature)
                action = "add"
            else:
                selected.remove(best_feature)
                action = "remove"
            history_item = {"轮次": iteration, "动作": action, "特征": best_feature, "得分": best_score}
            history_bytes = event_bytes(history_item, self.max_report_bytes)
            if (
                len(self.selection_history_) < self.max_report_events
                and self.selection_history_bytes_ + history_bytes <= self.max_report_bytes
            ):
                self.selection_history_.append(history_item)
                self.selection_history_bytes_ += history_bytes
            else:
                self.selection_history_truncated_ += 1
            record_event(
                self,
                {
                    "轮次": iteration,
                    "特征": best_feature,
                    "动作": "加入" if self.direction == "forward" else "剔除",
                    "指标名称": self.score_name_,
                    "指标值": best_score,
                    "有效阈值": None,
                    "是否有效": True,
                    "原因": "本轮有效候选中评分最高",
                    "补充信息": {"总保留数量预算": n_to_select},
                },
            )

        selected_mask = X.columns.isin(selected)
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.scores_ = self.candidate_scores_.copy()
        self.selection_stopping_reason_ = "达到包含强制保留字段的总数量预算"
        self.n_iterations_ = iteration
        self._drop_reason = "未进入交叉验证搜索的最终特征子集"
