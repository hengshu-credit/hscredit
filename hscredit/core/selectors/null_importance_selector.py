"""零重要性筛选器（Null Importance）.

使用实际重要性与随机目标下的 null 重要性差值识别真正有价值的特征。

**参考样例**

>>> from hscredit.core.selectors import NullImportanceSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = NullImportanceSelector(
...     RandomForestClassifier(n_estimators=50, random_state=42),  # 传入基模型
...     threshold=0.0,  # 实际重要性-null重要性>0才保留
...     cv=3, n_runs=3  # 交叉验证次数
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv
from sklearn.base import clone
from sklearn.utils import check_random_state

from .base import BaseFeatureSelector, _set_estimator_parallel_budget, get_feature_importances
from ...utils.parallel import ParallelWorkload, _current_parallel_budget


def _fit_importance_model(estimator, X, y):
    """克隆并拟合单个重要性模型，遵守当前子预算。"""
    model = clone(estimator)
    model = _set_estimator_parallel_budget(model, _current_parallel_budget().available)
    model.fit(X, y)
    return get_feature_importances(model)


def _run_null_importance_experiment(task):
    """执行一个独立的实际/null 重要性实验。"""
    ordinal, seed, estimator, X, y, cv_spec = task
    rng = check_random_state(seed)
    order = rng.permutation(len(X))
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y[order]
    cv = check_cv(cv_spec, y_ordered, classifier=True)
    n_splits = cv.get_n_splits()
    actual = np.zeros((X.shape[1], n_splits))
    null = np.zeros((X.shape[1], n_splits))

    for fold_idx, (train_idx, _) in enumerate(cv.split(X_ordered, y_ordered)):
        actual[:, fold_idx] = _fit_importance_model(
            estimator,
            X_ordered.iloc[train_idx],
            y_ordered[train_idx],
        )

    y_null = rng.permutation(y_ordered)
    cv_null = check_cv(cv_spec, y_null, classifier=True)
    for fold_idx, (train_idx, _) in enumerate(cv_null.split(X_ordered, y_null)):
        null[:, fold_idx] = _fit_importance_model(
            estimator,
            X_ordered.iloc[train_idx],
            y_null[train_idx],
        )

    return ordinal, actual, null


class NullImportanceSelector(BaseFeatureSelector):
    """零重要性筛选器.

    使用 null importance 识别真正有价值的特征。
    通过多次 shuffle 目标变量得到随机情况下的 null 重要性，
    再用实际重要性减去 null 重要性作为特征得分。

    **参数**

    :param estimator: 评估器
    :param threshold: 阈值，默认为0.0
        - 保留 ``实际重要性 - null重要性 > threshold`` 的特征
    :param cv: 交叉验证折数，默认为5
    :param n_runs: 置换次数，默认为5
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'

    **参考样例**

    ::

        >>> from hscredit.core.selectors import NullImportanceSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = NullImportanceSelector(
        ...     RandomForestClassifier(n_estimators=50, random_state=42),
        ...     threshold=0.0, cv=3, n_runs=3
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    本方法通过多次打乱**目标变量**得到"零假设"下的重要性分布（null importances），
    再以 ``实际重要性 - null重要性`` 判断特征是否显著优于随机，能有效剔除高基数/噪声特征的
    虚高重要性。计算量为 ``n_runs × cv`` 次模型训练。

    **引用**

    Altmann, A. et al. (2010). *Permutation importance: a corrected feature
    importance measure.* Bioinformatics, 26(10).
    https://doi.org/10.1093/bioinformatics/btq134
    """

    method_name = "零重要性筛选"

    def __init__(
        self,
        estimator,
        threshold: float = 0.0,
        cv: int = 5,
        n_runs: int = 5,
        random_state: Optional[int] = 42,
        target: str = "target",
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
            target=target,
            threshold=threshold,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.estimator = estimator
        self.cv = cv
        self.n_runs = n_runs
        self.random_state = random_state

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

        # 确保 y 是 numpy 数组（base.fit 传入的可能是 Series，索引不连续会导致 y[idx] KeyError）
        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.asarray(y)

        # 重置 DataFrame 索引以确保 iloc 与 positional index 一致
        X = X.reset_index(drop=True)

        self._get_feature_names(X)

        cv = check_cv(self.cv, y, classifier=True)

        n_samples, n_features = X.shape
        n_splits = cv.get_n_splits()

        # 计算实际标签下的重要性和 shuffle 目标后的 null 重要性。
        actual_importances = np.zeros((n_features, n_splits * self.n_runs))
        null_importances = np.zeros((n_features, n_splits * self.n_runs))

        if self.random_state is None:
            base_seed = int(check_random_state(None).randint(0, np.iinfo(np.int32).max))
        else:
            base_seed = int(self.random_state)
        max_seed = np.iinfo(np.int32).max
        tasks = [(run, (base_seed + run) % max_seed, self.estimator, X, y, self.cv) for run in range(self.n_runs)]
        estimator_params = self.estimator.get_params(deep=True) if hasattr(self.estimator, "get_params") else {}
        worker_aliases = {"n_jobs", "thread_count", "num_workers"}
        has_parallel_children = any(
            name.rsplit("__", 1)[-1] in worker_aliases
            and isinstance(value, (int, np.integer))
            and value not in (0, 1)
            for name, value in estimator_params.items()
        )
        cv_cost = float(self.cv) if isinstance(self.cv, (int, np.integer)) else 5.0
        results = self._parallel_execute(
            _run_null_importance_experiment,
            tasks,
            task_labels=[f"实验{run + 1}" for run in range(self.n_runs)],
            has_parallel_children=has_parallel_children,
            default_backend="loky",
            workload=ParallelWorkload(
                task_count=self.n_runs,
                rows=len(X),
                columns=X.shape[1],
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=max(10.0, cv_cost * 10.0),
                capability="process_safe",
                has_parallel_children=has_parallel_children,
                operation="Null Importance重复实验",
            ),
        )
        for run, actual, null in results:
            start = n_splits * run
            stop = start + n_splits
            actual_importances[:, start:stop] = actual
            null_importances[:, start:stop] = null

        actual_mean = actual_importances.mean(axis=1)
        null_mean = null_importances.mean(axis=1)
        scores = actual_mean - null_mean

        self.actual_importances_ = pd.Series(actual_mean, index=X.columns)
        self.null_importances_ = pd.Series(null_mean, index=X.columns)
        self.scores_ = pd.Series(scores, index=X.columns)
        self.actual_importance_runs_ = pd.DataFrame(actual_importances.T, columns=X.columns)
        self.null_importance_runs_ = pd.DataFrame(null_importances.T, columns=X.columns)
        self.importance_details_ = pd.DataFrame(
            {
                "特征": X.columns,
                "实际重要性": actual_mean,
                "Null重要性": null_mean,
                "特征得分": scores,
            }
        )

        # 筛选
        selected_mask = scores > self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f"实际重要性-Null重要性 <= {self.threshold}"

        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            details = self.importance_details_.set_index("特征")
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [self._drop_reason] * len(dropped_cols),
                    "实际重要性": [details.loc[col, "实际重要性"] for col in dropped_cols],
                    "Null重要性": [details.loc[col, "Null重要性"] for col in dropped_cols],
                    "特征得分": [details.loc[col, "特征得分"] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "实际重要性", "Null重要性", "特征得分", "阈值"])

    def get_importance_details(self) -> pd.DataFrame:
        """获取实际重要性、Null重要性和差值得分明细。

        :returns: 包含 ``特征``、``实际重要性``、``Null重要性``、``特征得分`` 的 DataFrame
        """
        if not hasattr(self, "importance_details_"):
            return pd.DataFrame(columns=["特征", "实际重要性", "Null重要性", "特征得分"])
        return self.importance_details_.copy()
