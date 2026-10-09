"""零重要性筛选器（Null Importance）.

使用实际重要性与随机目标下的 null 重要性占比之差识别真正有价值的特征。

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
...     threshold=0.0,  # 实际重要性占比-null重要性占比>0才保留
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
from ._selection_history import initialize_history, record_event
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
    n_splits = cv.get_n_splits(X_ordered, y_ordered)
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
    对实际重要性和 null 重要性分别取各折、各次实验的均值，
    再分别除以各自的特征重要性总和，得到 0～1 的占比。
    特征得分为实际重要性占比减去 null 重要性占比；总重要性为 0 时占比全为 0。
    ``actual_importance_runs_`` / ``null_importance_runs_`` 为有界诊断前缀；
    截断数量见 ``importance_runs_truncated_``，最终重要性始终使用全部实验。

    **参数**

    :param estimator: 评估器
    :param threshold: 阈值，默认为0.0
        - 保留 ``实际重要性% - Null重要性% > threshold`` 的特征
        - 阈值使用 0～1 的占比尺度，例如 0.01 表示实际占比需比 null 占比高出 1 个百分点
    :param cv: 交叉验证折数，默认为5
    :param n_runs: 置换次数，默认为5
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'

    **属性**

    :ivar actual_importances_: 各特征的原始实际重要性均值
    :ivar null_importances_: 各特征的原始 null 重要性均值
    :ivar scores_: 归一化后的实际重要性占比与 null 重要性占比之差
    :ivar importance_details_: 包含原始重要性、两列重要性占比和特征得分的明细表；
        ``实际重要性%``、``Null重要性%`` 为 0～1 的数值，例如 0.25 表示 25%

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
    再以 ``实际重要性% - Null重要性%`` 判断特征是否显著优于随机，能有效剔除高基数/噪声特征的
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
        target_rm: bool = False,
        report_history: str = "summary",
        max_report_events: int = 10000,
        max_report_bytes: int = 8 * 1024 * 1024,
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
            target_rm=target_rm,
        )
        self.estimator = estimator
        self.cv = cv
        self.n_runs = n_runs
        self.random_state = random_state
        self.report_history = report_history
        self.max_report_events = max_report_events
        self.max_report_bytes = max_report_bytes

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合零重要性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        initialize_history(self)
        if (
            isinstance(self.n_runs, (bool, np.bool_))
            or not isinstance(self.n_runs, (int, np.integer))
            or self.n_runs < 1
        ):
            raise ValueError("n_runs 必须为正整数")
        if (
            isinstance(self.threshold, (bool, np.bool_))
            or not isinstance(self.threshold, (int, float, np.number))
            or not np.isfinite(self.threshold)
        ):
            raise ValueError("Null Importance 阈值必须为有限数值")
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
        n_splits = cv.get_n_splits(X, y)
        if n_splits < 1:
            raise ValueError("交叉验证至少需要一个有效拆分")

        # 计算实际标签下的重要性和 shuffle 目标后的 null 重要性。
        total_fits = n_splits * self.n_runs
        retained_fits = min(
            total_fits, self.max_report_events // max(1, n_features), self.max_report_bytes // (16 * max(1, n_features))
        )
        actual_importances = np.zeros((n_features, retained_fits))
        null_importances = np.zeros((n_features, retained_fits))
        actual_sum = np.zeros(n_features)
        null_sum = np.zeros(n_features)

        if self.random_state is None:
            base_seed = int(check_random_state(None).randint(0, np.iinfo(np.int32).max))
        else:
            base_seed = int(self.random_state)
        max_seed = np.iinfo(np.int32).max
        tasks = [(run, (base_seed + run) % max_seed, self.estimator, X, y, self.cv) for run in range(self.n_runs)]
        estimator_params = self.estimator.get_params(deep=True) if hasattr(self.estimator, "get_params") else {}
        worker_aliases = {"n_jobs", "thread_count", "num_workers"}
        has_parallel_children = any(
            name.rsplit("__", 1)[-1] in worker_aliases and isinstance(value, (int, np.integer)) and value not in (0, 1)
            for name, value in estimator_params.items()
        )
        cv_cost = float(self.cv) if isinstance(self.cv, (int, np.integer)) else 5.0
        # 实验分批归并：最终均值利用全部实验，诊断矩阵仅保留明确上限内的前缀。
        for batch_start in range(0, self.n_runs, 32):
            batch = tasks[batch_start : batch_start + 32]
            results = self._parallel_execute(
                _run_null_importance_experiment,
                batch,
                task_labels=[f"实验{task[0] + 1}" for task in batch],
                has_parallel_children=has_parallel_children,
                default_backend="loky",
                workload=ParallelWorkload(
                    task_count=len(batch),
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
                if (
                    actual.shape != (n_features, n_splits)
                    or null.shape != actual.shape
                    or not np.isfinite(actual).all()
                    or not np.isfinite(null).all()
                ):
                    raise ValueError("Null Importance 实验返回的模型重要性形状或数值无效")
                actual_sum += actual.sum(axis=1)
                null_sum += null.sum(axis=1)
                start = n_splits * run
                stop = min(start + n_splits, retained_fits)
                if stop > start:
                    actual_importances[:, start:stop] = actual[:, : stop - start]
                    null_importances[:, start:stop] = null[:, : stop - start]
                for position, feature in enumerate(X.columns):
                    record_event(
                        self,
                        {
                            "轮次": run + 1,
                            "特征": feature,
                            "动作": "实际与随机重要性实验",
                            "指标名称": "实际重要性",
                            "指标值": float(actual[position].mean()),
                            "是否有效": True,
                            "原因": "本轮交叉验证训练折均值",
                            "补充信息": {"随机标签重要性": float(null[position].mean())},
                        },
                        diagnostic=True,
                    )

        actual_mean = actual_sum / total_fits
        null_mean = null_sum / total_fits
        self.importance_runs_total_ = total_fits
        self.importance_runs_stored_ = retained_fits
        self.importance_runs_truncated_ = total_fits - retained_fits
        actual_total = actual_mean.sum()
        null_total = null_mean.sum()
        actual_pct = actual_mean / actual_total if actual_total != 0 else np.zeros_like(actual_mean)
        null_pct = null_mean / null_total if null_total != 0 else np.zeros_like(null_mean)
        scores = actual_pct - null_pct

        self.actual_importances_ = pd.Series(actual_mean, index=X.columns)
        self.null_importances_ = pd.Series(null_mean, index=X.columns)
        self.scores_ = pd.Series(scores, index=X.columns)
        if not np.isfinite(scores).all():
            raise ValueError("Null Importance 模型重要性产生非有限得分")
        self.score_name_ = "实际与随机重要性占比差"
        self.score_direction_ = "越大越好"
        self.effective_threshold_ = self.threshold
        self.selection_stopping_reason_ = "完成全部实际与随机标签实验"
        self.actual_importance_runs_ = pd.DataFrame(actual_importances.T, columns=X.columns)
        self.null_importance_runs_ = pd.DataFrame(null_importances.T, columns=X.columns)
        self.importance_details_ = pd.DataFrame(
            {
                "特征": X.columns,
                "实际重要性": actual_mean,
                "Null重要性": null_mean,
                "实际重要性%": actual_pct,
                "Null重要性%": null_pct,
                "特征得分": scores,
            }
        )

        # 筛选
        selected_mask = scores > self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f"实际重要性%-Null重要性% <= {self.threshold}"
        for feature, value in self.scores_.items():
            record_event(
                self,
                {
                    "轮次": self.n_runs,
                    "特征": feature,
                    "动作": "保留" if feature in self.selected_features_ else "剔除",
                    "指标名称": self.score_name_,
                    "指标值": value,
                    "有效阈值": self.threshold,
                    "是否有效": True,
                    "原因": "占比差严格大于阈值" if value > self.threshold else self._drop_reason,
                },
            )

        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            details = self.importance_details_.set_index("特征")
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [self._drop_reason] * len(dropped_cols),
                    "实际重要性": [details.loc[col, "实际重要性"] for col in dropped_cols],
                    "Null重要性": [details.loc[col, "Null重要性"] for col in dropped_cols],
                    "实际重要性%": [details.loc[col, "实际重要性%"] for col in dropped_cols],
                    "Null重要性%": [details.loc[col, "Null重要性%"] for col in dropped_cols],
                    "特征得分": [details.loc[col, "特征得分"] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(
                columns=[
                    "特征",
                    "剔除原因",
                    "实际重要性",
                    "Null重要性",
                    "实际重要性%",
                    "Null重要性%",
                    "特征得分",
                    "阈值",
                ]
            )

    def get_importance_details(self) -> pd.DataFrame:
        """获取原始重要性、重要性占比和占比差值得分明细。

        :returns: 包含 ``特征``、``实际重要性``、``Null重要性``、``实际重要性%``、
            ``Null重要性%``、``特征得分`` 的 DataFrame；百分比列为 0～1 的数值，得分为两列占比之差
        """
        if not hasattr(self, "importance_details_"):
            return pd.DataFrame(columns=["特征", "实际重要性", "Null重要性", "实际重要性%", "Null重要性%", "特征得分"])
        return self.importance_details_.copy()
