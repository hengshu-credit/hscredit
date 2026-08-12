"""相关性筛选器.

移除与其它特征高度相关的特征，保留指标值更优的特征。
参考 scorecardpipeline / toad 的相关性筛选逻辑。

**参考样例**

>>> from hscredit.core.selectors import CorrSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量（用于IV计算）
>>> selector = CorrSelector(threshold=0.7, metric='iv')  # 默认Spearman，相关性>0.7时移除
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

import numbers
from typing import Union, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .base import BaseFeatureSelector
from ...exceptions import ValidationError
from ...utils.parallel import ParallelWorkload, _resolve_current_n_jobs


# bin_tables_ 中指标列名 → 聚合方式的映射
_METRIC_COL_MAP = {
    "iv": ("指标IV值", lambda s: s.iloc[0]),  # 每行相同，取第一个
    "ks": ("分档KS值", "max"),  # 取最大KS
    "lift": ("LIFT值", "max"),  # 取最大LIFT
    "bad_rate": ("坏样本率", "max"),  # 取最大坏样本率
}

DEFAULT_BINNING_PARAMS = {
    "method": "best_iv",
    "max_n_bins": 5,
    "min_bin_size": 0.01,
    "missing_separate": True,
}


def _rank_corr_column(task):
    """按列计算平均秩；线程后端可直接写回当前列。"""
    ordinal, column, *options = task
    ranked = pd.Series(column, copy=False).rank(method="average").to_numpy(dtype=np.float64)
    if options and options[0]:
        column[:] = ranked
        return ordinal, None
    return ordinal, ranked


def _corr_block_worker(task):
    """计算当前候选块与一个已处理块的相关性。"""
    (
        ordinal,
        values,
        row_start,
        row_stop,
        column_start,
        column_stop,
        threshold,
        method,
        fast_matrix,
        kept_rows,
    ) = task

    right = values[:, column_start:column_stop]
    diagonal = row_start == column_start
    if diagonal:
        left = values[:, row_start:row_stop]
        kept_positions = None
    else:
        kept_positions = np.flatnonzero(np.asarray(kept_rows, dtype=bool))
        width = column_stop - column_start
        if kept_positions.size == 0:
            return (
                ordinal,
                "prior",
                column_start,
                np.full(width, np.nan, dtype=np.float64),
                np.full(width, -1, dtype=np.int64),
            )
        left = values[:, row_start:row_stop][:, kept_positions]

    if fast_matrix:
        corr = np.matmul(left.T, right)
        corr = np.clip(corr, -1.0, 1.0)
    else:
        if diagonal:
            corr = pd.DataFrame(left).corr(method=method).to_numpy(dtype=np.float64)
        else:
            combined = np.concatenate((left, right), axis=1)
            combined_corr = pd.DataFrame(combined).corr(method=method).to_numpy(dtype=np.float64)
            corr = combined_corr[: left.shape[1], left.shape[1] :]

    absolute = np.abs(np.asarray(corr, dtype=np.float64))
    if diagonal:
        np.fill_diagonal(absolute, np.nan)
        return ordinal, "diagonal", column_start, absolute, None

    width = column_stop - column_start
    safe = np.where(np.isnan(absolute), -np.inf, absolute)
    positions = np.argmax(safe, axis=0)
    column_positions = np.arange(width)
    max_values = safe[positions, column_positions]
    related_indices = row_start + kept_positions[positions]
    conflicts = np.isfinite(max_values) & (max_values > threshold)
    return (
        ordinal,
        "prior",
        column_start,
        np.where(conflicts, max_values, np.nan).astype(np.float64, copy=False),
        np.where(conflicts, related_indices, -1).astype(np.int64, copy=False),
    )


class CorrSelector(BaseFeatureSelector):
    """相关性筛选器.

    移除与其它特征相关性高于阈值的特征。
    当两个特征高度相关时，保留指定指标（默认 IV）更高的特征。
    特征按指标稳定降序逐步处理，只允许已实际保留的特征淘汰后续特征，
    避免相关链中已删除变量继续造成过度筛除。

    **参数**

    :param threshold: 相关系数阈值，默认为0.7
        - 0.7: 移除与其它特征相关性超过0.7的特征
        - 范围: 0-1之间的浮点数
    :param method: 相关性计算方法，默认为'spearman'
        - 'pearson': 皮尔逊相关系数
        - 'spearman': 斯皮尔曼等级相关系数
        - 'kendall': 肯德尔相关系数
    :param metric: 用于决定保留哪个特征的指标，默认为'iv'
        - 'iv': 信息值（需要目标变量 y）
        - 'ks': KS 统计量
        - 'lift': LIFT 值
        - 'bad_rate': 坏样本率
        指标通过分箱后的 bin_tables_ 计算得到。
    :param weights: 特征权重，用于决定保留哪个特征，默认为None
        如果同时传入 weights 和 metric，weights 优先。
    :param binning_params: 透传给 OptimalBinning 的筛选前分箱参数，例如:
        - method: 分箱方法，默认'best_iv'
        - max_n_bins: 最大分箱数，默认5
        - min_bin_size: 最小分箱比例，默认0.01
        - missing_separate: 是否缺失单独分箱，默认True
        - prebinning: 预分箱方法
        等等，详见 OptimalBinning 文档。
    :param corr_block_size: 相关矩阵分块列数，默认为512。仅控制内存和任务粒度，
        不改变相关系数、筛选阈值或结果顺序。

    **参考样例**

    >>> from hscredit.core.selectors import CorrSelector
    >>> selector = CorrSelector(threshold=0.7, metric='iv')
    >>> selector.fit(X, y)
    >>> print(selector.selected_features_)
    >>>
    >>> # 基于 KS 的相关性筛选
    >>> selector = CorrSelector(threshold=0.7, metric='ks')
    >>> selector.fit(X, y)
    >>>
    >>> # 自定义分箱参数
    >>> selector = CorrSelector(
    ...     threshold=0.7,
    ...     metric='iv',
    ...     binning_params={'method': 'cart', 'max_n_bins': 8}
    ... )
    >>> selector.fit(X, y)
    >>>
    >>> # 使用自定义权重（不做分箱）
    >>> selector = CorrSelector(threshold=0.7, weights=iv_series)
    >>> selector.fit(X)

    **引用**

    无缺失 Pearson 使用数学等价的标准化分块矩阵乘法；含缺失数据及
    Spearman/Kendall 使用 ``pandas.DataFrame.corr`` 的分块兼容路径：
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html ；
    高相关特征对中保留 IV/KS 更高者的原则参考 toad / scorecardpipeline。
    """

    def __init__(
        self,
        threshold: float = 0.7,
        method: str = "spearman",
        metric: str = "iv",
        weights: Optional[Union[pd.Series, Dict[str, float], List[float]]] = None,
        binning_params: Optional[Dict[str, Any]] = DEFAULT_BINNING_PARAMS,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        corr_block_size: int = 512,
    ):
        self._uses_default_binning_params = binning_params is DEFAULT_BINNING_PARAMS or binning_params == DEFAULT_BINNING_PARAMS
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
        self.method = method
        self.metric = metric
        self.weights = weights
        self.corr_block_size = corr_block_size
        self.method_name = "相关性筛选"

    def _validate_corr_block_size(self) -> int:
        """校验并返回相关计算块大小。"""
        if isinstance(self.corr_block_size, (bool, np.bool_)) or not isinstance(self.corr_block_size, numbers.Integral) or int(self.corr_block_size) < 1:
            raise ValidationError("corr_block_size 必须为正整数")
        return int(self.corr_block_size)

    def _resolve_corr_total_workers(self, task_count: int) -> int:
        """解析 Corr 阶段可由外层任务和原生线程共同使用的总预算。"""
        return _resolve_current_n_jobs(self.n_jobs) or 1

    def _rank_corr_values(self, values: np.ndarray, default_backend: str) -> np.ndarray:
        """按输入列顺序分批并行排名，并原地复用 float64 输入缓冲区。"""
        if values.dtype != np.float64:
            values = values.astype(np.float64, copy=False)

        expected_shape = (values.shape[0],)
        effective_backend = self.parallel_backend or default_backend
        if effective_backend == "threading":
            ordinals = list(range(values.shape[1]))
            tasks = [(ordinal, values[:, ordinal], True) for ordinal in ordinals]
            results = self._parallel_execute(
                _rank_corr_column,
                tasks,
                task_labels=ordinals,
                default_backend=default_backend,
                workload=ParallelWorkload(
                    task_count=len(tasks),
                    rows=values.shape[0],
                    columns=values.shape[1],
                    data_bytes=int(values.nbytes),
                    cost_per_item=2.0,
                    capability="thread_safe",
                    releases_gil=True,
                    operation="Spearman字段排名",
                ),
            )
            for expected_ordinal, result in zip(ordinals, results):
                if result != (expected_ordinal, None):
                    raise TypeError(f"相关排名任务 {expected_ordinal} 返回结果无效")
            return values

        batch_size = self._validate_corr_block_size()
        for batch_start in range(0, values.shape[1], batch_size):
            batch_stop = min(batch_start + batch_size, values.shape[1])
            ordinals = list(range(batch_start, batch_stop))
            tasks = [(ordinal, values[:, ordinal]) for ordinal in ordinals]
            results = self._parallel_execute(
                _rank_corr_column,
                tasks,
                task_labels=ordinals,
                default_backend=default_backend,
                workload=ParallelWorkload(
                    task_count=len(tasks),
                    rows=values.shape[0],
                    columns=len(tasks),
                    data_bytes=int(values[:, batch_start:batch_stop].nbytes),
                    cost_per_item=2.0,
                    capability="process_safe",
                    operation="Spearman字段排名",
                ),
            )
            for expected_ordinal, result in zip(ordinals, results):
                if not isinstance(result, tuple) or len(result) != 2 or result[0] != expected_ordinal:
                    raise TypeError(f"相关排名任务 {expected_ordinal} 返回结果无效")
                ranked = np.asarray(result[1], dtype=np.float64)
                if ranked.shape != expected_shape:
                    raise TypeError(f"相关排名任务 {expected_ordinal} 返回形状无效")
                values[:, expected_ordinal] = ranked
        return values

    def _execute_corr_tasks(
        self,
        tasks,
        labels,
        default_backend: str,
        total_workers: int,
    ):
        """在同一总预算内协调相关块 joblib 任务和 BLAS 原生线程。"""
        effective_backend = self.parallel_backend or default_backend
        matrix = tasks[0][1] if tasks else np.empty((0, 0), dtype=float)
        workload = ParallelWorkload(
            task_count=len(tasks),
            rows=matrix.shape[0] if matrix.ndim > 0 else 0,
            columns=matrix.shape[1] if matrix.ndim > 1 else 1,
            data_bytes=int(matrix.nbytes),
            cost_per_item=max(4.0, float(matrix.shape[1] if matrix.ndim > 1 else 1)),
            capability="thread_safe" if effective_backend == "threading" else "process_safe",
            releases_gil=effective_backend == "threading",
            operation=f"{self.method}相关块计算",
        )
        if effective_backend != "threading":
            return self._parallel_execute(
                _corr_block_worker,
                tasks,
                task_labels=labels,
                default_backend=default_backend,
                workload=workload,
            )

        outer_workers = min(total_workers, max(1, len(tasks)))
        native_threads = max(1, total_workers // outer_workers)
        with threadpool_limits(limits=native_threads):
            return self._parallel_execute(
                _corr_block_worker,
                tasks,
                task_labels=labels,
                default_backend=default_backend,
                workload=workload,
            )

    @staticmethod
    def _merge_max_candidates(
        max_values: np.ndarray,
        max_indices: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        others: np.ndarray,
    ) -> None:
        """按完整相关矩阵的列顺序确定性合并最大相关候选。"""
        for index, value, other in zip(indices, values, others):
            index = int(index)
            other = int(other)
            if other < 0 or np.isnan(value):
                continue
            current_value = max_values[index]
            current_other = max_indices[index]
            if np.isnan(current_value) or value > current_value or (value == current_value and (current_other < 0 or other < current_other)):
                max_values[index] = float(value)
                max_indices[index] = other

    def _compute_block_correlations(
        self,
        X: pd.DataFrame,
        sorted_names: List[str],
        forced_keep_count: int = 0,
    ):
        """按 metric 顺序分块筛选，只与已实际保留的特征比较。"""
        block_size = self._validate_corr_block_size()
        n_features = len(sorted_names)
        if n_features == 0:
            return set(), np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

        try:
            values = X.loc[:, sorted_names].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError("CorrSelector 相关计算仅支持可转换为数值的特征") from exc

        fast_matrix = self.method in ("pearson", "spearman") and np.isfinite(values).all()
        has_inner_thread_limit = isinstance(self.parallel_config, dict) and self.parallel_config.get("inner_max_num_threads") is not None
        default_backend = "threading" if fast_matrix and not has_inner_thread_limit else "loky"
        if fast_matrix:
            if self.method == "spearman":
                # 无缺失时，Spearman 等价于先按列进行一次平均秩转换，再计算 Pearson；
                # 排名只做一次，避免每个相关块重复排序。
                values = self._rank_corr_values(values, default_backend)
            values -= values.mean(axis=0)
            norms = np.sqrt(np.einsum("ij,ij->j", values, values))
            with np.errstate(divide="ignore", invalid="ignore"):
                values /= norms
            values[:, norms == 0.0] = np.nan

        blocks = [(start, min(start + block_size, n_features)) for start in range(0, n_features, block_size)]
        kept = np.zeros(n_features, dtype=bool)
        drops = set()
        max_values = np.full(n_features, np.nan, dtype=np.float64)
        max_indices = np.full(n_features, -1, dtype=np.int64)
        total_workers = self._resolve_corr_total_workers(n_features)

        for column_block, (column_start, column_stop) in enumerate(blocks):
            tasks = []
            labels = []
            ordinal = 0
            for row_start, row_stop in blocks[:column_block]:
                kept_rows = kept[row_start:row_stop]
                if not np.any(kept_rows):
                    continue
                tasks.append(
                    (
                        ordinal,
                        values,
                        row_start,
                        row_stop,
                        column_start,
                        column_stop,
                        self.threshold,
                        self.method,
                        fast_matrix,
                        kept_rows.copy(),
                    )
                )
                labels.append(ordinal)
                ordinal += 1
            tasks.append(
                (
                    ordinal,
                    values,
                    column_start,
                    column_stop,
                    column_start,
                    column_stop,
                    self.threshold,
                    self.method,
                    fast_matrix,
                    None,
                )
            )
            labels.append(ordinal)

            results = self._execute_corr_tasks(
                tasks,
                labels,
                default_backend,
                total_workers,
            )

            diagonal = None
            for expected_ordinal, result in enumerate(results):
                if not isinstance(result, tuple) or len(result) != 5 or result[0] != expected_ordinal:
                    raise TypeError(f"相关块任务 {expected_ordinal} 返回结果无效")
                _, result_type, result_start, result_values, result_indices = result
                if result_type == "diagonal":
                    if result_start != column_start or diagonal is not None:
                        raise TypeError(f"相关块任务 {expected_ordinal} 返回对角块无效")
                    diagonal = result_values
                    continue
                if result_type != "prior" or result_start != column_start:
                    raise TypeError(f"相关块任务 {expected_ordinal} 返回前置块无效")
                column_indices = np.arange(column_start, column_stop, dtype=np.int64)
                self._merge_max_candidates(
                    max_values,
                    max_indices,
                    column_indices,
                    result_values,
                    result_indices,
                )

            width = column_stop - column_start
            if diagonal is None or diagonal.shape != (width, width):
                raise TypeError(f"相关块 {column_block} 缺少有效的对角相关矩阵")

            for local_index, feature_index in enumerate(range(column_start, column_stop)):
                if local_index:
                    local_kept = kept[column_start:feature_index]
                    if np.any(local_kept):
                        local_values = diagonal[:local_index, local_index]
                        safe = np.where(local_kept & ~np.isnan(local_values), local_values, -np.inf)
                        related_local = int(np.argmax(safe))
                        related_value = float(safe[related_local])
                        if np.isfinite(related_value) and related_value > self.threshold:
                            self._merge_max_candidates(
                                max_values,
                                max_indices,
                                np.asarray([feature_index], dtype=np.int64),
                                np.asarray([related_value], dtype=np.float64),
                                np.asarray([column_start + related_local], dtype=np.int64),
                            )

                if feature_index < forced_keep_count:
                    max_values[feature_index] = np.nan
                    max_indices[feature_index] = -1
                    kept[feature_index] = True
                elif max_indices[feature_index] >= 0:
                    drops.add(feature_index)
                else:
                    kept[feature_index] = True

        return drops, max_values, max_indices

    def _metric_weights_from_binner(self, feature_names: List[str]) -> pd.Series:
        """从基类管理的同一分箱器中提取特征指标权重。"""
        metric_key = self.metric.lower()
        if metric_key not in _METRIC_COL_MAP:
            raise ValidationError(f"不支持的指标 '{self.metric}'，可选: {list(_METRIC_COL_MAP.keys())}")
        col_name, agg_func = _METRIC_COL_MAP[metric_key]
        binner = getattr(self, "_binner_instance", None)
        bin_tables = getattr(binner, "bin_tables_", {}) if binner is not None else {}
        if not bin_tables:
            raise ValidationError("CorrSelector 使用指标权重时需要配置 binner 或 binning_params，" "也可以显式传入 weights")

        scores = {}
        for col in feature_names:
            if col in bin_tables:
                bt = bin_tables[col]
                if col_name in bt.columns:
                    scores[col] = bt[col_name].agg(agg_func)
                else:
                    scores[col] = 0.0
            else:
                scores[col] = 0.0

        return pd.Series(scores)

    def _should_apply_binner(
        self,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> bool:
        """无目标时仅跳过构造函数提供的默认监督分箱。"""
        if y is None and self.binner is None and self._uses_default_binning_params:
            return False
        return super()._should_apply_binner(y)

    def _resolve_binner(self) -> Optional[Any]:
        """创建指标分箱器，并在未显式覆盖时继承 CorrSelector 的并行配置。"""
        binner = super()._resolve_binner()
        if binner is None or self.binner is not None:
            return binner

        configured = self.binning_params if isinstance(self.binning_params, dict) else {}
        if "n_jobs" not in configured:
            binner.n_jobs = self.n_jobs
        if "parallel_backend" not in configured:
            # BestIV/MDLP 等包含较多 Python 循环；默认使用进程避免 GIL 退化。
            binner.parallel_backend = self.parallel_backend or "loky"
        if "parallel_config" not in configured:
            binner.parallel_config = dict(self.parallel_config) if self.parallel_config is not None else None
        return binner

    def _included_features_participate_in_selection(self) -> bool:
        """强制保留字段仍作为相关性比较基准参与分箱和相关计算。"""
        return True

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合相关性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)
        self._validate_corr_block_size()

        n_features = X.shape[1]
        feature_names = X.columns.tolist()

        # ── 构建权重 ──
        if self.weights is not None:
            # 用户显式传入权重
            if isinstance(self.weights, pd.Series):
                weight_series = self.weights.reindex(feature_names).fillna(0.0)
            elif isinstance(self.weights, dict):
                weight_series = pd.Series(self.weights).reindex(feature_names).fillna(0.0)
            else:
                weight_series = (
                    pd.Series(
                        np.array(self.weights)[:n_features],
                        index=feature_names[: len(self.weights)],
                    )
                    .reindex(feature_names)
                    .fillna(0.0)
                )
        elif y is not None or getattr(self, "_binner_instance", None) is not None:
            # 从基类已训练的同一分箱器中读取指标权重
            weight_series = self._metric_weights_from_binner(feature_names)
            weight_series = weight_series.reindex(feature_names).fillna(0.0)
        else:
            # 无 y 且无 weights，使用等权（退化为按列顺序保留）
            weight_series = pd.Series(np.ones(n_features), index=feature_names)

        try:
            weight_arr = weight_series.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError("CorrSelector 的 weights 或 metric 必须为数值") from exc
        self.feature_scores_ = weight_series.copy()
        self.scores_ = weight_series.copy()

        # ── 强制保留优先，其余特征按权重稳定降序排列 ──
        forced_exclude = set(getattr(self, "exclude_", []))
        forced_include = set(getattr(self, "include_", [])).difference(forced_exclude)
        included_names = [name for name in feature_names if name in forced_include]
        sort_idx = np.argsort(-weight_arr, kind="stable")
        sorted_names = included_names + [feature_names[index] for index in sort_idx if feature_names[index] not in forced_include and feature_names[index] not in forced_exclude]
        forced_keep_count = len(included_names)

        # ── 分块计算相关性，不再分配完整 p×p 相关矩阵 ──
        if forced_keep_count == len(sorted_names):
            drops = set()
            max_corr_by_index = np.full(len(sorted_names), np.nan, dtype=np.float64)
            max_corr_feature_index = np.full(len(sorted_names), -1, dtype=np.int64)
        else:
            drops, max_corr_by_index, max_corr_feature_index = self._compute_block_correlations(
                X,
                sorted_names,
                forced_keep_count=forced_keep_count,
            )

        # ── 获取保留的特征 ──
        keep_idx = [idx for idx in range(len(sorted_names)) if idx not in drops]
        self.selected_features_ = [sorted_names[idx] for idx in keep_idx]

        # 保存 scores（与原始列顺序一致）
        self.scores_ = weight_series

        # ── 构建 dropped_ 报告 ──
        if len(drops) > 0:
            ordered_drops = sorted(drops)
            dropped_cols = [sorted_names[idx] for idx in ordered_drops]
            max_corr_values = []
            max_corr_features = []
            metric_values = []
            for idx in ordered_drops:
                col_name = sorted_names[idx]
                max_corr = max_corr_by_index[idx]
                related_index = int(max_corr_feature_index[idx])
                max_corr_feat = sorted_names[related_index] if related_index >= 0 else None
                max_corr_values.append(max_corr)
                max_corr_features.append(max_corr_feat)
                metric_values.append(weight_series.get(col_name, 0.0))

            metric_label = self.metric.upper() if self.weights is None and getattr(self, "_binner_instance", None) is not None else "权重"
            drop_reasons = []
            for index, col_name in enumerate(dropped_cols):
                related_name = max_corr_features[index]
                if related_name in forced_include:
                    suffix = "相关特征为强制保留变量"
                else:
                    suffix = f"{metric_label}({metric_values[index]:.4f})不高于相关特征"
                drop_reasons.append(f"与{related_name}相关系数({max_corr_values[index]:.4f})>" f"{self.threshold}，{suffix}")

            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": drop_reasons,
                    "最大相关系数": max_corr_values,
                    "相关特征": max_corr_features,
                    metric_label: metric_values,
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "最大相关系数", "相关特征", "阈值"])
