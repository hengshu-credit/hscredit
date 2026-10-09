"""VIF筛选器.

使用方差膨胀因子（VIF）检测和移除多重共线性特征。

**参考样例**

>>> from hscredit.core.selectors import VIFSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [1, 2, 3, 4, 5],  # 与a完全线性相关，VIF会很高
...     'c': [5, 4, 3, 2, 1]
... })
>>> selector = VIFSelector(threshold=4.0)  # VIF>4表示存在多重共线性
>>> selector.fit(X)
>>> print(selector.selected_features_)
['a', 'c']
"""

import logging
from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from .base import BaseFeatureSelector
from ._selection_history import initialize_history, record_event
from ...utils.parallel import ParallelWorkload, resolve_native_workers

logger = logging.getLogger(__name__)


def _compute_vif_single(x: np.ndarray, idx: int) -> float:
    """计算单个特征的VIF值。

    :param x: 特征矩阵
    :param idx: 特征索引
    :return: VIF值
    """
    n_features = x.shape[1]
    if n_features <= 1:
        return 1.0  # 只有一个特征时，VIF=1

    # 获取其他特征
    mask = np.ones(n_features, dtype=bool)
    mask[idx] = False
    x_other = x[:, mask]
    x_target = x[:, idx]

    # 处理缺失值
    valid = ~(np.isnan(x_target) | np.any(np.isnan(x_other), axis=1))
    if valid.sum() < 2:
        return np.inf
    x_other = x_other[valid]
    x_target = x_target[valid]

    # 检查目标特征是否为常数
    if np.std(x_target) < 1e-10:
        return 1.0  # 常数特征VIF=1

    # 检查其他特征是否全为常数
    if np.all(np.std(x_other, axis=0) < 1e-10):
        return 1.0  # 其他特征都是常数，无法预测目标特征

    # 线性回归
    try:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(x_other, x_target)
        y_pred = lr.predict(x_other)

        # 计算VIF
        ss_res = np.sum((x_target - y_pred) ** 2)
        ss_tot = np.sum((x_target - np.mean(x_target)) ** 2)

        if ss_tot < 1e-10:
            return 1.0

        r2 = 1 - ss_res / ss_tot

        # 处理r2接近1或大于1的情况
        if r2 >= 1.0:
            return np.inf
        elif r2 < 0:
            # r2 < 0 表示模型比均值还差，VIF很小
            return 1.0
        else:
            vif = 1 / (1 - r2)
            return vif
    except Exception:
        return np.inf


def _compute_vif_feature(task):
    """计算单个特征 VIF 并携带列位置返回。"""
    idx, x = task
    return idx, _compute_vif_single(x, idx)


def _compute_vif_bulk(task):
    """一次带截距的共享QR；数值边界不可靠时返回None，沿用逐列OLS。

    不对原始列缩放，以免改变sklearn对秩和微小变化量的判断。
    额外工作区为中心化矩阵O(np)和三角矩阵O(p²)，不是p份回归输入。
    """
    x, native_threads = task
    n_features = x.shape[1]
    if n_features <= 1:
        return np.ones(n_features, dtype=float)
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    values = values[~np.isnan(values).any(axis=1)]
    if len(values) < 2:
        return np.full(n_features, np.inf)
    if not np.isfinite(values).all():
        return None
    with threadpool_limits(limits=resolve_native_workers(-1, native_threads)):
        centered = values - values.mean(axis=0)
        ss_tot = np.einsum("ij,ij->j", centered, centered)
        deviations = np.std(values, axis=0)
        nonzero = np.flatnonzero(ss_tot > 0)
        output = np.ones(n_features, dtype=float)
        if len(nonzero) <= 1:
            return output
        if np.any(np.abs(values.mean(axis=0)[nonzero]) > 1e8 * deviations[nonzero]):
            return None
        if len(values) <= len(nonzero):
            return None
        try:
            triangular = np.linalg.qr(centered[:, nonzero], mode="r")
            singular = np.linalg.svd(triangular, compute_uv=False)
            if singular[-1] <= 0 or singular[0] / singular[-1] > 1e7:
                return None
            inverse = solve_triangular(triangular, np.eye(len(nonzero)), lower=False)
            vif = ss_tot[nonzero] * np.einsum("ij,ij->i", inverse, inverse)
            # 在R²舍入敏感区域保留旧OLS数值与平局剔除顺序。
            if not np.isfinite(vif).all() or np.any(vif > 1e8):
                return None
            ratio = 1.0 / np.maximum(vif, 1.0)
            output[nonzero] = 1.0 / (1.0 - (1.0 - ratio))
            constant_targets = (deviations < 1e-10) | (ss_tot < 1e-10)
            other_all_constant = (np.sum(deviations >= 1e-10) - (deviations >= 1e-10)) == 0
            output[constant_targets | other_all_constant] = 1.0
            return output
        except np.linalg.LinAlgError:
            return None


class VIFSelector(BaseFeatureSelector):
    """VIF筛选器.

    使用方差膨胀因子（VIF）检测多重共线性。
    VIF值越高，表示特征与其他特征的多重共线性越严重。
    在金融风控中，通常认为VIF > 4存在多重共线性问题。

    **算法说明：**

    采用迭代剔除策略：
    1. 计算所有特征的VIF值
    2. 如果最大VIF > threshold，剔除VIF最大的特征
    3. 重新计算剩余特征的VIF
    4. 重复步骤2-3，直到所有VIF <= threshold

    这种方法避免了"一刀切"地剔除所有高VIF特征的问题。

    **参数**

    :param threshold: VIF阈值，默认为4.0
        - 4.0: 移除VIF值超过4的特征
        - 范围: 正数
    :param missing: 缺失值填充值，默认为-1
    :param max_iter: 最大迭代次数，默认为100
    :param n_jobs: 并行计算的任务数
    :param verbose: 是否显示详细过程，默认为False

    **参考样例**

    >>> from hscredit.core.selectors import VIFSelector
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     'a': [1, 2, 3, 4, 5],
    ...     'b': [1, 2, 3, 4, 5],  # 与a完全相关
    ...     'c': [5, 4, 3, 2, 1]
    ... })
    >>> selector = VIFSelector(threshold=4.0)
    >>> selector.fit(X)
    >>> print(selector.selected_features_)
    ['a', 'c']  # 或 ['b', 'c']，保留其中一个高相关特征

    **引用**

    方差膨胀因子 ``VIF_i = 1 / (1 - R_i²)``（``R_i²`` 为特征 i 对其余特征回归的判定系数），
    经验上 VIF>5（严格 >10）提示多重共线性。参见 Kutner, M. et al. (2004). *Applied
    Linear Statistical Models*；https://en.wikipedia.org/wiki/Variance_inflation_factor
    """

    method_name = "VIF筛选"

    def __init__(
        self,
        threshold: float = 4.0,
        missing: float = -1.0,
        max_iter: int = 100,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        verbose: bool = False,
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
        self.missing = missing
        self.max_iter = max_iter
        self.verbose = verbose
        self.report_history = report_history
        self.max_report_events = max_report_events
        self.max_report_bytes = max_report_bytes

    def _included_features_participate_in_selection(self) -> bool:
        """强制保留字段参与共线性计算，但不会成为迭代剔除对象。"""
        return True

    def _compute_vif_all(self, X: pd.DataFrame) -> pd.Series:
        """计算所有特征的VIF值。

        :param X: 特征DataFrame
        :return: VIF值Series
        """
        x_filled = X.fillna(self.missing).values
        n_features = x_filled.shape[1]

        if n_features == 0:
            return pd.Series(dtype=float)
        # QR/OLS都使用浮点工作区，不能用int8/float32输入的nbytes低估副本。
        working_matrix_bytes = X.shape[0] * n_features * np.dtype(np.float64).itemsize

        bulk = self._parallel_execute(
            _compute_vif_bulk,
            [(x_filled, resolve_native_workers(self.n_jobs))],
            task_labels=["共享QR"],
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=1,
                rows=X.shape[0],
                columns=n_features,
                data_bytes=int(x_filled.nbytes),
                capability="vectorized",
                releases_gil=True,
                cost_per_item=max(10.0, float(n_features)),
                operation="VIF共享QR计算",
                working_bytes_per_task=int(5 * working_matrix_bytes + 4 * n_features * n_features * 8),
                result_bytes_per_task=n_features * 8,
                output_rows_per_task=n_features,
            ),
        )[0]
        if bulk is not None:
            self.vif_last_solver_ = "共享QR"
            eligible = np.array([name not in set(getattr(self, "include_", [])) for name in X.columns])
            if eligible.any():
                maximum = np.max(bulk[eligible])
                tied = np.flatnonzero(eligible & np.isclose(bulk, maximum, rtol=1e-12, atol=1e-12))
                if maximum > self.threshold and len(tied) > 1:
                    # VIF对两变量对称，但浮点残差会影响idxmax的旧剔除顺序。
                    # 只复核真正可能被剔除的平局候选，不重复全部p个OLS。
                    with threadpool_limits(limits=resolve_native_workers(self.n_jobs)):
                        for index in tied:
                            bulk[index] = _compute_vif_single(x_filled, int(index))
                    self.vif_last_solver_ = "共享QR（平局OLS复核）"
            return pd.Series(bulk, index=X.columns)
        self.vif_last_solver_ = "逐列OLS回退"

        results = self._parallel_execute(
            _compute_vif_feature,
            ((i, x_filled) for i in range(n_features)),
            task_labels=X.columns,
            default_backend="loky",
            workload=ParallelWorkload(
                task_count=n_features,
                rows=X.shape[0],
                columns=n_features,
                data_bytes=int(x_filled.nbytes),
                cost_per_item=max(10.0, float(n_features)),
                capability="process_safe",
                operation="VIF回归计算",
                working_bytes_per_task=int(4 * working_matrix_bytes + X.shape[0] * 24),
                result_bytes_per_task=16,
                output_rows_per_task=1,
            ),
        )
        vif_values = np.array([value for _, value in results])

        return pd.Series(vif_values, index=X.columns)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合VIF筛选器。

        采用迭代剔除策略，每次只剔除VIF最大的特征。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)
        initialize_history(self)
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter < 1
        ):
            raise ValueError("max_iter 必须为正整数")
        if (
            isinstance(self.threshold, (bool, np.bool_))
            or not isinstance(self.threshold, (int, float, np.number))
            or not np.isfinite(self.threshold)
            or self.threshold <= 0
        ):
            raise ValueError("VIF 阈值必须为有限正数")
        self.score_name_ = "VIF值"
        self.score_direction_ = "越小越好"
        self.effective_threshold_ = self.threshold
        self.selection_stopping_reason_ = "达到最大迭代次数"
        self.decision_vif_ = pd.Series(np.nan, index=X.columns, dtype=float)
        self.elimination_round_ = pd.Series(0, index=X.columns, dtype=int)

        # 保留的特征列表
        remaining_features = X.columns.tolist()
        forced_include = set(self.include_).intersection(remaining_features)
        # 被剔除的特征及原因
        dropped_features = []
        dropped_reasons = []
        dropped_values = []
        # 记录每次迭代的VIF值
        vif_history = []
        stored_cells = 0
        stored_bytes = 0
        iteration_count = 0
        last_vif = None

        # 迭代剔除
        for iteration in range(self.max_iter):
            if len(remaining_features) == 0:
                break

            # 计算当前所有特征的VIF
            X_current = X[remaining_features]
            vif_series = self._compute_vif_all(X_current)
            iteration_count += 1
            last_vif = vif_series
            series_bytes = int(vif_series.memory_usage(index=True, deep=True)) + 256
            if (
                stored_cells + len(vif_series) <= self.max_report_events
                and stored_bytes + series_bytes <= self.max_report_bytes
            ):
                vif_history.append(vif_series.copy())
                stored_cells += len(vif_series)
                stored_bytes += series_bytes
            for feature, value in vif_series.items():
                record_event(
                    self,
                    {
                        "轮次": iteration + 1,
                        "特征": feature,
                        "动作": "候选评估",
                        "指标名称": "VIF值",
                        "指标值": value,
                        "有效阈值": self.threshold,
                        "是否有效": not np.isnan(value),
                        "原因": "当前条件变量集合的多重共线性",
                    },
                    diagnostic=True,
                )

            # include 字段需要参与 VIF 回归，但不能被迭代剔除；只在其余
            # 候选字段中寻找本轮最大 VIF。
            removable_vif = vif_series.drop(
                labels=[feature for feature in forced_include if feature in vif_series.index]
            )
            if removable_vif.empty:
                self.selection_stopping_reason_ = "仅剩强制保留字段"
                break
            max_vif = removable_vif.max()
            max_feature = removable_vif.idxmax()

            if self.verbose:
                logger.info(f"迭代 {iteration + 1}: 最大VIF = {max_vif:.4f} (特征: {max_feature})")

            # 如果最大VIF <= threshold，停止
            if max_vif <= self.threshold:
                self.selection_stopping_reason_ = "所有可剔除字段满足VIF阈值"
                if self.verbose:
                    logger.info(f"所有特征VIF <= {self.threshold}，停止迭代")
                break

            # 剔除VIF最大的特征
            remaining_features.remove(max_feature)
            dropped_features.append(max_feature)
            dropped_reasons.append(f"VIF={max_vif:.4f} (第{iteration + 1}轮剔除)")
            dropped_values.append(float(max_vif))
            self.decision_vif_[max_feature] = max_vif
            self.elimination_round_[max_feature] = iteration + 1
            record_event(
                self,
                {
                    "轮次": iteration + 1,
                    "特征": max_feature,
                    "动作": "剔除",
                    "指标名称": "VIF值",
                    "指标值": float(max_vif),
                    "有效阈值": self.threshold,
                    "是否有效": not np.isnan(max_vif),
                    "原因": "本轮可剔除字段中VIF最大且超过阈值",
                },
            )

            if self.verbose:
                logger.info(f"  剔除特征: {max_feature}")

        # 保存结果
        self.selected_features_ = remaining_features
        self.removed_features_ = dropped_features

        # 保存最终的VIF值作为scores_
        if len(remaining_features) > 0:
            # 优先使用最后一次迭代计算的VIF值（避免重复计算）
            if last_vif is not None and list(last_vif.index) == remaining_features:
                self.scores_ = last_vif
            else:
                self.scores_ = self._compute_vif_all(X[remaining_features])
        else:
            self.scores_ = pd.Series(dtype=float)

        # 记录剔除历史
        self.vif_history_ = vif_history
        self.n_iterations_ = iteration_count
        self.vif_history_truncated_ = iteration_count - len(vif_history)
        self.vif_history_bytes_ = stored_bytes
        self.decision_vif_.loc[self.scores_.index] = self.scores_
        self.unresolved_features_ = self.scores_.index[self.scores_ > self.threshold].tolist()
        self.converged_ = not any(name not in forced_include for name in self.unresolved_features_)
        for feature, value in self.scores_.items():
            record_event(
                self,
                {
                    "轮次": iteration_count,
                    "特征": feature,
                    "动作": "保留",
                    "指标名称": "VIF值",
                    "指标值": value,
                    "有效阈值": self.threshold,
                    "是否有效": not np.isnan(value),
                    "原因": (
                        "强制保留"
                        if feature in forced_include
                        else ("满足VIF阈值" if value <= self.threshold else "迭代上限结束，仍超过阈值")
                    ),
                },
            )

        # 构建dropped_ DataFrame（提取VIF数值）
        if len(dropped_features) > 0:
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_features,
                    "剔除原因": dropped_reasons,
                    "VIF值": dropped_values,
                    "决策轮次": [int(self.elimination_round_[name]) for name in dropped_features],
                    "阈值": [self.threshold] * len(dropped_features),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "VIF值", "阈值"])

        self._drop_reason = f"VIF值 > {self.threshold}"

        if self.verbose:
            logger.info(f"\nVIF筛选完成:")
            logger.info(f"  迭代次数: {self.n_iterations_}")
            logger.info(f"  保留特征: {len(self.selected_features_)}")
            logger.info(f"  剔除特征: {len(self.removed_features_)}")
            if len(self.selected_features_) > 0:
                logger.info(f"  最终最大VIF: {self.scores_.max():.4f}")
