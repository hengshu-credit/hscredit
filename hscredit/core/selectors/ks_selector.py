"""基于原始字段 KS 值的特征筛选器。"""

from numbers import Real
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .base import BaseFeatureSelector
from ..metrics import compute_bin_stats, ks_2samps
from ...exceptions import ValidationError
from ...utils.parallel import ParallelWorkload


def _compute_ks_feature(task):
    """按有效样本计算单字段 KS，相同取值作为整体参与累积。"""
    feature, series, y, binned = task
    valid = series.notna().to_numpy()
    series = series.iloc[np.flatnonzero(valid)]
    target = y[valid]
    if series.nunique() <= 1 or np.unique(target).size < 2:
        return feature, 0.0

    if binned:
        # 统一计算入口同时计算 WOE，需保留正平滑值以避免纯好/坏箱除零。
        # 半个机器精度不会改变至少为 1 的好坏样本总数，KS=1 的阈值边界仍准确。
        stats = compute_bin_stats(series.to_numpy(), target, epsilon=np.finfo(float).eps / 2, round_digits=False)
        return feature, float(stats["分档KS值"].max())

    if is_numeric_dtype(series.dtype):
        values = series.to_numpy()
    else:
        # 无序类别按训练样本坏样本率排序，避免类别名称和首次出现顺序影响结果。
        codes, _ = pd.factorize(series, sort=False)
        counts = np.bincount(codes)
        bad_rates = np.bincount(codes, weights=target) / counts
        values = bad_rates[codes]

    return feature, float(ks_2samps(values[target == 0], values[target == 1]))


class KSSelector(BaseFeatureSelector):
    """KS 值筛选器。

    对每个字段计算好坏样本经验分布的最大绝对差，保留 ``KS >= threshold``
    的特征。数值字段直接使用原始取值，无需预先分箱，正向与反向特征采用相同口径。
    相同取值整体累计，不受样本排列顺序影响。

    非数值字段（包括 object、category、string）按训练样本的类别坏样本率排序后
    计算 KS；这是样本内区分度，高基数类别建议先合并或通过分箱器处理。
    每个字段独立排除缺失值；全缺失、常量或排除缺失后仅剩一种标签时 KS 为 0。
    配置分箱器时，使用 ``compute_bin_stats`` 的最大分档 KS，缺失箱与特殊值箱
    遵循该函数的排序约定。

    **参数**

    :param threshold: KS 保留阈值，范围为 [0, 1]，默认为 0.1；等于阈值时保留
    :param target: 目标变量列名，默认为 'target'；目标必须同时包含 0 和 1，且无缺失
    :param include: 强制保留的字段，遵循基类约定跳过指标计算
    :param exclude: 强制剔除的字段，优先于 include，跳过指标计算
    :param force_drop: 额外强制剔除字段，与 exclude 合并
    :param n_jobs: 并行工作数，默认为 -1；1 或 None 表示串行
    :param binner: 可选分箱器实例，未拟合时自动训练，已拟合时复用规则
    :param binning_params: 可选 OptimalBinning 构造参数，binner 优先
    :param parallel_backend: 可选 joblib 并行后端
    :param parallel_config: 可选 joblib 并行配置

    **属性**

    - scores_: 参与计算的各字段 KS 值，pd.Series，名称为“KS值”
    - selected_features_: 按原始列顺序保留的字段列表，不含目标列
    - removed_features_: 剔除字段列表
    - dropped_: 中文剔除详情，包含“特征”“剔除原因”“KS值”“阈值”
    - n_features_in_: 输入特征数量，不含目标列

    **参考样例**

    >>> from hscredit import KSSelector
    >>> import pandas as pd
    >>> df = pd.DataFrame({'评分': [10, 20, 80, 90], '常量': [1, 1, 1, 1], 'FPD': [0, 0, 1, 1]})
    >>> selector = KSSelector(target='FPD', threshold=0.2)
    >>> selected_df = selector.fit_transform(df)  # 输出保留原始字段值，并透传 FPD
    >>> selector.selected_features_
    ['评分']
    >>> selector.fit(df.drop(columns='FPD'), df['FPD'])  # sklearn 风格
    KSSelector(...)
    >>> report = selector.get_selection_report()

    同时传入含目标列的数据框与外部 y 时，使用外部 y，目标列始终不参与筛选。
    ``transform`` 仅选择原始列，不输出类别坏样本率或分箱索引。
    """

    method_name = "KS值筛选"

    def __init__(
        self,
        threshold: float = 0.1,
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

    def _check_input(self, X, y=None):
        """在分箱或跳过强制字段前校验参数，始终将目标列排除在特征之外。"""
        if (
            isinstance(self.threshold, (bool, np.bool_))
            or not isinstance(self.threshold, Real)
            or not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValidationError("KS 阈值必须是 [0, 1] 范围内的有限数值")
        if y is not None and np.asarray(y).ndim != 1:
            raise ValidationError("KSSelector 要求目标变量为一维数组")
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValidationError("输入数据字段名不能重复")
            if y is not None and self.target in X.columns:
                X = X.drop(columns=[self.target])
        X, y = super()._check_input(X, y)
        if y is None:
            raise ValidationError("KSSelector 需要目标变量 y，或通过 target 指定数据中的目标列")
        y = np.asarray(y)
        if y.ndim != 1 or pd.isna(y).any() or not pd.Series(y).isin([0, 1]).all():
            raise ValidationError("KSSelector 要求目标变量为无缺失的一维 0/1 二分类标签")
        if np.unique(y).size != 2:
            raise ValidationError("KSSelector 要求目标变量同时包含 0 和 1")
        return X, y.astype(int)

    def _fit_impl(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """并行计算字段 KS，并记录满足阈值的特征及中文剔除详情。"""
        binned = self._binner_instance is not None
        results = self._parallel_execute(
            _compute_ks_feature,
            ((column, X[column], y, binned) for column in X.columns),
            task_labels=X.columns,
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=X.shape[1],
                rows=X.shape[0],
                columns=X.shape[1],
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=5.0,
                capability="thread_safe",
                releases_gil=True,
                operation="KS字段计算",
            ),
        )
        self.scores_ = pd.Series([score for _, score in results], index=X.columns, dtype=float, name="KS值")
        selected_mask = self.scores_ >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        dropped_columns = X.columns[~selected_mask].tolist()
        self.dropped_ = pd.DataFrame(
            {
                "特征": dropped_columns,
                "剔除原因": [
                    f"KS值({self.scores_[column]:.4f}) < 阈值({self.threshold})" for column in dropped_columns
                ],
                "KS值": [self.scores_[column] for column in dropped_columns],
                "阈值": [self.threshold] * len(dropped_columns),
            }
        )
