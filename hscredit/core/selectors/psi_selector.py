"""PSI稳定性筛选器.

使用群体稳定性指标（PSI）筛选特征。

**参考样例**

>>> from hscredit.core.selectors import PSISelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = PSISelector(threshold=0.25, n_splits=5)  # 筛选PSI<0.25的稳定特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions, record_counts, validate_real
from ...exceptions import ValidationError
from ...utils.parallel import ParallelWorkload


def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    """用轻量平滑从同一组桶计数计算 PSI。"""
    epsilon = 1e-6
    expected_counts = np.asarray(expected_counts, dtype=float)
    actual_counts = np.asarray(actual_counts, dtype=float)
    expected_rates = (expected_counts + epsilon) / (expected_counts.sum() + epsilon * len(expected_counts))
    actual_rates = (actual_counts + epsilon) / (actual_counts.sum() + epsilon * len(actual_counts))
    return float(np.sum((actual_rates - expected_rates) * np.log(actual_rates / expected_rates)))


def _compute_psi_single(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """计算单个特征的PSI值。

    :param expected: 期望值数组（训练集）
    :param actual: 实际值数组（测试集）
    :return: PSI值
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 2:
        raise ValueError("n_bins 必须是大于等于 2 的整数")

    from ..metrics.monitoring import MonitoringBaseline

    return MonitoringBaseline(max_n_bins=int(n_bins)).fit(expected).evaluate(actual)["PSI"]


def _compute_psi_feature(task):
    """按既定折序计算单个特征的平均 PSI。"""
    feature, values, fold_ids, n_bins = task
    scores = []
    for fold in range(int(fold_ids.max()) + 1):
        actual = fold_ids == fold
        scores.append(_compute_psi_single(values[~actual], values[actual], n_bins=n_bins))
    return feature, float(np.mean(scores)), scores


def _compute_psi_pair_feature(task):
    """计算一个训练/OOT字段对的 PSI。"""
    feature, expected, actual, n_bins = task
    value = _compute_psi_single(expected, actual, n_bins=n_bins)
    return feature, value, [value]


class PSISelector(BaseFeatureSelector):
    """PSI筛选器.

    使用群体稳定性指标（Population Stability Index）筛选特征。
    PSI衡量特征在不同样本间的分布稳定性。
    常用于跨时间验证和oot验证。

    PSI值解释:
    - < 0.1: 特征稳定性好
    - 0.1 - 0.25: 特征有轻微变化，需要关注
    - > 0.25: 特征分布变化显著，需要处理

    **参数**

    :param threshold: PSI阈值，默认为0.25
        - 0.25: 移除PSI值达到或超过0.25的特征
    :param n_splits: 交叉验证折数，用于计算PSI
    :param oot_df: 可选的真实 OOT 对照集；传入时直接计算训练集与 OOT 的 PSI
    :param psi_bins: 连续变量分箱数，默认为10；低基数/类别变量按类别对齐
    :param random_state: 未传 OOT 时交叉分折的随机种子
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **参考样例**

    ::

        >>> from hscredit.core.selectors import PSISelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 1000))
        >>> selector = PSISelector(threshold=0.25, n_splits=5)
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **引用**

    PSI（群体稳定性指标）公式 ``PSI = Σ (实际占比 - 预期占比) × ln(实际占比 / 预期占比)``，
    经验阈值 <0.1 稳定、0.1~0.25 轻微漂移、>0.25 显著漂移，广泛用于信用评分的稳定性
    监控，参见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley。
    """

    method_name = "PSI筛选"

    def __init__(
        self,
        threshold: float = 0.25,
        n_splits: int = 5,
        oot_df: Optional[pd.DataFrame] = None,
        psi_bins: int = 10,
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
    ):
        """初始化筛选器；默认透传已有目标列，仅target_rm=True移除。"""
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
        self.n_splits = n_splits
        self.oot_df = oot_df
        self.psi_bins = psi_bins
        self.random_state = random_state

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "PSI阈值", minimum=0)
        if (
            isinstance(self.psi_bins, (bool, np.bool_))
            or not isinstance(self.psi_bins, (int, np.integer))
            or self.psi_bins < 2
        ):
            raise ValueError("psi_bins 必须是大于等于 2 的整数")
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合PSI筛选器。

        使用交叉验证，将数据分为训练集和验证集，计算PSI。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "PSI值", "越小越好"

        if isinstance(self.psi_bins, (bool, np.bool_)) or not isinstance(self.psi_bins, (int, np.integer)):
            raise ValueError("psi_bins 必须是整数")
        if int(self.psi_bins) < 2:
            raise ValueError("psi_bins 必须大于等于 2")
        if isinstance(self.threshold, (bool, np.bool_)) or not isinstance(
            self.threshold, (int, float, np.integer, np.floating)
        ):
            raise ValueError("threshold 必须是非负数")
        if float(self.threshold) < 0:
            raise ValueError("threshold 必须是非负数")

        if self.oot_df is not None:
            if not isinstance(self.oot_df, pd.DataFrame):
                raise ValidationError("oot_df 必须是 pandas DataFrame")
            oot = self.oot_df.drop(columns=[self.target], errors="ignore")
            missing = [column for column in X.columns if column not in oot.columns]
            if missing:
                raise ValidationError(f"OOT 数据缺少拟合字段: {missing}")
            oot = oot.loc[:, X.columns]
            if len(oot) == 0:
                raise ValidationError("OOT 数据不能为空")
            if not oot.columns.is_unique:
                raise ValidationError("OOT 数据字段名不能重复")
            if self._binner_instance is not None:
                oot = self._transform_with_fitted_binner(oot)
            worker = _compute_psi_pair_feature
            tasks = ((col, X[col].values, oot[col].values, self.psi_bins) for col in X.columns)
            operation = "训练集与OOT的PSI计算"
            cost = 6.0
            self.comparison_mode_ = "训练集与OOT"
            reference_counts = [len(X)]
            comparison_counts = [len(oot)]
        else:
            if isinstance(self.n_splits, (bool, np.bool_)) or not isinstance(self.n_splits, (int, np.integer)):
                raise ValueError("n_splits 必须是整数")
            if not 2 <= int(self.n_splits) <= len(X):
                raise ValueError("n_splits 必须在 [2, 样本数] 范围内")
            # 等价于 KFold(shuffle=True) 的折分配，仅保存 O(n) 整数标记；
            # 不在宽表并行任务间复制 O(n × 折数) 的训练/验证索引列表。
            from sklearn.utils import check_random_state

            order = np.arange(len(X))
            check_random_state(self.random_state).shuffle(order)
            fold_ids = np.empty(len(X), dtype=np.int32)
            sizes = np.full(int(self.n_splits), len(X) // int(self.n_splits), dtype=int)
            sizes[: len(X) % int(self.n_splits)] += 1
            start = 0
            for ordinal, size in enumerate(sizes):
                fold_ids[order[start : start + size]] = ordinal
                start += size
            worker = _compute_psi_feature
            tasks = ((col, X[col].values, fold_ids, self.psi_bins) for col in X.columns)
            operation = "PSI交叉验证"
            cost = max(4.0, float(self.n_splits) * 2.0)
            self.comparison_mode_ = "随机交叉折"
            reference_counts = (len(X) - sizes).tolist()
            comparison_counts = sizes.tolist()

        results = self._parallel_execute(
            worker,
            tasks,
            task_labels=X.columns,
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=X.shape[1],
                rows=X.shape[0],
                columns=X.shape[1],
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=cost,
                capability="thread_safe",
                releases_gil=True,
                operation=operation,
            ),
        )
        psi_values = np.array([result[1] for result in results])
        self.scores_ = pd.Series(psi_values, index=X.columns)
        fold_names = [f"第{ordinal + 1}折PSI" for ordinal in range(len(reference_counts))]
        self.fold_scores_ = pd.DataFrame([result[2] for result in results], index=X.columns, columns=fold_names)
        self.psi_std_ = self.fold_scores_.std(axis=1, ddof=0)
        self.psi_max_ = self.fold_scores_.max(axis=1)
        self.reference_counts_ = pd.Series(reference_counts, index=fold_names, dtype=np.int64)
        self.comparison_counts_ = pd.Series(comparison_counts, index=fold_names, dtype=np.int64)

        # 选择PSI值小于阈值的特征（PSI越小越稳定）
        selected_mask = psi_values < self.threshold
        record_conditions(self, X.columns, PSI达标=selected_mask)
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含PSI值
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [f"PSI值({self.scores_[col]:.4f}) >= 阈值({self.threshold})" for col in dropped_cols],
                    "PSI值": [self.scores_[col] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "PSI值", "阈值"])
