"""自定义评估指标.

提供金融风控场景常用的评估指标，如KS、Gini、PSI等。
"""

import numpy as np
from typing import Optional
from .base import BaseMetric
from ...metrics import auc as auc_metric, ks as ks_metric


class KSMetric(BaseMetric):
    """KS (Kolmogorov-Smirnov) 指标.

    衡量模型区分好坏客户的能力，KS值越大表示模型区分能力越强。

    ``KS = max(|累积好客户比例 - 累积坏客户比例|)``

    :param name: 指标名称，默认为"ks"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import KSMetric
    >>>
    >>> ks_metric = KSMetric()
    >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
    >>> ks_value = ks_metric(y_true, y_pred)
    >>> print(f"KS: {ks_value:.4f}")

    >>> # 在LightGBM中使用
    >>> import lightgbm as lgb
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> bst = lgb.train(
    ...     params={'objective': 'binary'},
    ...     train_set=train_data,
    ...     feval=ks_metric.to_lightgbm(api="native"),
    ...     num_boost_round=100
    ... )

    **引用**

    KS（Kolmogorov–Smirnov）统计量：https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test ；
    其在信用评分中的应用见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley。
    """

    def __init__(self, name: str = "ks"):
        super().__init__(name, greater_is_better=True)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight=None,
    ) -> float:
        """计算KS值.

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率；统一输入校验请使用 evaluate。
        :param sample_weight: 可选非负有限权重，须与标签等长且两类都有正权重。
        :return: KS值，范围[0, 1]，越大越好

        >>> value = KSMetric().evaluate([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        # 确保输入是一维数组
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        if sample_weight is not None:
            from sklearn.metrics import roc_curve
            from .base import _validate_binary_inputs

            y_true, y_pred, sample_weight = _validate_binary_inputs(y_true, y_pred, sample_weight)
            if not all(np.sum(sample_weight[y_true == label]) > 0 for label in (0, 1)):
                raise ValueError("计算加权KS需要有效权重下同时存在好样本和坏样本")
            false_positive, true_positive, _ = roc_curve(y_true, y_pred, sample_weight=sample_weight)
            return float(np.max(np.abs(true_positive - false_positive)))
        return ks_metric(y_true, y_pred)


class GiniMetric(BaseMetric):
    """Gini系数指标.

    Gini = 2 * AUC - 1

    衡量模型区分能力，Gini越大越好。

    :param name: 指标名称，默认为"gini"
    :param score_direction: 同 ``metrics.auc``，默认auto；higher_risk保留原始方向

    **参考样例**

    >>> from hscredit.core.models.losses import GiniMetric
    >>> gini_metric = GiniMetric()
    >>> gini_value = gini_metric(y_true, y_pred)
    """

    def __init__(self, name: str = "gini", score_direction: str = "auto"):
        super().__init__(name, greater_is_better=True)
        self.score_direction = score_direction

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight=None,
    ) -> float:
        """计算Gini系数.

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率；统一输入校验请使用 evaluate。
        :param sample_weight: 可选非负有限权重，须与标签等长且总和大于 0。
        :return: Gini系数，默认范围[0, 1]；显式方向时范围[-1, 1]，越大越好

        >>> value = GiniMetric(score_direction='higher_risk').evaluate(
        ...     [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        # 确保输入是一维数组
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        # 计算AUC
        auc = self._compute_auc(y_true, y_pred, sample_weight=sample_weight)

        # Gini = 2*AUC - 1
        gini = 2 * auc - 1

        return float(gini)

    def _compute_auc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight=None,
    ) -> float:
        """调用公共AUC；保留单类别回调返回中性值的既有约定。"""
        if np.unique(y_true).size < 2:
            return 0.5
        return auc_metric(y_true, y_pred, sample_weight=sample_weight, score_direction=self.score_direction)


class PSIMetric(BaseMetric):
    """PSI (Population Stability Index) 指标.

    衡量样本分布的稳定性，常用于模型监控。

    PSI = sum((实际占比 - 期望占比) * ln(实际占比/期望占比))

    :param expected: 基准预测分数的一维非空有限数组，默认 None，
        为 None 时需要在 __call__ 中提供；不要求与当前样本等长。
    :param n_bins: 基准分位分箱的最大数量，整数且至少为 2，默认 10。
    :param name: 指标名称，默认为"psi"

    **参考样例**

    >>> from hscredit.core.models.losses import PSIMetric
    >>>
    >>> # 使用训练集作为基准
    >>> psi_metric = PSIMetric(expected=y_train_pred, n_bins=10)
    >>>
    >>> # 计算测试集的PSI
    >>> psi_value = psi_metric(y_test, y_test_pred)
    >>> print(f"PSI: {psi_value:.4f}")

    **注意**

    复用 ``hscredit.core.metrics.psi`` 的冻结基准分箱，不丢弃落在基准
    最小值和最大值之外的样本。低基数或常量基准遵循公共 PSI 的类别口径。
    本指标不支持样本权重；它衡量分布漂移，不能单独代替 AUC 等区分度指标。

    PSI解释:
    - PSI < 0.1: 分布稳定
    - 0.1 <= PSI < 0.25: 分布有轻微变化
    - PSI >= 0.25: 分布变化显著，需要关注
    """

    def __init__(self, expected: Optional[np.ndarray] = None, n_bins: int = 10, name: str = "psi"):
        if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
            raise ValueError("PSI 的 n_bins 必须是至少为 2 的整数")
        super().__init__(name, greater_is_better=False)  # PSI越小越好
        self.expected = self._validate_distribution(expected, "基准分布") if expected is not None else None
        self.n_bins = int(n_bins)

    @staticmethod
    def _validate_distribution(values, name):
        """校验一维有限分布，返回独立副本以固定基准。"""
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PSI 的{name}必须是一维非空有限数值数组") from exc
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"PSI 的{name}必须是一维非空有限数值数组")
        return values.copy()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, expected: Optional[np.ndarray] = None) -> float:
        """计算PSI.

        :param y_true: 真实标签，此方法中未使用；调用 evaluate 时仍须提供 0/1 标签。
        :param y_pred: 当前分布的一维非空有限预测分数。
        :param expected: 可选基准分布，覆盖构造时的基准，仅对本次计算生效。
        :return: float，PSI 越小表示分布越接近；计算口径同公共 metrics.psi。
        :raises ValueError: 未设置基准，或输入为空、多维、非有限数值。

        >>> metric = PSIMetric(expected=[0.1, 0.2, 0.3, 0.4], n_bins=2)
        >>> value = metric([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        """
        expected = expected if expected is not None else self.expected
        if expected is None:
            raise ValueError("需要提供期望分布(expected)")
        expected = self._validate_distribution(expected, "基准分布")
        actual = self._validate_distribution(y_pred, "当前分布")
        from ...metrics import psi

        return float(psi(expected, actual, method="quantile", max_n_bins=self.n_bins))
