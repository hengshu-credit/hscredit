"""自定义评估指标.

提供金融风控场景常用的评估指标，如KS、Gini、PSI等。
"""

import numpy as np
from typing import Optional
from .base import BaseMetric


class KSMetric(BaseMetric):
    """KS (Kolmogorov-Smirnov) 指标.

    衡量模型区分好坏客户的能力，KS值越大表示模型区分能力越强。

    KS = max(|累积好客户比例 - 累积坏客户比例|)

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
    ...     feval=ks_metric.to_lightgbm(),
    ...     num_boost_round=100
    ... )
    """

    def __init__(self, name: str = "ks"):
        super().__init__(name, greater_is_better=True)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算KS值.

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: KS值，范围[0, 1]，越大越好
        """
        # 确保输入是一维数组
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        # 按预测分数排序
        sorted_indices = np.argsort(y_pred)
        sorted_y_true = y_true[sorted_indices]
        sorted_y_pred = y_pred[sorted_indices]

        # 计算累积分布
        n_samples = len(y_true)
        n_good = np.sum(y_true == 0)
        n_bad = np.sum(y_true == 1)

        if n_good == 0 or n_bad == 0:
            return 0.0

        # 累积好客户和坏客户数量
        cum_good = np.cumsum(sorted_y_true == 0)
        cum_bad = np.cumsum(sorted_y_true == 1)

        # 累积比例
        cum_good_rate = cum_good / n_good
        cum_bad_rate = cum_bad / n_bad

        # KS值
        ks = np.max(np.abs(cum_good_rate - cum_bad_rate))

        return float(ks)


class GiniMetric(BaseMetric):
    """Gini系数指标.

    Gini = 2 * AUC - 1

    衡量模型区分能力，Gini越大越好。

    :param name: 指标名称，默认为"gini"

    **参考样例**

    >>> from hscredit.core.models.losses import GiniMetric
    >>> gini_metric = GiniMetric()
    >>> gini_value = gini_metric(y_true, y_pred)
    """

    def __init__(self, name: str = "gini"):
        super().__init__(name, greater_is_better=True)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算Gini系数.

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: Gini系数，范围[-1, 1]，越大越好
        """
        # 确保输入是一维数组
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        # 计算AUC
        auc = self._compute_auc(y_true, y_pred)

        # Gini = 2*AUC - 1
        gini = 2 * auc - 1

        return float(gini)

    def _compute_auc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算AUC."""
        # 按预测分数排序（降序）
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_y_true = y_true[sorted_indices]

        # 计算TPR和FPR
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # 使用梯形法则计算AUC
        cum_pos = np.cumsum(sorted_y_true == 1)
        cum_neg = np.cumsum(sorted_y_true == 0)

        # TPR和FPR
        tpr = cum_pos / n_pos
        fpr = cum_neg / n_neg

        # 添加原点
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # 计算AUC（梯形法则）
        auc = np.trapz(tpr, fpr)

        return float(auc)


class PSIMetric(BaseMetric):
    """PSI (Population Stability Index) 指标.

    衡量样本分布的稳定性，常用于模型监控。

    PSI = sum((实际占比 - 期望占比) * ln(实际占比/期望占比))

    :param expected: 期望分布的预测分数（基准数据），默认为None。如果为None，需要在__call__时提供
    :param n_bins: 分箱数量，默认为10
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

    PSI解释:
    - PSI < 0.1: 分布稳定
    - 0.1 <= PSI < 0.25: 分布有轻微变化
    - PSI >= 0.25: 分布变化显著，需要关注
    """

    def __init__(
        self,
        expected: Optional[np.ndarray] = None,
        n_bins: int = 10,
        name: str = "psi"
    ):
        super().__init__(name, greater_is_better=False)  # PSI越小越好
        self.expected = expected
        self.n_bins = n_bins

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        expected: Optional[np.ndarray] = None
    ) -> float:
        """计算PSI.

        :param y_true: 真实标签（此指标中未使用，保持接口一致性）
        :param y_pred: 实际分布的预测分数
        :param expected: 期望分布的预测分数
        :return: PSI值，越小越好
        """
        # 使用传入的expected或初始化时的expected
        expected = expected if expected is not None else self.expected

        if expected is None:
            raise ValueError("需要提供期望分布(expected)")

        # 确保一维
        expected = np.ravel(expected)
        actual = np.ravel(y_pred)

        # 计算分位数作为分箱边界
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(expected, quantiles)

        # 确保边界唯一
        bin_edges = np.unique(bin_edges)

        # 计算期望分布和实际分布的频率
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # 转换为比例
        expected_rates = expected_counts / len(expected)
        actual_rates = actual_counts / len(actual)

        # 避免除零和log(0)
        expected_rates = np.clip(expected_rates, 1e-10, 1)
        actual_rates = np.clip(actual_rates, 1e-10, 1)

        # 计算PSI
        psi = np.sum(
            (actual_rates - expected_rates) *
            np.log(actual_rates / expected_rates)
        )

        return float(psi)
