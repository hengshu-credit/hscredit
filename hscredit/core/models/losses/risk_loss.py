"""
风控业务损失函数

针对金融风控场景设计的专用损失函数，考虑坏账率、通过率、利润最大化等业务指标。
"""

import numpy as np
from typing import Optional, Dict
from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, nonnegative, positive, unit_interval


class BadDebtLoss(BaseLoss):
    """坏账率优化损失函数，最小化坏账率同时保持通过率在合理水平。

    适用于信贷审批场景，希望降低通过客户的坏账比例。

    业务惩罚依赖硬阈值，训练仅使用 BCE 代理导数；本类不直接对坏账率求导。

    :param target_approval_rate: 目标通过率，范围 (0, 1]，默认为0.3
    :param bad_debt_weight: 坏账率权重，默认为1.0
    :param approval_weight: 通过率权重，默认为0.5
    :param name: 损失函数名称，默认为"bad_debt_loss"

    **参考样例**

    >>> from hscredit.core.models.losses import BadDebtLoss
    >>>
    >>> # 目标通过率30%，重点优化坏账率
    >>> loss = BadDebtLoss(
    ...     target_approval_rate=0.3,
    ...     bad_debt_weight=1.0,
    ...     approval_weight=0.3
    ... )
    >>>
    >>> # 该目标依赖全量排序，使用 LightGBM 的全量目标回调
    >>> from lightgbm import LGBMClassifier
    >>> model = LGBMClassifier(
    ...     n_estimators=1000,
    ...     objective=loss.to_lightgbm(),
    ...     metric='None',
    ... )
    >>> model.fit(
    ...     X_train, y_train, eval_set=[(X_valid, y_valid)],
    ...     eval_metric=loss.metric().to_lightgbm(raw_score=True),
    ... )

    **引用**

    在目标通过率约束下最小化坏账率，属信贷审批的业务驱动目标；成本敏感学习背景见
    Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI 2001。
    """

    def __init__(
        self,
        target_approval_rate: float = 0.3,
        bad_debt_weight: float = 1.0,
        approval_weight: float = 0.5,
        name: str = "bad_debt_loss",
    ):
        super().__init__(name)
        unit_interval(target_approval_rate=target_approval_rate)
        positive(target_approval_rate=target_approval_rate)
        nonnegative(bad_debt_weight=bad_debt_weight, approval_weight=approval_weight)
        self.target_approval_rate = target_approval_rate
        self.bad_debt_weight = bad_debt_weight
        self.approval_weight = approval_weight

    def _compute_metrics(self, y_true, y_pred, threshold):
        """风险概率低于或等于阈值才通过；同分客户统一决策。"""
        y, p = binary_inputs(y_true, y_pred)
        approved = p <= threshold
        return {
            "approval_rate": float(np.mean(approved)),
            "bad_debt_rate": float(np.mean(y[approved])) if np.any(approved) else 0.0,
        }

    def __call__(self, y_true, y_pred) -> float:
        """BCE + 目标通过率处的硬坏账惩罚；业务项为分段常数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import BadDebtLoss
        >>> loss = BadDebtLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        count = int(np.ceil(len(p) * self.target_approval_rate))
        threshold = np.sort(p)[count - 1] if count else -1.0
        metrics = self._compute_metrics(y, p, threshold)
        return float(
            np.mean(bce_terms(y, p)[0])
            + self.bad_debt_weight * metrics["bad_debt_rate"]
            + self.approval_weight * abs(metrics["approval_rate"] - self.target_approval_rate)
        )

    def gradient(self, y_true, y_pred):
        """硬审批惩罚无连续导数，训练仅使用 BCE 代理梯度。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import BadDebtLoss
        >>> loss = BadDebtLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return bce_terms(y, p)[1]

    def hessian(self, y_true, y_pred):
        """返回 BCE 代理目标的概率二阶导。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import BadDebtLoss
        >>> loss = BadDebtLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return bce_terms(y, p)[2]


class ApprovalRateLoss(BaseLoss):
    """通过率优化损失函数，在保证坏账率不超过目标的前提下最大化通过率。

    硬阈值通过率用于评估与模型选择；训练仅使用 BCE 代理导数。

    :param target_bad_debt_rate: 目标坏账率，默认为0.05
    :param name: 损失函数名称，默认为"approval_rate_loss"

    **参考样例**

    >>> from hscredit.core.models.losses import ApprovalRateLoss
    >>>
    >>> # 目标坏账率不超过5%
    >>> loss = ApprovalRateLoss(target_bad_debt_rate=0.05)
    """

    def __init__(self, target_bad_debt_rate: float = 0.05, name: str = "approval_rate_loss"):
        super().__init__(name)
        unit_interval(target_bad_debt_rate=target_bad_debt_rate)
        self.target_bad_debt_rate = target_bad_debt_rate

    def __call__(self, y_true, y_pred) -> float:
        """BCE + (1 - 可行通过率)，其中硬阈值通过率为分段常数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import ApprovalRateLoss
        >>> loss = ApprovalRateLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return float(np.mean(bce_terms(y, p)[0]) + 1 - self.approval_rate(y, p))

    def gradient(self, y_true, y_pred):
        """硬阈值业务项用于评估，训练使用 BCE 代理梯度。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import ApprovalRateLoss
        >>> loss = ApprovalRateLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return bce_terms(y, p)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import ApprovalRateLoss
        >>> loss = ApprovalRateLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return bce_terms(y, p)[2]

    def approval_rate(self, y_true, y_pred) -> float:
        """在当前数据标签上回看、满足目标坏账率的最大阈值通过率。

        用于离线评价，不应据此在测试集选择上线阈值；同分客户统一处理。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状的坏样本概率，低概率优先通过。
        :return: float，范围 [0, 1]；没有满足约束的客户组时为 0。

        >>> value = ApprovalRateLoss(target_bad_debt_rate=0.05).approval_rate(
        ...     [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        order = np.argsort(p, kind="stable")
        counts = np.arange(1, len(y) + 1)
        debt_rates = np.cumsum(y[order]) / counts
        group_end = np.r_[p[order][1:] != p[order][:-1], True]
        feasible = (debt_rates <= self.target_bad_debt_rate) & group_end
        return float(np.max(counts[feasible]) / len(y)) if np.any(feasible) else 0.0


class ProfitMaxLoss(BaseLoss):
    """利润最大化损失函数，综合考虑坏账损失和利息收益最大化总利润。

    训练目标为成本加权 BCE 代理；``profit()`` 单独计算实际阈值利润。
    需要连续软利润目标时使用 ``ExpectedProfitLoss``。

    :param interest_income: 单位利息收益，默认为1.0
    :param bad_debt_loss: 单位坏账损失，默认为10.0
    :param name: 损失函数名称，默认为"profit_max_loss"

    **参考样例**

    >>> from hscredit.core.models.losses import ProfitMaxLoss
    >>>
    >>> # 假设每笔贷款利息收益100元，坏账损失1000元
    >>> loss = ProfitMaxLoss(interest_income=100, bad_debt_loss=1000)
    """

    def __init__(self, interest_income: float = 1.0, bad_debt_loss: float = 10.0, name: str = "profit_max_loss"):
        super().__init__(name)
        nonnegative(interest_income=interest_income, bad_debt_loss=bad_debt_loss)
        positive(收益损失总和=interest_income + bad_debt_loss)
        self.is_additive = True
        self.interest_income = interest_income
        self.bad_debt_loss = bad_debt_loss

    def _compute_profit(self, y_true, y_pred, threshold) -> float:
        """兼容历史方法：返回低风险客户通过后的人均真实利润。"""
        return self.profit(y_true, y_pred, threshold)

    def __call__(self, y_true, y_pred) -> float:
        """利润成本加权 BCE 代理；实际硬决策利润通过 profit() 计算。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import ProfitMaxLoss
        >>> loss = ProfitMaxLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import ProfitMaxLoss
        >>> loss = ProfitMaxLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import ProfitMaxLoss
        >>> loss = ProfitMaxLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        weights = np.where(y == 1, self.bad_debt_loss, self.interest_income)
        return tuple(weights * term for term in bce_terms(y, p))

    def loss_values(self, y_true, y_pred):
        """逐样本利润成本加权 BCE，不等于实际货币利润。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import ProfitMaxLoss
        >>> loss = ProfitMaxLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]

    def profit(self, y_true, y_pred, threshold=0.5) -> float:
        """计算审批决策后的全体申请客户人均实际利润，越大越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状的坏样本概率。
        :param threshold: 通过阈值，范围 [0, 1]，默认 0.5；p < threshold 才通过。
        :return: float，通过好客户计收益，通过坏客户计损失，拒绝计零；
            分母是全量人数，不是通过人数。

        >>> value = ProfitMaxLoss(interest_income=100, bad_debt_loss=1000).profit(
        ...     [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        p = np.asarray(y_pred, dtype=float)
        unit_interval(threshold=threshold)
        profit = (1 - y) * self.interest_income - y * self.bad_debt_loss
        return float(np.mean((p < threshold) * profit))
