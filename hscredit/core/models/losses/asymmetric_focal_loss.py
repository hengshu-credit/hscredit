"""
不对称Focal Loss

针对风控极度不平衡数据，分别控制正负样本的聚焦强度。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, focal_terms, nonnegative, positive, unit_interval


class AsymmetricFocalLoss(BaseLoss):
    """不对称 Focal Loss。

    与标准 Focal Loss 不同，该损失允许对正负样本使用不同的聚焦参数，
    从而更灵活地强调坏样本识别或抑制易分类好样本的影响。

    数学形式:
        - 正样本: -alpha * (1 - p)^gamma_pos * log(p)
        - 负样本: -(1 - alpha) * p^gamma_neg * log(1 - p)

    :param alpha: 正样本权重，默认 0.25
    :param gamma_pos: 正样本聚焦参数，默认 2.0
    :param gamma_neg: 负样本聚焦参数，默认 1.0
    :param clip_value: 对负样本概率进行裁剪，抑制极端易分类负样本影响，默认 0.0
    :param name: 损失函数名称，默认 "asymmetric_focal_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import AsymmetricFocalLoss
    >>> loss = AsymmetricFocalLoss(alpha=0.7, gamma_pos=2.5, gamma_neg=1.0)
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    >>> round(loss(y_true, y_pred), 6) >= 0
    True

    **引用**

    在 Focal Loss（Lin et al., 2017, https://arxiv.org/abs/1708.02002）基础上对正负样本
    采用不同聚焦参数，思想与非对称损失 Ben-Baruch, E. et al. (2021). *Asymmetric Loss for
    Multi-Label Classification* 相通（https://arxiv.org/abs/2009.14119）。
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma_pos: float = 2.0,
        gamma_neg: float = 1.0,
        clip_value: float = 0.0,
        name: str = "asymmetric_focal_loss",
    ):
        super().__init__(name)
        unit_interval(alpha=alpha, clip_value=clip_value)
        nonnegative(gamma_pos=gamma_pos, gamma_neg=gamma_neg)
        if clip_value >= 1:
            raise ValueError("clip_value 必须小于 1。")
        self.is_additive = True
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_value = clip_value

    def _clip_probabilities(self, y_pred):
        """仅负类使用 p_minus=max(p-clip_value, 0)。"""
        return np.clip(np.asarray(y_pred, dtype=float) - self.clip_value, 1e-7, 1 - 1e-7)

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import AsymmetricFocalLoss
        >>> loss = AsymmetricFocalLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import AsymmetricFocalLoss
        >>> loss = AsymmetricFocalLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import AsymmetricFocalLoss
        >>> loss = AsymmetricFocalLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        negative_p = self._clip_probabilities(p)
        pt = np.where(y == 1, p, 1 - negative_p)
        gamma = np.where(y == 1, self.gamma_pos, self.gamma_neg)
        alpha = np.where(y == 1, self.alpha, 1 - self.alpha)
        value, grad, hess = focal_terms(pt, gamma)
        active = (y == 1) | (p > self.clip_value + 1e-7)
        return alpha * value, alpha * grad * (2 * y - 1) * active, alpha * hess * active

    def loss_values(self, y_true, y_pred):
        """逐样本不对称 Focal 损失；负类截断区内导数为零。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import AsymmetricFocalLoss
        >>> loss = AsymmetricFocalLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]
