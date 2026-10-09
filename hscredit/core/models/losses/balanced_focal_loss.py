"""
平衡 Focal Loss

在标准 Focal Loss 上加入基于有效样本数的类别平衡方案与标签平滑，
提供比固定 alpha 更稳定的不平衡处理能力。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, focal_terms, nonnegative, positive, unit_interval


class BalancedFocalLoss(BaseLoss):
    """平衡 Focal Loss，在 FocalLoss 上加入更稳定的类别平衡方案。

    核心改进（相比标准 FocalLoss）:

    1. **有效样本数加权**: 基于 *Class-Balanced Loss* 论文，用
       ``E_n = (1 - β^n) / (1 - β)`` 自动计算正负样本权重，
       比手动设置 alpha 更鲁棒。
    2. **标签平滑**: 将硬标签 {0,1} 软化为 {ε/2, 1-ε/2}，
       减少过拟合并提升梯度稳定性。

    数学形式::

        y_smooth = y × (1-ε) + ε/2
        p_t      = y_smooth × p + (1-y_smooth) × (1-p)
        E_pos    = (1-β^n_pos) / (1-β)
        E_neg    = (1-β^n_neg) / (1-β)
        alpha    = E_neg / (E_pos+E_neg)
        w_t      = alpha if y=1 else 1-alpha
        Loss     = -w_t × (1-p_t)^γ × log(p_t)

    :param gamma: 聚焦参数，默认 2.0
        - gamma=0 且 label_smoothing=0 时退化为类别加权交叉熵
        - gamma 越大，易分类样本权重衰减越快
    :param beta: 有效样本数衰减因子，默认 0.999
        - 内部经验: 0.99~0.9999 之间；样本量越大建议 beta 越接近 1
    :param label_smoothing: 标签平滑系数，默认 0.0（不启用）
        - 内部经验: 0.01~0.1 之间可改善校准; 过大会降低区分度
    :param auto_alpha: 是否自动根据有效样本数计算 alpha，默认 True
        - 当 auto_alpha=False 时回退为固定 alpha
    :param alpha: 固定正样本权重，仅在 auto_alpha=False 时生效，默认 0.25
    :param name: 损失函数名称，默认 "balanced_focal_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import BalancedFocalLoss
    >>>
    >>> # 自动平衡（推荐）
    >>> loss = BalancedFocalLoss(gamma=2.0, beta=0.999, label_smoothing=0.05)
    >>>
    >>> y_true = np.array([0, 0, 0, 0, 1])  # 极不平衡
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.8])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 XGBoost 中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> bst = xgb.train({}, dtrain, obj=loss.to_xgboost(), num_boost_round=100)

    **引用**

    有效样本数加权出自 Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019).
    *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019.
    https://arxiv.org/abs/1901.05555 ；聚焦项来自 Focal Loss（Lin et al., 2017,
    https://arxiv.org/abs/1708.02002）。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.999,
        label_smoothing: float = 0.0,
        auto_alpha: bool = True,
        alpha: float = 0.25,
        name: str = "balanced_focal_loss",
    ):
        super().__init__(name)
        unit_interval(alpha=alpha, beta=beta, label_smoothing=label_smoothing)
        nonnegative(gamma=gamma)
        if beta >= 1:
            raise ValueError("beta 必须小于 1。")
        self.is_additive = not auto_alpha
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.auto_alpha = auto_alpha
        self.alpha = alpha

    def _compute_alpha(self, y_true: np.ndarray) -> float:
        """根据有效样本数自动计算 alpha。"""
        if not self.auto_alpha:
            return self.alpha

        n_pos = max(int(np.sum(y_true == 1)), 1)
        n_neg = max(int(np.sum(y_true == 0)), 1)

        # 有效样本数
        e_pos = (1 - self.beta**n_pos) / (1 - self.beta + 1e-12)
        e_neg = (1 - self.beta**n_neg) / (1 - self.beta + 1e-12)

        # alpha 为负样本有效数占比（给少数类更大权重）
        alpha = e_neg / (e_pos + e_neg + 1e-12)
        return float(alpha)

    def _smooth_labels(self, y_true: np.ndarray) -> np.ndarray:
        """标签平滑。"""
        if self.label_smoothing <= 0:
            return y_true
        return y_true * (1 - self.label_smoothing) + self.label_smoothing / 2

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import BalancedFocalLoss
        >>> loss = BalancedFocalLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import BalancedFocalLoss
        >>> loss = BalancedFocalLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import BalancedFocalLoss
        >>> loss = BalancedFocalLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        alpha = self._compute_alpha(y)
        smooth = self._smooth_labels(y)
        pt = smooth * p + (1 - smooth) * (1 - p)
        weight = np.where(y == 1, alpha, 1 - alpha)
        slope = 2 * smooth - 1
        value, grad, hess = focal_terms(pt, self.gamma)
        return weight * value, weight * grad * slope, weight * hess * slope**2

    def loss_values(self, y_true, y_pred):
        """逐样本贡献；auto_alpha=True 时权重依赖整个评估数据集。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import BalancedFocalLoss
        >>> loss = BalancedFocalLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]
