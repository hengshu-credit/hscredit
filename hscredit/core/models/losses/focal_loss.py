"""
Focal Loss - 处理类别不平衡的损失函数

Focal Loss通过降低易分类样本的权重，专注于难分类样本，特别适合金融风控场景中的
不平衡数据问题（如坏账率通常很低）。
"""

import numpy as np
from typing import Optional
from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, focal_terms, nonnegative, positive, unit_interval


class FocalLoss(BaseLoss):
    """Focal Loss，通过调整样本权重来解决类别不平衡问题。

    数学公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    其中:
        p_t = p if y=1 else 1-p
        α_t = α if y=1 else 1-α

    :param alpha: 正样本权重，默认为0.25，用于平衡正负样本的总体权重
    :param gamma: 聚焦参数，默认为2.0，控制易分类样本的权重衰减程度
        - gamma=0: 等价于类别加权交叉熵
        - gamma越大，易分类样本权重越小
    :param name: 损失函数名称，默认为"focal_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import FocalLoss
    >>>
    >>> # 创建损失函数
    >>> loss = FocalLoss(alpha=0.75, gamma=2.0)
    >>>
    >>> # 计算损失
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在XGBoost中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> params = {'objective': 'binary:logistic'}
    >>> bst = xgb.train(params, dtrain, obj=loss.to_xgboost(), num_boost_round=100)

    **引用**

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for
    Dense Object Detection.* ICCV 2017. https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: str = "focal_loss"):
        super().__init__(name)
        unit_interval(alpha=alpha)
        nonnegative(gamma=gamma)
        self.is_additive = True
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """逐样本损失相对坏样本概率的一阶导。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """逐样本损失相对坏样本概率的二阶导。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        pt = np.where(y == 1, p, 1 - p)
        alpha = np.where(y == 1, self.alpha, 1 - self.alpha)
        value, grad, hess = focal_terms(pt, self.gamma)
        return alpha * value, alpha * grad * (2 * y - 1), alpha * hess

    def loss_values(self, y_true, y_pred):
        """逐样本 Focal 损失，其均值等于本损失。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]
