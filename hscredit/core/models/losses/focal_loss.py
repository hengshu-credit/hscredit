"""
Focal Loss - 处理类别不平衡的损失函数

Focal Loss通过降低易分类样本的权重，专注于难分类样本，特别适合金融风控场景中的
不平衡数据问题（如坏账率通常很低）。
"""

import numpy as np
from typing import Optional
from .base import BaseLoss


class FocalLoss(BaseLoss):
    """Focal Loss，通过调整样本权重来解决类别不平衡问题。

    数学公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    其中:
        p_t = p if y=1 else 1-p
        α_t = α if y=1 else 1-α

    :param alpha: 正样本权重，默认为0.25，用于平衡正负样本的总体权重
    :param gamma: 聚焦参数，默认为2.0，控制易分类样本的权重衰减程度
        - gamma=0: 等价于标准交叉熵
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

    **参考**

    Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        name: str = "focal_loss"
    ):
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算Focal Loss。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 平均损失值
        """
        # 确保概率在[0, 1]范围内
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算p_t
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)

        # 计算alpha_t
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)

        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 计算交叉熵
        ce_loss = -np.log(p_t)

        # 计算focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        return np.mean(focal_loss)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算Focal Loss的梯度（一阶导数）。

        推导过程: d(FL)/d(p) = d/dp [ -α_t * (1-p_t)^γ * log(p_t) ]

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组
        """
        # 确保概率在合理范围内
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算p_t
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)

        # 计算alpha_t
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)

        # 计算梯度
        grad = np.zeros_like(y_pred)

        # 正样本梯度
        pos_mask = y_true == 1
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            grad[pos_mask] = (
                alpha_t[pos_mask] * (
                    self.gamma * (1 - p) ** (self.gamma - 1) * np.log(p) -
                    (1 - p) ** self.gamma / p
                )
            )

        # 负样本梯度
        neg_mask = y_true == 0
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            grad[neg_mask] = (
                alpha_t[neg_mask] * (
                    -self.gamma * p ** (self.gamma - 1) * np.log(1 - p) +
                    p ** self.gamma / (1 - p)
                )
            )

        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算Focal Loss的二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组
        """
        # 确保概率在合理范围内
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算alpha_t
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)

        # 计算二阶导数（简化版本）
        hess = np.zeros_like(y_pred)

        # 正样本二阶导
        pos_mask = y_true == 1
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            hess[pos_mask] = alpha_t[pos_mask] * (
                self.gamma * (self.gamma - 1) * (1 - p) ** (self.gamma - 2) * np.log(p) +
                2 * self.gamma * (1 - p) ** (self.gamma - 1) / p +
                (1 - p) ** self.gamma / (p ** 2)
            )

        # 负样本二阶导
        neg_mask = y_true == 0
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            hess[neg_mask] = alpha_t[neg_mask] * (
                self.gamma * (self.gamma - 1) * p ** (self.gamma - 2) * np.log(1 - p) +
                2 * self.gamma * p ** (self.gamma - 1) / (1 - p) +
                p ** self.gamma / ((1 - p) ** 2)
            )

        # 确保二阶导为正
        hess = np.abs(hess) + 1e-6

        return hess
