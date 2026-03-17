"""
加权损失函数和成本敏感损失

提供多种加权损失函数，用于处理类别不平衡和成本敏感学习场景。
"""

import numpy as np
from typing import Optional, Union
from .base import BaseLoss


class WeightedBCELoss(BaseLoss):
    """加权二元交叉熵损失，通过为正负样本分配不同权重来处理类别不平衡问题。

    数学公式: Loss = -[w_pos * y * log(p) + w_neg * (1-y) * log(1-p)]

    :param pos_weight: 正样本权重，默认为1.0
    :param neg_weight: 负样本权重，默认为1.0
    :param auto_balance: 是否自动根据样本比例平衡权重，默认为False。如果为True，pos_weight和neg_weight将被忽略
    :param name: 损失函数名称，默认为"weighted_bce"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import WeightedBCELoss
    >>>
    >>> # 手动设置权重
    >>> loss = WeightedBCELoss(pos_weight=5.0, neg_weight=1.0)
    >>>
    >>> # 自动平衡权重
    >>> loss = WeightedBCELoss(auto_balance=True)
    >>> # 假设正样本占比10%，自动设置pos_weight=9, neg_weight=1
    >>>
    >>> # 在LightGBM中使用
    >>> import lightgbm as lgb
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> bst = lgb.train(
    ...     params={'objective': 'binary'},
    ...     train_set=train_data,
    ...     fobj=loss.to_lightgbm(),
    ...     num_boost_round=100
    ... )
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        auto_balance: bool = False,
        name: str = "weighted_bce"
    ):
        super().__init__(name)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.auto_balance = auto_balance
        self._fitted_weights = None

    def _auto_balance_weights(self, y_true: np.ndarray):
        """根据样本比例自动平衡权重。"""
        if not self.auto_balance:
            return

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return

        # 权重与样本数量成反比
        self.pos_weight = n_neg / n_pos
        self.neg_weight = 1.0

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算加权BCE损失。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 平均损失值
        """
        # 自动平衡权重
        self._auto_balance_weights(y_true)

        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算加权交叉熵
        pos_loss = -self.pos_weight * y_true * np.log(y_pred)
        neg_loss = -self.neg_weight * (1 - y_true) * np.log(1 - y_pred)

        total_loss = pos_loss + neg_loss
        return np.mean(total_loss)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算梯度。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组
        """
        # 自动平衡权重
        self._auto_balance_weights(y_true)

        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算梯度
        grad = np.where(
            y_true == 1,
            -self.pos_weight / y_pred,
            self.neg_weight / (1 - y_pred)
        )

        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组
        """
        # 自动平衡权重
        self._auto_balance_weights(y_true)

        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算二阶导
        hess = np.where(
            y_true == 1,
            self.pos_weight / (y_pred ** 2),
            self.neg_weight / ((1 - y_pred) ** 2)
        )

        return hess


class CostSensitiveLoss(BaseLoss):
    """成本敏感损失函数，根据不同预测错误的成本为不同类型的错误分配不同的权重。

    特别适合金融风控场景，因为漏抓坏客户的成本往往远大于误拒好客户的成本。

    :param fn_cost: 假阴性成本（漏抓坏客户的成本），默认为1.0
    :param fp_cost: 假阳性成本（误拒好客户的成本），默认为1.0
    :param name: 损失函数名称，默认为"cost_sensitive"

    **参考样例**

    >>> from hscredit.core.models.losses import CostSensitiveLoss
    >>>
    >>> # 假设漏抓一个坏客户损失10000元，误拒一个好客户损失100元
    >>> # 成本比例为100:1
    >>> loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
    >>>
    >>> # 在模型中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> params = {'objective': 'binary:logistic'}
    >>> bst = xgb.train(params, dtrain, obj=loss.to_xgboost(), num_boost_round=100)

    **注意**

    损失矩阵:
                预测负    预测正
    实际负        0       fp_cost
    实际正     fn_cost      0

    我们希望最小化总成本: FP * fp_cost + FN * fn_cost
    """

    def __init__(
        self,
        fn_cost: float = 1.0,
        fp_cost: float = 1.0,
        name: str = "cost_sensitive"
    ):
        super().__init__(name)
        self.fn_cost = fn_cost  # 假阴性成本（漏抓）
        self.fp_cost = fp_cost  # 假阳性成本（误拒）

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """计算总成本。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :param threshold: 分类阈值，默认为0.5
        :return: 总成本
        """
        # 预测标签
        y_pred_label = (y_pred >= threshold).astype(int)

        # 计算FP和FN
        fp = np.sum((y_true == 0) & (y_pred_label == 1))
        fn = np.sum((y_true == 1) & (y_pred_label == 0))

        # 总成本
        total_cost = fp * self.fp_cost + fn * self.fn_cost

        return total_cost / len(y_true)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算梯度，成本敏感的梯度计算。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组
        """
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 对于正样本(y=1)，我们希望预测概率尽可能高，梯度: -fn_cost / p
        # 对于负样本(y=0)，我们希望预测概率尽可能低，梯度: fp_cost / (1-p)

        grad = np.where(
            y_true == 1,
            -self.fn_cost / y_pred,
            self.fp_cost / (1 - y_pred)
        )

        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组
        """
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 计算二阶导
        hess = np.where(
            y_true == 1,
            self.fn_cost / (y_pred ** 2),
            self.fp_cost / ((1 - y_pred) ** 2)
        )

        return hess
