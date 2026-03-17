"""
风控业务损失函数

针对金融风控场景设计的专用损失函数，考虑坏账率、通过率、利润最大化等业务指标。
"""

import numpy as np
from typing import Optional, Dict
from .base import BaseLoss


class BadDebtLoss(BaseLoss):
    """坏账率优化损失函数，最小化坏账率同时保持通过率在合理水平。

    适用于信贷审批场景，希望降低通过客户的坏账比例。

    :param target_approval_rate: 目标通过率，默认为0.3
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
    >>> # 在CatBoost中使用
    >>> from catboost import CatBoostClassifier
    >>> model = CatBoostClassifier(
    ...     iterations=1000,
    ...     loss_function=loss.to_catboost(),
    ...     eval_metric='AUC'
    ... )
    """

    def __init__(
        self,
        target_approval_rate: float = 0.3,
        bad_debt_weight: float = 1.0,
        approval_weight: float = 0.5,
        name: str = "bad_debt_loss"
    ):
        super().__init__(name)
        self.target_approval_rate = target_approval_rate
        self.bad_debt_weight = bad_debt_weight
        self.approval_weight = approval_weight

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """计算通过率和坏账率。"""
        # 预测标签
        y_pred_label = (y_pred >= threshold).astype(int)

        # 通过率
        approval_rate = np.mean(y_pred_label == 1)

        # 坏账率（通过客户中坏客户的比例）
        approved_mask = y_pred_label == 1
        if np.sum(approved_mask) == 0:
            bad_debt_rate = 0.0
        else:
            bad_debt_rate = np.mean(y_true[approved_mask])

        return {
            'approval_rate': approval_rate,
            'bad_debt_rate': bad_debt_rate
        }

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算损失。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 损失值
        """
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 标准交叉熵作为基础损失
        ce_loss = -(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        # 找到使通过率接近目标的阈值
        sorted_pred = np.sort(y_pred)[::-1]
        threshold_idx = int(len(sorted_pred) * (1 - self.target_approval_rate))
        threshold = sorted_pred[threshold_idx] if threshold_idx < len(sorted_pred) else sorted_pred[-1]

        # 计算当前通过率和坏账率
        metrics = self._compute_metrics(y_true, y_pred, threshold)

        # 损失 = 坏账率损失 + 通过率偏离惩罚
        bad_debt_loss = metrics['bad_debt_rate'] * self.bad_debt_weight
        approval_loss = abs(metrics['approval_rate'] - self.target_approval_rate) * self.approval_weight

        # 总损失
        total_loss = np.mean(ce_loss) + bad_debt_loss + approval_loss

        return total_loss

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
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 基础梯度（交叉熵梯度）
        grad = y_pred - y_true

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

        # 交叉熵二阶导
        hess = y_pred * (1 - y_pred)

        # 确保非零
        hess = np.maximum(hess, 1e-6)

        return hess


class ApprovalRateLoss(BaseLoss):
    """通过率优化损失函数，在保证坏账率不超过目标的前提下最大化通过率。

    :param target_bad_debt_rate: 目标坏账率，默认为0.05
    :param name: 损失函数名称，默认为"approval_rate_loss"

    **参考样例**

    >>> from hscredit.core.models.losses import ApprovalRateLoss
    >>>
    >>> # 目标坏账率不超过5%
    >>> loss = ApprovalRateLoss(target_bad_debt_rate=0.05)
    """

    def __init__(
        self,
        target_bad_debt_rate: float = 0.05,
        name: str = "approval_rate_loss"
    ):
        super().__init__(name)
        self.target_bad_debt_rate = target_bad_debt_rate

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算损失。"""
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 基础交叉熵损失
        ce_loss = -(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return np.mean(ce_loss)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算梯度。"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        grad = y_pred - y_true
        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算二阶导。"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        hess = y_pred * (1 - y_pred)
        hess = np.maximum(hess, 1e-6)
        return hess


class ProfitMaxLoss(BaseLoss):
    """利润最大化损失函数，综合考虑坏账损失和利息收益最大化总利润。

    利润模型: 利润 = 通过客户数 * (利息收益 - 坏账率 * 坏账损失)

    :param interest_income: 单位利息收益，默认为1.0
    :param bad_debt_loss: 单位坏账损失，默认为10.0
    :param name: 损失函数名称，默认为"profit_max_loss"

    **参考样例**

    >>> from hscredit.core.models.losses import ProfitMaxLoss
    >>>
    >>> # 假设每笔贷款利息收益100元，坏账损失1000元
    >>> loss = ProfitMaxLoss(interest_income=100, bad_debt_loss=1000)
    """

    def __init__(
        self,
        interest_income: float = 1.0,
        bad_debt_loss: float = 10.0,
        name: str = "profit_max_loss"
    ):
        super().__init__(name)
        self.interest_income = interest_income
        self.bad_debt_loss = bad_debt_loss

    def _compute_profit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float
    ) -> float:
        """计算总利润。"""
        # 预测标签
        y_pred_label = (y_pred >= threshold).astype(int)

        # 通过客户数
        approved_mask = y_pred_label == 1
        n_approved = np.sum(approved_mask)

        if n_approved == 0:
            return 0.0

        # 通过客户中的好客户和坏客户
        y_true_approved = y_true[approved_mask]
        n_good = np.sum(y_true_approved == 0)
        n_bad = np.sum(y_true_approved == 1)

        # 总利润 = 好客户收益 - 坏客户损失
        total_profit = (
            n_good * self.interest_income -
            n_bad * self.bad_debt_loss
        )

        return total_profit / len(y_true)  # 人均利润

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算损失（负利润）。"""
        # 确保概率在合理范围
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 基础交叉熵损失
        ce_loss = -(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return np.mean(ce_loss)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算梯度。"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 根据利润模型调整梯度
        # 好客户(y=0): 希望预测概率低（拒绝），如果预测高则惩罚
        # 坏客户(y=1): 希望预测概率高（拒绝），如果预测低则重惩罚

        grad = np.where(
            y_true == 0,
            (y_pred - 0) * self.interest_income,  # 好客户梯度
            (y_pred - 1) * self.bad_debt_loss      # 坏客户梯度
        )

        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """计算二阶导。"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        hess = np.where(
            y_true == 0,
            self.interest_income,
            self.bad_debt_loss
        )

        return hess
