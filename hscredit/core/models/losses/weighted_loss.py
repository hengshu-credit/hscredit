"""
加权损失函数和成本敏感损失

提供多种加权损失函数，用于处理类别不平衡和成本敏感学习场景。
"""

import numpy as np
from typing import Optional, Union
from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, focal_terms, nonnegative, positive, unit_interval


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
    >>> # 同一目标的离线评估：越小越好
    >>> value = loss.metric()(y_valid, p_valid)
    >>> # 在LightGBM 4.x 原生接口中使用
    >>> import lightgbm as lgb
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> bst = lgb.train(
    ...     params={'objective': loss.to_lightgbm(api='native'), 'metric': 'None'},
    ...     train_set=train_data,
    ...     feval=loss.metric().to_lightgbm(api='native', raw_score=True),
    ...     num_boost_round=100
    ... )

    **引用**

    类别加权（class weighting）是处理不平衡数据的经典成本敏感方法，参见 King, G., & Zeng,
    L. (2001). *Logistic Regression in Rare Events Data.* Political Analysis 9(2)，
    以及 Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI 2001。
    """

    def __init__(
        self, pos_weight: float = 1.0, neg_weight: float = 1.0, auto_balance: bool = False, name: str = "weighted_bce"
    ):
        super().__init__(name)
        nonnegative(pos_weight=pos_weight, neg_weight=neg_weight)
        positive(权重总和=pos_weight + neg_weight)
        self.is_additive = not auto_balance
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
            self.pos_weight = self.neg_weight = 1.0
            return

        # 权重与样本数量成反比
        self.pos_weight = n_neg / n_pos
        self.neg_weight = 1.0

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import WeightedBCELoss
        >>> loss = WeightedBCELoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import WeightedBCELoss
        >>> loss = WeightedBCELoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import WeightedBCELoss
        >>> loss = WeightedBCELoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        self._auto_balance_weights(y)
        weights = np.where(y == 1, self.pos_weight, self.neg_weight)
        return tuple(weights * term for term in bce_terms(y, p))

    def loss_values(self, y_true, y_pred):
        """逐样本类别加权 BCE；自动权重按当前数据集重新计算。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import WeightedBCELoss
        >>> loss = WeightedBCELoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]


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

    损失矩阵::

                    预测负    预测正
        实际负        0       fp_cost
        实际正     fn_cost      0

    我们希望最小化总成本: FP * fp_cost + FN * fn_cost
    """

    def __init__(self, fn_cost: float = 1.0, fp_cost: float = 1.0, name: str = "cost_sensitive"):
        super().__init__(name)
        nonnegative(fn_cost=fn_cost, fp_cost=fp_cost)
        positive(成本总和=fn_cost + fp_cost)
        self.is_additive = True
        self.fn_cost = fn_cost  # 假阴性成本（漏抓）
        self.fp_cost = fp_cost  # 假阳性成本（误拒）

    def __call__(self, y_true, y_pred, threshold=None) -> float:
        """默认返回可优化的成本加权 BCE；显式 threshold 返回历史硬分类成本。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :param threshold: 默认 None 使用可微 BCE 代理；显式阈值返回历史硬分类成本，范围 [0, 1]。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import CostSensitiveLoss
        >>> loss = CostSensitiveLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        if threshold is not None:
            return self.classification_cost(y_true, y_pred, threshold)
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import CostSensitiveLoss
        >>> loss = CostSensitiveLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import CostSensitiveLoss
        >>> loss = CostSensitiveLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        weights = np.where(y == 1, self.fn_cost, self.fp_cost)
        return tuple(weights * term for term in bce_terms(y, p))

    def classification_cost(self, y_true, y_pred, threshold=0.5) -> float:
        """计算固定阈值下的实际人均误判成本，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状的坏样本概率。
        :param threshold: 拒绝阈值，范围 [0, 1]，默认 0.5；p >= threshold 拒绝。
        :return: float，误拒好客户成本和漏抓坏客户成本之和除以全量人数。
            这是硬决策评价，不作为可微训练目标。

        >>> cost = CostSensitiveLoss(fn_cost=100, fp_cost=1).classification_cost(
        ...     [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], threshold=0.5)
        """
        y, p = binary_inputs(y_true, y_pred)
        p = np.asarray(y_pred, dtype=float)
        unit_interval(threshold=threshold)
        return float(np.mean((y == 0) * (p >= threshold) * self.fp_cost + (y == 1) * (p < threshold) * self.fn_cost))

    def loss_values(self, y_true, y_pred):
        """成本加权 BCE 的逐样本贡献，其均值与默认调用一致。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import CostSensitiveLoss
        >>> loss = CostSensitiveLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]
