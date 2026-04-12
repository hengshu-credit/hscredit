"""
金额加权损失函数

按授信金额、风险敞口或期望价值对样本加权，使模型更关注高金额/高敞口样本的预测准确性。
包含两个损失类:

- :class:`AmountWeightedLoss`: 按授信金额/风险敞口加权
- :class:`ExpectedValueLoss`: 结合 LGD / EAD / 利率 / 成本 的期望价值优化
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .base import BaseLoss


class AmountWeightedLoss(BaseLoss):
    """按授信金额/风险敞口加权的损失函数。

    在信贷场景中，不同客户的授信金额差异很大。一个 10 万元客户的
    坏账和一个 1000 元客户的坏账，业务影响完全不同。本损失通过
    金额加权使模型更关注高金额客户的预测准确性。

    数学形式::

        w_i  = amount_i / mean(amount)      # 归一化权重
        Loss = weighted_mean(BCE_i, w_i)

    :param amounts: 样本级金额/敞口数组, shape (n_samples,)。
        可在构造时传入，也可通过 :meth:`set_sample_params` 动态设置。
    :param normalize: 是否对金额进行均值归一化，默认 True
        - 归一化后平均权重为 1.0，不影响学习率
    :param floor_weight: 权重下限，防止低金额样本完全被忽略，默认 0.1
    :param name: 损失函数名称，默认 "amount_weighted_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import AmountWeightedLoss
    >>>
    >>> amounts = np.array([50000, 100000, 200000, 10000])
    >>> loss = AmountWeightedLoss(amounts=amounts)
    >>>
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 动态设置金额（适用于每轮训练数据不同的场景）
    >>> loss2 = AmountWeightedLoss()
    >>> loss2.set_sample_params(amounts=np.array([30000, 80000, 150000, 5000]))
    """

    def __init__(
        self,
        amounts: Optional[np.ndarray] = None,
        normalize: bool = True,
        floor_weight: float = 0.1,
        name: str = "amount_weighted_loss",
    ):
        super().__init__(name)
        self.amounts_ = (
            np.asarray(amounts, dtype=float) if amounts is not None else None
        )
        self.normalize = normalize
        self.floor_weight = floor_weight

    def set_sample_params(
        self,
        amounts: np.ndarray,
    ) -> "AmountWeightedLoss":
        """设置样本级金额参数。

        :param amounts: 金额数组, shape (n_samples,)
        :return: self
        """
        self.amounts_ = np.asarray(amounts, dtype=float)
        return self

    def _get_weights(self, n_samples: int) -> np.ndarray:
        """获取归一化后的金额权重。"""
        if self.amounts_ is None:
            return np.ones(n_samples, dtype=float)

        w = self.amounts_.copy()
        if len(w) != n_samples:
            raise ValueError(
                f"金额数组长度 ({len(w)}) 与样本数 ({n_samples}) 不一致，"
                f"请通过 set_sample_params() 更新金额数组。"
            )

        # 下限截断
        w = np.maximum(w, 0)

        if self.normalize:
            mean_w = np.mean(w) + 1e-12
            w = w / mean_w

        # 权重下限
        w = np.maximum(w, self.floor_weight)

        return w

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算金额加权损失。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 加权平均损失值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(len(y_true))

        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.average(bce, weights=weights))

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(len(y_true))
        return weights * (y_pred - y_true)

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(len(y_true))
        hess = weights * y_pred * (1 - y_pred)
        return np.maximum(hess, 1e-6)


class ExpectedValueLoss(BaseLoss):
    """期望价值损失函数，结合 LGD / EAD / 利率 / 成本进行期望价值优化。

    在信贷全生命周期管理中，不同客户的风险敞口（EAD）、违约损失率（LGD）、
    收益率各不相同。本损失将这些金融参数融入损失函数，使模型直接优化
    期望经济价值而非简单的分类准确率。

    样本级权重::

        坏样本 (y=1): 权重 = LGD_i × EAD_i         → 漏捕的经济损失
        好样本 (y=0): 权重 = rate_i × EAD_i - cost_i → 误拒的机会成本

    数学形式::

        w_i  = y_i × LGD_i × EAD_i + (1-y_i) × max(rate_i × EAD_i - cost_i, ε)
        Loss = weighted_mean(BCE_i, w_i)

    :param lgd: 违约损失率，标量或数组，默认 0.5
        - 标量: 所有样本使用相同 LGD
        - 数组: 每个样本独立的 LGD, shape (n_samples,)
    :param ead: 违约风险敞口，标量或数组，默认 None（使用全 1）
        - 通常等于授信余额或授信额度
    :param rate: 年化收益率，标量或数组，默认 0.08（8%）
    :param cost: 单客运营成本，标量或数组，默认 0.0
    :param floor_weight: 权重下限，默认 0.1
    :param name: 损失函数名称，默认 "expected_value_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import ExpectedValueLoss
    >>>
    >>> # 全局参数
    >>> loss = ExpectedValueLoss(lgd=0.5, rate=0.12)
    >>>
    >>> # 样本级参数
    >>> loss = ExpectedValueLoss(
    ...     lgd=np.array([0.4, 0.6, 0.5, 0.3]),
    ...     ead=np.array([50000, 100000, 200000, 30000]),
    ...     rate=0.10,
    ...     cost=500
    ... )
    >>>
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 XGBoost 中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> bst = xgb.train({}, dtrain, obj=loss.to_xgboost(), num_boost_round=100)
    """

    def __init__(
        self,
        lgd: Union[float, np.ndarray] = 0.5,
        ead: Optional[Union[float, np.ndarray]] = None,
        rate: Union[float, np.ndarray] = 0.08,
        cost: Union[float, np.ndarray] = 0.0,
        floor_weight: float = 0.1,
        name: str = "expected_value_loss",
    ):
        super().__init__(name)
        self.lgd = lgd
        self.ead = ead
        self.rate = rate
        self.cost = cost
        self.floor_weight = floor_weight

    def set_sample_params(
        self,
        lgd: Optional[Union[float, np.ndarray]] = None,
        ead: Optional[Union[float, np.ndarray]] = None,
        rate: Optional[Union[float, np.ndarray]] = None,
        cost: Optional[Union[float, np.ndarray]] = None,
    ) -> "ExpectedValueLoss":
        """设置样本级金融参数。

        :param lgd: 违约损失率
        :param ead: 违约风险敞口
        :param rate: 年化收益率
        :param cost: 单客运营成本
        :return: self
        """
        if lgd is not None:
            self.lgd = lgd
        if ead is not None:
            self.ead = ead
        if rate is not None:
            self.rate = rate
        if cost is not None:
            self.cost = cost
        return self

    def _broadcast(
        self,
        value: Union[float, np.ndarray, None],
        n: int,
        default: float = 1.0,
    ) -> np.ndarray:
        """将标量或数组广播为 (n,) 数组。"""
        if value is None:
            return np.full(n, default, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        return arr

    def _get_weights(self, y_true: np.ndarray) -> np.ndarray:
        """计算样本级期望价值权重。"""
        n = len(y_true)

        lgd = self._broadcast(self.lgd, n, 0.5)
        ead = self._broadcast(self.ead, n, 1.0)
        rate = self._broadcast(self.rate, n, 0.08)
        cost = self._broadcast(self.cost, n, 0.0)

        # 坏样本权重: 违约损失 = LGD × EAD
        bad_weight = lgd * ead

        # 好样本权重: 机会收益 = rate × EAD - cost
        good_weight = np.maximum(rate * ead - cost, 1e-6)

        # 按标签混合
        weights = y_true * bad_weight + (1 - y_true) * good_weight

        # 归一化
        mean_w = np.mean(weights) + 1e-12
        weights = weights / mean_w

        # 权重下限
        weights = np.maximum(weights, self.floor_weight)

        return weights

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算期望价值损失。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 加权平均损失值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(y_true)

        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.average(bce, weights=weights))

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(y_true)
        return weights * (y_pred - y_true)

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        weights = self._get_weights(y_true)
        hess = weights * y_pred * (1 - y_pred)
        return np.maximum(hess, 1e-6)
