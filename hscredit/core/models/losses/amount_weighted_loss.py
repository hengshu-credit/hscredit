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
from ._loss_math import binary_inputs, bce_terms, nonnegative, positive, unit_interval


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

    **引用**

    按风险敞口/金额加权，等价于以"预期损失金额"为代价的样本级成本敏感学习，理论框架见
    Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI 2001。属信贷
    业务驱动设计。
    """

    def __init__(
        self,
        amounts: Optional[np.ndarray] = None,
        normalize: bool = True,
        floor_weight: float = 0.1,
        name: str = "amount_weighted_loss",
    ):
        super().__init__(name)
        self.amounts_ = np.asarray(amounts, dtype=float) if amounts is not None else None
        nonnegative(floor_weight=floor_weight)
        self.is_additive = amounts is None
        self.normalize = normalize
        self.floor_weight = floor_weight

    def set_sample_params(
        self,
        amounts: np.ndarray,
    ) -> "AmountWeightedLoss":
        """设置样本级金额参数。

        :param amounts: 金额数组, shape (n_samples,)
            必须非负且有限，顺序与当前输入样本一致。
        :return: self，原地更新；评估验证集时优先用 ``loss.metric(amounts=...)``
            创建副本，避免覆盖训练金额。

        >>> loss = AmountWeightedLoss().set_sample_params(amounts=[1000, 2000])
        >>> metric = loss.metric(amounts=[3000, 4000])
        """
        self.amounts_ = np.asarray(amounts, dtype=float)
        self.is_additive = False
        return self

    def _get_weights(self, n_samples: int) -> np.ndarray:
        """获取归一化后的金额权重。"""
        if self.amounts_ is None:
            return np.ones(n_samples, dtype=float)

        w = self.amounts_.copy()
        if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("金额必须是一维非负有限数组。")
        if len(w) != n_samples:
            raise ValueError(
                f"金额数组长度 ({len(w)}) 与样本数 ({n_samples}) 不一致，" f"请通过 set_sample_params() 更新金额数组。"
            )

        # 下限截断
        w = np.maximum(w, 0)

        if self.normalize:
            mean_w = np.mean(w) + 1e-12
            w = w / mean_w

        # 权重下限
        w = np.maximum(w, self.floor_weight)

        return w

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import AmountWeightedLoss
        >>> loss = AmountWeightedLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import AmountWeightedLoss
        >>> loss = AmountWeightedLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import AmountWeightedLoss
        >>> loss = AmountWeightedLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _external_weight_normalizer(self, y_true, sample_weight):
        weights = self._get_weights(len(y_true))
        normalizer = np.average(weights / np.mean(weights), weights=sample_weight)
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("金额与额外样本权重相乘后的权重总和必须为有限正数")
        return normalizer

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        weights = self._get_weights(len(y))
        mean_weight = np.mean(weights)
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("样本权重的平均值必须是有限正数。")
        weights = weights / mean_weight
        return tuple(weights * term for term in bce_terms(y, p))

    def loss_values(self, y_true, y_pred):
        """逐样本贡献已按整体平均权重归一化；其均值等于加权平均 BCE。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import AmountWeightedLoss
        >>> loss = AmountWeightedLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]


class ExpectedValueLoss(BaseLoss):
    """期望价值损失函数，结合 LGD / EAD / 利率 / 成本进行期望价值优化。

    在信贷全生命周期管理中，不同客户的风险敞口（EAD）、违约损失率（LGD）、
    收益率各不相同。本损失以这些金融参数构造成本加权 BCE 代理，
    返回值不是实际货币利润；metric() 与 business_metric() 均按同一期望价值加权损失评估。

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
        nonnegative(floor_weight=floor_weight)
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

        :param lgd: 违约损失率，标量或一维数组，范围 [0, 1]；None 保持原值。
        :param ead: 违约风险敞口，非负标量或一维数组；None 保持原值。
        :param rate: 年化收益率，非负标量或一维数组；None 保持原值。
        :param cost: 单客运营成本，非负标量或一维数组；None 保持原值。
        :return: self，原地更新；数组须有限且与之后计算的样本顺序一致。

        >>> loss = ExpectedValueLoss().set_sample_params(lgd=0.5, ead=[1000, 2000])
        >>> metric = loss.metric(ead=[3000, 4000])
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

    def _broadcast(self, value, n, default=1.0):
        """检查并广播标量/长度严格匹配的一维金融参数。"""
        arr = np.asarray(default if value is None else value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if arr.shape != (n,) or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"金融参数必须为非负有限标量或长度为 {n} 的一维数组。")
        return arr

    def _get_weights(self, y_true: np.ndarray) -> np.ndarray:
        """计算样本级期望价值权重。"""
        n = len(y_true)

        lgd = self._broadcast(self.lgd, n, 0.5)
        if np.any(lgd > 1):
            raise ValueError("违约损失率 lgd 必须在 [0, 1] 范围内。")
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

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedValueLoss
        >>> loss = ExpectedValueLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedValueLoss
        >>> loss = ExpectedValueLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedValueLoss
        >>> loss = ExpectedValueLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _external_weight_normalizer(self, y_true, sample_weight):
        weights = self._get_weights(y_true)
        normalizer = np.average(weights / np.mean(weights), weights=sample_weight)
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("金融参数与额外样本权重相乘后的权重总和必须为有限正数")
        return normalizer

    def _terms(self, y_true, y_pred):
        y, p = binary_inputs(y_true, y_pred)
        weights = self._get_weights(y)
        mean_weight = np.mean(weights)
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("样本权重的平均值必须是有限正数。")
        weights = weights / mean_weight
        return tuple(weights * term for term in bce_terms(y, p))

    def loss_values(self, y_true, y_pred):
        """逐样本贡献已按整体平均权重归一化；其均值等于加权平均 BCE。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedValueLoss
        >>> loss = ExpectedValueLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]
