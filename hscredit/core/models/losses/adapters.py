"""框架适配器.

为 XGBoost、LightGBM、CatBoost、TabNet、NGBoost 等框架提供统一的自定义损失函数与
评估指标接口。各适配器统一处理 sigmoid 链接函数（原始分数→概率）及各框架的符号/接口
约定，使同一个 :class:`~hscredit.core.models.losses.base.BaseLoss` 可在不同框架间复用。
通常无需直接使用本模块，优先用 ``loss.to_xgboost()`` / ``to_lightgbm()`` /
``to_catboost()`` / ``to_ngboost()`` 便捷方法。

**引用（各框架自定义目标/评估文档）**

- XGBoost 自定义目标：https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
- LightGBM 自定义目标（4.0+ 通过 ``params['objective']`` 传入）：
  https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
- CatBoost 自定义损失（``calc_ders_range``）：
  https://catboost.ai/docs/concepts/python-usages-examples
- NGBoost 自定义 Score：https://stanfordmlgroup.github.io/ngboost/
"""

from typing import Callable, Tuple
import numpy as np
from .base import BaseLoss, BaseMetric
from scipy.special import expit


class XGBoostLossAdapter:
    """XGBoost 目标与配套指标，默认使用原生 train 接口。

    **参数**

    :param loss: BaseLoss 实例，训练接收一维原始分数并自动处理样本权重。

    **属性**

    ``loss`` 保存目标对象；指标默认使用独立的损失参数快照。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss, XGBoostLossAdapter
    >>> adapter = XGBoostLossAdapter(FocalLoss(alpha=0.75))
    >>> objective = adapter.objective(api='sklearn')
    >>> eval_metric = adapter.metric(api='sklearn')
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self, api="native"):
        """生成训练目标。

        :param api: ``'native'`` 接收 (preds, DMatrix)，``'sklearn'`` 接收
            (y_true, y_pred, sample_weight=None)；预测均为原始分数。
        :return: 回调，返回对原始分数求导的 (grad, hess) 数组。

        >>> from hscredit.core.models.losses import FocalLoss, XGBoostLossAdapter
        >>> objective = XGBoostLossAdapter(FocalLoss()).objective()
        """
        return self.loss.to_xgboost(api=api)

    def metric(self, metric=None, api="native"):
        """生成配合本自定义目标使用的评估回调，自动转换原始分数。

        :param metric: BaseMetric 实例，默认 ``loss.metric()``。
        :param api: ``'native'`` 或 ``'sklearn'``，须与训练接口相同。
        :return: native 返回 (name, value)，sklearn 返回 value 的回调。
            不改变指标符号，早停方向须按 metric.greater_is_better 配置。

        >>> from hscredit.core.models.losses import FocalLoss, AUCMetric, XGBoostLossAdapter
        >>> callback = XGBoostLossAdapter(FocalLoss()).metric(AUCMetric())
        """
        metric = self.loss.metric() if metric is None else metric
        return metric.to_xgboost(api=api, raw_score=True)


class LightGBMLossAdapter:
    """LightGBM 目标与配套指标，原生 train 需指定 api='native'。

    **参数**

    :param loss: BaseLoss 实例，训练输入为一维原始分数。

    **属性**

    ``loss`` 保存损失对象，样本权重与链接函数由目标回调统一处理。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss, LightGBMLossAdapter
    >>> adapter = LightGBMLossAdapter(FocalLoss())
    >>> objective, eval_metric = adapter.objective(), adapter.metric()
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self, api="sklearn"):
        """生成接收原始分数并返回 (grad, hess) 的训练回调。

        :param api: 默认 ``'sklearn'`` 接收 (y_true, y_pred, sample_weight=None)；
            ``'native'`` 接收 (preds, Dataset)，用于 params['objective']。
        :return: 目标回调，自动应用 sigmoid 和数据集权重。

        >>> from hscredit.core.models.losses import FocalLoss, LightGBMLossAdapter
        >>> objective = LightGBMLossAdapter(FocalLoss()).objective(api='native')
        """
        return self.loss.to_lightgbm(api=api)

    def metric(self, metric=None, api="sklearn"):
        """生成自动转换原始分数的评估回调，和自定义目标配套使用。

        :param metric: BaseMetric 实例，默认 ``loss.metric()``。
        :param api: ``'sklearn'`` 或 ``'native'``，须与训练接口相同。
        :return: 回调，返回 (name, value, greater_is_better)。

        >>> from hscredit.core.models.losses import FocalLoss, KSMetric, LightGBMLossAdapter
        >>> callback = LightGBMLossAdapter(FocalLoss()).metric(KSMetric())
        """
        metric = self.loss.metric() if metric is None else metric
        return metric.to_lightgbm(api=api, raw_score=True)


class CatBoostLossAdapter:
    """CatBoost 逐样本损失与配套指标。

    **参数**

    :param loss: 支持独立逐行计算的 BaseLoss，例如固定类别权重的 FocalLoss。

    **属性**

    ``loss`` 保存损失对象；非可加损失会在生成训练目标时拒绝。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss, CatBoostLossAdapter
    >>> adapter = CatBoostLossAdapter(FocalLoss())
    >>> objective, eval_metric = adapter.objective(), adapter.metric()
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self):
        """生成 CatBoost 目标对象。

        :return: 对象，calc_ders_range 接收原始分数、0/1 标签和可选权重，
            输出逐样本 (-grad, -hess)，符合 CatBoost 的符号约定。
        :raises ValueError: 损失依赖全量排序、动态类别权重或绑定逐行数组。

        >>> from hscredit.core.models.losses import FocalLoss, CatBoostLossAdapter
        >>> objective = CatBoostLossAdapter(FocalLoss()).objective()
        """
        from .base import _objective_derivatives

        if not self.loss.is_additive:
            raise ValueError("CatBoost会分批计算梯度，不支持依赖全量排序、动态类别权重或样本金额数组的损失")
        loss = self.loss

        class CatBoostLoss:
            def calc_ders_range(self, approxes, targets, weights):
                grad, hess = _objective_derivatives(loss, targets, approxes, weights)
                return list(zip((-grad).tolist(), (-hess).tolist()))

        return CatBoostLoss()

    def metric(self, metric=None):
        """生成 CatBoost 指标对象，按全量验证集保留指标语义。

        :param metric: BaseMetric 实例，默认 loss.metric()。
        :return: 对象，evaluate 接收原始分数并返回 (误差和, 权重和)，
            get_final_error 还原指标值，is_max_optimal 返回优化方向。

        >>> from hscredit.core.models.losses import FocalLoss, CatBoostLossAdapter
        >>> callback = CatBoostLossAdapter(FocalLoss()).metric()
        """
        metric = self.loss.metric() if metric is None else metric
        return metric.to_catboost()


def _tabnet_binary_loss_and_gradient(loss_obj: BaseLoss, logits, y_true):
    """计算 TabNet 二分类 raw logits 对应的平均损失和梯度。"""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)

    if logits.ndim == 1:
        if len(logits) != len(y_true):
            raise ValueError("TabNet预测行数与标签行数不一致")
        probability = expit(logits)
        probability_gradient = np.asarray(loss_obj.gradient(y_true, probability), dtype=float) / max(1, len(y_true))
        gradient = probability_gradient * probability * (1.0 - probability)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        if logits.shape[0] != len(y_true):
            raise ValueError("TabNet预测行数与标签行数不一致")
        probability = expit(logits[:, 0])
        probability_gradient = np.asarray(loss_obj.gradient(y_true, probability), dtype=float) / max(1, len(y_true))
        gradient = (probability_gradient * probability * (1.0 - probability))[:, None]
    elif logits.ndim == 2 and logits.shape[1] == 2:
        if logits.shape[0] != len(y_true):
            raise ValueError("TabNet预测行数与标签行数不一致")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probability = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
        probability_gradient = np.asarray(loss_obj.gradient(y_true, probability), dtype=float) / max(1, len(y_true))
        logit_gradient = probability_gradient * probability * (1.0 - probability)
        gradient = np.column_stack([-logit_gradient, logit_gradient])
    else:
        raise ValueError("TabNet二分类损失仅支持一维、单列或两列raw logits")

    return float(loss_obj(y_true, probability)), gradient


class TabNetLossAdapter:
    """TabNet损失函数适配器.

    将自定义损失函数转换为PyTorch可用的格式，适用于TabNet。

    :param loss: 损失函数对象

    **参考样例**

    >>> from pytorch_tabnet.tab_model import TabNetClassifier
    >>> from hscredit.core.models.losses import FocalLoss, TabNetLossAdapter
    >>>
    >>> loss = FocalLoss(alpha=0.75, gamma=2.0)
    >>> adapter = TabNetLossAdapter(loss)
    >>>
    >>> model = TabNetClassifier()
    >>> model.fit(
    ...     X_train, y_train,
    ...     loss_fn=adapter.loss_fn(),
    ...     max_epochs=100
    ... )

    **注意**

    TabNet使用PyTorch，因此需要PyTorch环境。
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def loss_fn(self):
        """获取支持 PyTorch 反向传播的二分类损失模块。

        :return: ``nn.Module``，调用签名为 (y_pred, y_true)。y_pred 是一维、
            单列或两列原始 logits，y_true 是同批 0/1 标签；返回平均损失张量。
            NumPy 计算的解析梯度由自定义 autograd 返回到 logits。
        :raises ImportError: 未安装 PyTorch。
        :raises ValueError: 损失依赖全量排序、动态类别权重或绑定样本数组。

        >>> from hscredit.core.models.losses import FocalLoss, TabNetLossAdapter
        >>> loss_fn = TabNetLossAdapter(FocalLoss()).loss_fn()  # doctest: +SKIP
        """
        if not self.loss.is_additive:
            raise ValueError("TabNet按小批次训练，不支持依赖全量排序、动态类别权重或绑定样本数组的损失")
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("TabNet需要PyTorch环境，请先安装: pip install torch")

        loss_obj = self.loss

        class CustomLoss(nn.Module):
            class _NumpyLossFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, y_pred, y_true):
                    loss_value, gradient = _tabnet_binary_loss_and_gradient(
                        loss_obj,
                        y_pred.detach().cpu().numpy(),
                        y_true.detach().cpu().numpy(),
                    )
                    gradient_tensor = torch.as_tensor(gradient, dtype=y_pred.dtype, device=y_pred.device)
                    ctx.save_for_backward(gradient_tensor)
                    return y_pred.new_tensor(loss_value)

                @staticmethod
                def backward(ctx, grad_output):
                    (gradient_tensor,) = ctx.saved_tensors
                    return grad_output * gradient_tensor, None

            def forward(self, y_pred, y_true):
                """计算损失.

                :param y_pred: 二分类raw logits（一维、单列或两列）
                :param y_true: 真实标签
                :return: 损失值
                """
                return self._NumpyLossFunction.apply(y_pred, y_true)

        return CustomLoss()


class NGBoostLossAdapter:
    """NGBoost损失函数适配器.

    将自定义损失函数转换为NGBoost可用的Score类。

    NGBoost使用自然梯度 + 概率分布框架，与XGBoost/LightGBM的 ``(grad, hess)``
    接口完全不同。本适配器通过链式法则将 ``BaseLoss`` 的梯度（对概率 p 求导）
    转换为NGBoost所需的分布参数梯度（对 logit 求导）::

        dL/d(logit) = dL/dp × dp/d(logit) = dL/dp × p × (1 - p)

    :param loss: 损失函数对象

    **参考样例**

    >>> from ngboost import NGBClassifier
    >>> from hscredit.core.models.losses import ExpectedProfitLoss, NGBoostLossAdapter
    >>>
    >>> loss = ExpectedProfitLoss(revenue=100, default_cost=1000)
    >>> adapter = NGBoostLossAdapter(loss)
    >>>
    >>> model = NGBClassifier(
    ...     **loss.ngboost_params(),
    ...     n_estimators=500,
    ...     learning_rate=0.01
    ... )
    >>> model.fit(X_train, y_train)

    **注意**

    - 仅支持二分类 Bernoulli 分布；需使用本损失提供的分布子类。
    - ``score()`` 与 ``d_score()`` 使用同一个自定义损失
    - ``d_score()`` 使用自定义loss的梯度驱动自然梯度更新
    - 推荐使用 ``loss.ngboost_params()`` 同时配套 Score 与 Dist。
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def score_class(self):
        """生成 NGBoost Score 子类，必须搭配返回类的 distribution。

        :return: Score 类，score 返回逐行损失，d_score 返回 logit 导数，
            metric 返回 Bernoulli Fisher 信息；不直接接收框架外部预测。
        :raises ImportError: 未安装 NGBoost。
        :raises ValueError: 损失不支持独立逐行计算。

        >>> from hscredit.core.models.losses import FocalLoss, NGBoostLossAdapter
        >>> from ngboost import NGBClassifier
        >>> score = NGBoostLossAdapter(FocalLoss()).score_class()
        >>> model = NGBClassifier(Score=score, Dist=score.distribution)
        """
        return self.loss.to_ngboost()
