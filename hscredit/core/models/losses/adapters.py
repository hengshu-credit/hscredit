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
from .base import BaseLoss, BaseMetric, _margin_derivatives


class XGBoostLossAdapter:
    """XGBoost损失函数适配器.

    将自定义损失函数转换为XGBoost可用的格式。

    :param loss: 损失函数对象

    **参考样例**

    >>> import xgboost as xgb
    >>> from hscredit.core.models.losses import FocalLoss, XGBoostLossAdapter
    >>>
    >>> # 创建损失函数
    >>> loss = FocalLoss(alpha=0.75, gamma=2.0)
    >>> adapter = XGBoostLossAdapter(loss)
    >>>
    >>> # 在XGBoost中使用
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> params = {
    ...     'objective': 'binary:logistic',
    ...     'eval_metric': 'auc'
    ... }
    >>> bst = xgb.train(
    ...     params,
    ...     dtrain,
    ...     obj=adapter.objective(),
    ...     num_boost_round=100
    ... )
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self) -> Callable:
        """获取XGBoost目标函数.

        :return: XGBoost格式的目标函数
        """

        def xgb_objective(preds: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
            """XGBoost目标函数格式.

            :param preds: 预测值（原始分数，需要转换为概率）
            :param dtrain: 训练数据
            :return: (梯度, 二阶导数)
            """
            # 获取标签
            labels = dtrain.get_label()

            # 将原始分数转换为概率（sigmoid）
            probs = 1.0 / (1.0 + np.exp(-preds))

            return _margin_derivatives(self.loss, labels, probs)

        return xgb_objective

    def metric(self, metric: BaseMetric) -> Callable:
        """获取XGBoost评估指标.

        :param metric: 评估指标对象
        :return: XGBoost格式的评估指标
        """

        def xgb_metric(preds: np.ndarray, dtrain) -> Tuple[str, float]:
            """XGBoost评估指标格式.

            :param preds: 预测值（原始分数）
            :param dtrain: 数据
            :return: (指标名称, 指标值)
            """
            labels = dtrain.get_label()
            probs = 1.0 / (1.0 + np.exp(-preds))
            value = metric(labels, probs)
            return metric.name, value

        return xgb_metric


class LightGBMLossAdapter:
    """LightGBM损失函数适配器.

    将自定义损失函数转换为LightGBM可用的格式。

    :param loss: 损失函数对象

    **参考样例**

    >>> import lightgbm as lgb
    >>> from hscredit.core.models.losses import CostSensitiveLoss, LightGBMLossAdapter
    >>>
    >>> loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
    >>> adapter = LightGBMLossAdapter(loss)
    >>>
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> # objective() 采用 (y_true, y_pred) -> (grad, hess) 约定，
    >>> # 通过 params['objective'] 传入（LightGBM 4.0 起已移除 fobj 参数）
    >>> bst = lgb.train(
    ...     params={'objective': adapter.objective(), 'metric': 'auc'},
    ...     train_set=train_data,
    ...     num_boost_round=100
    ... )
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self) -> Callable:
        """获取LightGBM目标函数.

        :return: LightGBM格式的目标函数
        """

        def lgb_objective(y_true: np.ndarray, y_pred: np.ndarray):
            """LightGBM目标函数格式.

            :param y_true: 真实标签
            :param y_pred: 预测值（原始分数）
            :return: (梯度, 二阶导数)
            """
            # 将原始分数转换为概率
            probs = 1.0 / (1.0 + np.exp(-y_pred))

            return _margin_derivatives(self.loss, y_true, probs)

        return lgb_objective

    def metric(self, metric: BaseMetric) -> Callable:
        """获取LightGBM评估指标.

        :param metric: 评估指标对象
        :return: LightGBM格式的评估指标
        """

        def lgb_metric(y_true: np.ndarray, y_pred: np.ndarray):
            """LightGBM评估指标格式.

            :param y_true: 真实标签
            :param y_pred: 预测值（原始分数）
            :return: (指标名称, 指标值, 是否越大越好)
            """
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            value = metric(y_true, probs)
            return metric.name, value, metric.greater_is_better

        return lgb_metric


class CatBoostLossAdapter:
    """CatBoost损失函数适配器.

    将自定义损失函数转换为CatBoost可用的格式。

    :param loss: 损失函数对象

    **参考样例**

    >>> from catboost import CatBoostClassifier
    >>> from hscredit.core.models.losses import BadDebtLoss, CatBoostLossAdapter
    >>>
    >>> loss = BadDebtLoss(target_approval_rate=0.3)
    >>> adapter = CatBoostLossAdapter(loss)
    >>>
    >>> model = CatBoostClassifier(
    ...     iterations=1000,
    ...     loss_function=adapter.objective(),
    ...     eval_metric='AUC'
    ... )
    >>> model.fit(X_train, y_train)
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def objective(self):
        """获取CatBoost目标函数.

        :return: CatBoost格式的目标函数类
        """
        loss_obj = self.loss

        class CatBoostLoss:
            def calc_ders_range(self, approxes, targets, weights):
                """计算梯度和二阶导.

                CatBoost 对一批样本调用本方法，``approxes``/``targets``/``weights``
                均为与样本一一对应的原始分数/标签/权重数组（二分类单目标）。

                注意符号约定：CatBoost 执行梯度上升以最小化损失，要求返回
                ``der1 = -dL/dapprox``、``der2 = -d²L/dapprox²``，因此对 ``BaseLoss``
                给出的（定义在概率上的）梯度/二阶导取负。

                :param approxes: 预测原始分数数组（每个样本一个值）
                :param targets: 真实标签数组
                :param weights: 样本权重数组，可为 None
                :return: 每个元素为 (一阶导, 二阶导) 的列表
                """
                approx = np.asarray(approxes, dtype=float)
                target = np.asarray(targets, dtype=float)

                # 将原始分数转换为概率
                probs = 1.0 / (1.0 + np.exp(-approx))

                # 转换为相对于 raw margin 的真实导数。
                grad, hess = _margin_derivatives(loss_obj, target, probs)

                # CatBoost 约定：梯度上升最小化损失，取负
                der1 = -grad
                der2 = -hess

                # 应用样本权重
                if weights is not None:
                    w = np.asarray(weights, dtype=float)
                    der1 = der1 * w
                    der2 = der2 * w

                return list(zip(der1.tolist(), der2.tolist()))

        return CatBoostLoss()

    def metric(self, metric: BaseMetric):
        """获取CatBoost评估指标.

        :param metric: 评估指标对象
        :return: CatBoost格式的评估指标类
        """
        metric_obj = metric

        class CatBoostMetric:
            def get_final_error(self, error, weight):
                return error

            def is_max_optimal(self):
                return metric_obj.greater_is_better

            def evaluate(self, approxes, target, weight):
                """计算指标值.

                :param approxes: 预测值列表
                :param target: 真实标签列表
                :param weight: 样本权重
                :return: (指标值, 样本数量)
                """
                assert len(approxes) == 1
                approx = np.array(approxes[0])
                target = np.array(target)

                # 转换为概率
                probs = 1.0 / (1.0 + np.exp(-approx))

                # 计算指标值
                value = metric_obj(target, probs)

                return value, len(target)

        return CatBoostMetric()


def _tabnet_binary_loss_and_gradient(loss_obj: BaseLoss, logits, y_true):
    """计算 TabNet 二分类 raw logits 对应的平均损失和梯度。"""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)

    if logits.ndim == 1:
        if len(logits) != len(y_true):
            raise ValueError("TabNet预测行数与标签行数不一致")
        probability = 1.0 / (1.0 + np.exp(-logits))
        probability_gradient = np.asarray(loss_obj.gradient(y_true, probability), dtype=float) / max(1, len(y_true))
        gradient = probability_gradient * probability * (1.0 - probability)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        if logits.shape[0] != len(y_true):
            raise ValueError("TabNet预测行数与标签行数不一致")
        probability = 1.0 / (1.0 + np.exp(-logits[:, 0]))
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
        """获取PyTorch损失函数.

        :return: PyTorch格式的损失函数
        """
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
    ...     Score=adapter.score_class(),
    ...     n_estimators=500,
    ...     learning_rate=0.01
    ... )
    >>> model.fit(X_train, y_train)

    **注意**

    - 仅支持 ``Dist=Bernoulli``（NGBoost默认二分类分布）
    - ``score()`` 使用标准BCE作为监控/早停指标
    - ``d_score()`` 使用自定义loss的梯度驱动自然梯度更新
    - 也可直接使用 ``BaseLoss.to_ngboost()`` 快捷方法
    """

    def __init__(self, loss: BaseLoss):
        self.loss = loss

    def score_class(self):
        """获取NGBoost Score子类.

        :return: NGBoost Score子类（未实例化），可直接传给 ``NGBClassifier(Score=...)``
        """
        return self.loss.to_ngboost()
