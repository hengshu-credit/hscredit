"""框架适配器.

为XGBoost、LightGBM、CatBoost、TabNet、NGBoost等框架提供统一的损失函数和评估指标接口。
"""

from typing import Callable, Union, Tuple
import numpy as np
from .base import BaseLoss, BaseMetric


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

            # 计算梯度和二阶导
            grad = self.loss.gradient(labels, probs)
            hess = self.loss.hessian(labels, probs)

            if hess is None:
                hess = np.ones_like(grad) * 0.5

            return grad, hess

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
    >>> bst = lgb.train(
    ...     params={'objective': 'binary', 'metric': 'auc'},
    ...     train_set=train_data,
    ...     fobj=adapter.objective(),
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

            # 计算梯度和二阶导
            grad = self.loss.gradient(y_true, probs)
            hess = self.loss.hessian(y_true, probs)

            if hess is None:
                hess = np.ones_like(grad) * 0.5

            return grad, hess

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

                :param approxes: 预测值列表（每个元素对应一个类别）
                :param targets: 真实标签列表
                :param weights: 样本权重
                :return: 每个元素为(梯度, 二阶导)的列表
                """
                # CatBoost的approxes是列表，对于二分类只有一个元素
                approx = np.array(approxes[0])
                targets = np.array(targets)

                # 将原始分数转换为概率
                probs = 1.0 / (1.0 + np.exp(-approx))

                # 计算梯度和二阶导
                grad = loss_obj.gradient(targets, probs)
                hess = loss_obj.hessian(targets, probs)

                if hess is None:
                    hess = np.ones_like(grad) * 0.5

                # CatBoost返回元组列表
                result = list(zip(grad, hess))

                return result

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
            def forward(self, y_pred, y_true):
                """计算损失.

                :param y_pred: 预测概率
                :param y_true: 真实标签
                :return: 损失值
                """
                # 转换为numpy计算
                y_pred_np = y_pred.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()

                # 使用自定义损失计算
                loss_value = loss_obj(y_true_np, y_pred_np)

                # 转换回torch tensor
                return torch.tensor(loss_value, dtype=torch.float32, device=y_pred.device)

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
