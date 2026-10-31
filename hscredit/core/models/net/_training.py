"""可训练 net 模型的损失与优化器配置，保持 torch 懒加载。"""

import copy
import inspect
from functools import partial

from ....exceptions import ValidationError
from ..losses.base import BaseLoss


def make_loss(model, device):
    """损失统一返回逐行张量，样本权重及 TabM 成员归约由训练层处理。"""
    import torch
    from ..losses import get_loss
    from ..losses.torch_adapter import TorchLossAdapter

    if model.loss_params is not None and not isinstance(model.loss_params, dict):
        raise ValidationError("loss_params 必须是字典或 None")
    params = copy.deepcopy(model.loss_params or {})
    if params.pop("reduction", "none") != "none":
        raise ValidationError("loss_params.reduction 必须为 none，训练层统一应用样本权重及均值")
    loss = model.loss
    if loss is None:
        loss = "bce" if model.objective_ == "binary" else "mse"
    builtin = {
        "bce": torch.nn.BCEWithLogitsLoss,
        "mse": torch.nn.MSELoss,
        "mae": torch.nn.L1Loss,
        "huber": torch.nn.HuberLoss,
    }
    if isinstance(loss, str):
        key = loss.lower()
        if key in builtin:
            if (key == "bce") != (model.objective_ == "binary"):
                raise ValidationError("bce 用于二分类；mse、mae、huber 用于回归")
            loss = builtin[key]
        else:
            loss = get_loss(key, **params)
            params = {}
    if isinstance(loss, BaseLoss):
        if model.objective_ != "binary":
            raise ValidationError("BaseLoss 是二分类概率损失，回归请使用 PyTorch 损失或自定义张量函数")
        if params:
            raise ValidationError("BaseLoss 实例请在实例化时配置参数；搜索损失参数可使用名称及 loss_params")
        return TorchLossAdapter(loss).loss_fn()
    if inspect.isclass(loss) and issubclass(loss, torch.nn.Module):
        if "reduction" in inspect.signature(loss).parameters:
            params["reduction"] = "none"
        try:
            loss = loss(**params)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"损失函数参数无效: {exc}") from exc
    elif isinstance(loss, torch.nn.Module):
        if params:
            raise ValidationError("PyTorch 损失实例请在实例化时配置参数，loss_params 只用于损失名称、类或函数")
        loss = copy.deepcopy(loss)
    elif callable(loss):
        return partial(loss, **params) if params else loss
    else:
        raise ValidationError("loss 必须为损失名称、BaseLoss、PyTorch 损失类/实例或可调用张量函数")
    if hasattr(loss, "reduction"):
        loss.reduction = "none"
    return loss.to(device)


def optimizer_config(model):
    """解析统一优化器配置，供本地训练和官方 TabNetPretrainer 共用。"""
    import torch

    if model.optimizer_params is not None and not isinstance(model.optimizer_params, dict):
        raise ValidationError("optimizer_params 必须是字典或 None")
    params = copy.deepcopy(model.optimizer_params or {})
    if {"params", "lr", "weight_decay"}.intersection(params):
        raise ValidationError("优化器参数组由模型管理；学习率和权重衰减请使用 learning_rate、weight_decay")
    optimizers = {name.lower(): getattr(torch.optim, name) for name in ("AdamW", "Adam", "SGD", "RMSprop", "Adagrad")}
    optimizer = optimizers.get(model.optimizer.lower()) if isinstance(model.optimizer, str) else model.optimizer
    if not inspect.isclass(optimizer) or not issubclass(optimizer, torch.optim.Optimizer):
        raise ValidationError("optimizer 必须为 adamw、adam、sgd、rmsprop、adagrad 或 torch.optim.Optimizer 子类")
    if issubclass(optimizer, torch.optim.LBFGS):
        raise ValidationError("当前小批次训练循环不支持需要闭包的 LBFGS 优化器")
    return optimizer, dict(lr=model.learning_rate, weight_decay=model.weight_decay, **params)


def make_optimizer(model, parameters=None):
    """每次 fit 为当前网络新建优化器，拒绝已绑定其他网络的优化器实例。"""
    optimizer, params = optimizer_config(model)
    try:
        return optimizer(
            model._model.parameters() if parameters is None else parameters,
            **params,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"优化器参数无效: {exc}") from exc


def loss_values(model, prediction, target):
    """TabM 将成员轴展平独立求损失，再恢复成员轴并求均值。"""
    import torch

    labels = target[:, None].expand_as(prediction) if prediction.ndim == 2 else target
    values = model.loss_fn_(prediction.reshape(-1), labels.reshape(-1))
    if not isinstance(values, torch.Tensor) or values.shape != (prediction.numel(),):
        raise ValidationError("自定义损失必须返回与预测等长的一维逐样本张量；请使用 reduction='none'")
    if not torch.isfinite(values).all():
        raise ValidationError("损失函数返回非有限值，请检查输入与损失参数")
    if prediction.requires_grad and not values.requires_grad:
        raise ValidationError("自定义损失必须保留自动微分计算图，不能返回脱离计算图的张量")
    values = values.reshape(prediction.shape)
    return values.mean(dim=1) if values.ndim == 2 else values
