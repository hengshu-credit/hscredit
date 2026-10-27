"""逐样本二分类损失到 PyTorch 自动微分的适配，导入时不加载 torch。"""

import copy

import numpy as np
from scipy.special import expit

from ....exceptions import DependencyError, ValidationError
from .base import BaseLoss


class TorchLossAdapter:
    """将可分解的 BaseLoss 转为接收 logits 的 PyTorch 损失。

    **参数**

    :param loss: 二分类 BaseLoss，必须实现逐行 loss_values 并声明 is_additive=True。

    **属性**

    :ivar loss: 原始损失配置；创建损失模块时复制，不修改调用方对象。

    **参考样例**

    >>> from hscredit import FocalLoss
    >>> criterion = TorchLossAdapter(FocalLoss()).loss_fn(reduction='none')
    """

    def __init__(self, loss):
        if not isinstance(loss, BaseLoss):
            raise ValidationError("TorchLossAdapter 的 loss 必须是 BaseLoss 实例")
        self.loss = loss

    def loss_fn(self, reduction="none"):
        """返回可反向传播的损失模块。

        :param reduction: none 保留逐样本损失；mean/sum 分别求均值/和。
        :return: 接收 (一维原始 logits, 同形状 0/1 标签) 的 nn.Module。

        >>> criterion = TorchLossAdapter(FocalLoss()).loss_fn()
        """
        if reduction not in ("none", "mean", "sum"):
            raise ValidationError("reduction 必须为 none、mean 或 sum")
        if not self.loss.is_additive or type(self.loss).loss_values is BaseLoss.loss_values:
            raise ValidationError(
                "小批次网络训练只支持可逐样本分解的 BaseLoss；全量排序、动态权重或绑定样本数组的损失不适用"
            )
        try:
            import torch
        except ImportError as exc:
            raise DependencyError('PyTorch 损失需要 torch，请安装 pip install "hscredit[net]"') from exc
        loss = copy.deepcopy(self.loss)

        class LossFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, logits, target):
                if logits.ndim != 1 or logits.shape != target.shape:
                    raise ValidationError("PyTorch 二分类损失要求 logits 与标签是等长一维张量")
                probability = expit(logits.detach().cpu().double().numpy())
                labels = target.detach().cpu().numpy()
                values = np.asarray(loss.loss_values(labels, probability), dtype=float)
                gradient = np.asarray(loss.gradient(labels, probability), dtype=float) * probability * (1 - probability)
                if values.shape != probability.shape or gradient.shape != probability.shape:
                    raise ValidationError("BaseLoss 的逐样本损失及梯度必须与输入形状一致")
                if not np.isfinite(values).all() or not np.isfinite(gradient).all():
                    raise ValidationError("BaseLoss 返回了非有限损失或梯度")
                ctx.save_for_backward(torch.as_tensor(gradient, dtype=logits.dtype, device=logits.device))
                return torch.as_tensor(values, dtype=logits.dtype, device=logits.device)

            @staticmethod
            def backward(ctx, output_gradient):
                (gradient,) = ctx.saved_tensors
                return output_gradient * gradient, None

        class LossModule(torch.nn.Module):
            def forward(self, logits, target):
                values = LossFunction.apply(logits, target)
                return values.mean() if reduction == "mean" else values.sum() if reduction == "sum" else values

        return LossModule()
