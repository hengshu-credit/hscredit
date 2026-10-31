"""官方 TabNet 编码器、决策表示和原始字段注意力的薄适配。"""

import math

import torch
from torch import nn
from pytorch_tabnet.tab_network import GBN, TabNet


class _SafeGBN(GBN):
    """合并单行的虚拟批次，保留全部样本及官方其余归一化行为。"""

    def forward(self, x):
        chunks = list(x.chunk(math.ceil(x.shape[0] / self.virtual_batch_size), 0))
        if self.training and len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2] = torch.cat((chunks[-2], chunks[-1]), dim=0)
            chunks.pop()
        return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)


def _safe_batches(module):
    for name, child in list(module.named_children()):
        if isinstance(child, GBN):
            replacement = _SafeGBN(child.input_dim, child.virtual_batch_size, child.bn.momentum)
            replacement.bn = child.bn
            setattr(module, name, replacement)
        else:
            _safe_batches(child)


class TabNetNetwork(nn.Module):
    """各决策步的表示求和，单个 logit/标准化回归头与其他 net 模型一致。"""

    def __init__(self, n_numeric, cardinalities, cat_embedding_dim, **params):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_embedding_dim = cat_embedding_dim
        self.n_categorical = len(cardinalities)
        width = n_numeric + len(cardinalities)
        self.backbone = TabNet(
            input_dim=width,
            output_dim=1,
            cat_idxs=list(range(n_numeric, width)),
            cat_dims=cardinalities,
            cat_emb_dim=[cat_embedding_dim] * len(cardinalities),
            group_attention_matrix=torch.eye(width),
            **params,
        )
        # 官方矩阵为普通属性；注册为 buffer，确保 CPU/CUDA 迁移及保存恢复一致。
        for module, name in (
            (self.backbone.embedder, "embedding_group_matrix"),
            (self.backbone.tabnet.encoder, "group_attention_matrix"),
        ):
            value = getattr(module, name)
            delattr(module, name)
            module.register_buffer(name, value)
        _safe_batches(self.backbone)

    @property
    def head(self):
        return self.backbone.tabnet.final_mapping

    @staticmethod
    def _input(numeric, categorical):
        return torch.cat((numeric, categorical.to(numeric.dtype)), dim=1)

    def _encode(self, numeric, categorical):
        embedded = self.backbone.embedder(self._input(numeric, categorical))
        steps, mask_loss = self.backbone.tabnet.encoder(embedded)
        return torch.stack(steps).sum(dim=0), mask_loss

    def embeddings(self, numeric, categorical):
        return self._encode(numeric, categorical)[0]

    def forward(self, numeric, categorical):
        hidden, mask_loss = self._encode(numeric, categorical)
        if self.training:
            self.mask_loss = mask_loss
        return self.head(hidden).squeeze(-1)

    def feature_attention(self, numeric, categorical):
        """将类别 embedding 的注意力求和，还原为数值列后接类别列。"""
        values, _ = self.backbone.forward_masks(self._input(numeric, categorical))
        parts = [values[:, : self.n_numeric]]
        if self.n_categorical:
            parts.append(values[:, self.n_numeric :].reshape(len(values), self.n_categorical, -1).sum(dim=2))
        return torch.cat(parts, dim=1)
