"""论文作者 rtdl 网络的表示层与任务输出头（仅在训练时加载 PyTorch）。"""

import torch
from torch import nn
import rtdl_revisiting_models as rtdl


class TabularNetwork(nn.Module):
    """在原生 ResNet / FTTransformer 表示之后保留独立预测头。"""

    def __init__(self, kind, n_numeric, cardinalities, d_block, n_blocks, dropout, cat_embedding_dim=16, n_heads=8):
        super().__init__()
        self.kind = kind
        if kind == "ResNet":
            self.category_embeddings = nn.ModuleList([nn.Embedding(size, cat_embedding_dim) for size in cardinalities])
            self.backbone = rtdl.ResNet(
                d_in=n_numeric + len(cardinalities) * cat_embedding_dim,
                d_out=None,
                n_blocks=n_blocks,
                d_block=d_block,
                d_hidden_multiplier=2.0,
                dropout1=dropout,
                dropout2=dropout,
            )
            self.normalization = nn.BatchNorm1d(d_block)
        else:
            self.backbone = rtdl.FTTransformer(
                n_cont_features=n_numeric,
                cat_cardinalities=cardinalities,
                d_out=None,
                n_blocks=n_blocks,
                d_block=d_block,
                attention_n_heads=n_heads,
                attention_dropout=dropout,
                ffn_d_hidden_multiplier=4 / 3,
                ffn_dropout=dropout,
                residual_dropout=0.0,
            )
            self.normalization = nn.LayerNorm(d_block)
        self.head = nn.Linear(d_block, 1)

    def embeddings(self, numeric, categorical):
        if self.kind == "ResNet":
            columns = [numeric] + [module(categorical[:, i]) for i, module in enumerate(self.category_embeddings)]
            hidden = self.backbone(torch.cat(columns, dim=1))
        else:
            hidden = self.backbone(numeric if numeric.shape[1] else None, categorical if categorical.shape[1] else None)
        return torch.relu(self.normalization(hidden))

    def forward(self, numeric, categorical):
        return self.head(self.embeddings(numeric, categorical)).squeeze(-1)
