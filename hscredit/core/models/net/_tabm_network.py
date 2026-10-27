"""官方 TabM 网络的成员表示及预测头，保持官方前向计算顺序。"""

from torch import nn
from tabm import TabM as NativeTabM
import rtdl_num_embeddings

from ....exceptions import ValidationError


class TabMNetwork(nn.Module):
    """输出 (样本数, 成员数)，表示为 (样本数, 成员数, d_block)。"""

    def __init__(
        self, n_numeric, cardinalities, d_block, n_blocks, dropout, k, arch_type, num_embeddings, num_embedding_params
    ):
        super().__init__()
        embeddings = None
        if num_embeddings is not None and n_numeric:
            constructors = {
                "linear_relu": rtdl_num_embeddings.LinearReLUEmbeddings,
                "periodic": rtdl_num_embeddings.PeriodicEmbeddings,
            }
            options = {"lite": True} if num_embeddings == "periodic" else {}
            options.update(num_embedding_params or {})
            if "n_features" in options:
                raise ValidationError("数值表示的 n_features 由输入数据决定，不能手工覆盖")
            try:
                embeddings = constructors[num_embeddings](n_numeric, **options)
            except (ValueError, TypeError) as exc:
                raise ValidationError(f"TabM 数值表示参数无效: {exc}") from exc
        self.backbone = NativeTabM.make(
            n_num_features=n_numeric,
            cat_cardinalities=cardinalities,
            d_out=1,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=k,
            arch_type=arch_type,
            num_embeddings=embeddings,
        )
        self.head = self.backbone.output
        self.backbone.output = None

    def embeddings(self, numeric, categorical):
        return self.backbone(numeric if numeric.shape[1] else None, categorical if categorical.shape[1] else None)

    def forward(self, numeric, categorical):
        return self.head(self.embeddings(numeric, categorical)).squeeze(-1)
