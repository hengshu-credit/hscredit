"""net 模型的默认搜索空间，保留调用方明确固定的模型及优化器配置。"""

import math
from numbers import Integral


def net_search_space(model_class, params):
    name = getattr(model_class, "_network_kind", model_class.__name__)
    if name == "TabNet":
        return {
            "n_d": {"type": "categorical", "choices": [8, 16, 32, 64]},
            "n_a": {"type": "categorical", "choices": [8, 16, 32, 64]},
            "n_steps": {"type": "int", "low": 3, "high": 6},
            "gamma": {"type": "float", "low": 1.0, "high": 2.0},
            "lambda_sparse": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 3e-2, "log": True},
            "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
        }
    if name == "TabPFN":
        if params.get("finetune"):
            return {
                "finetune_params__learning_rate": {"type": "float", "low": 1e-6, "high": 3e-5, "log": True},
                "finetune_params__weight_decay": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
            }
        return {
            "n_estimators": {"type": "categorical", "choices": [2, 4, 8]},
            "softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        }
    widths = [64, 128, 256, 512] if name == "TabM" else [64, 128, 192, 256]
    if name == "FTTransformer":
        heads = params.get("n_heads", 8)
        if not isinstance(heads, Integral) or isinstance(heads, bool) or heads < 1:
            raise ValueError("FTTransformer 的 n_heads 必须为正整数")
        widths = sorted({math.ceil(width / heads) * heads for width in widths})
    # k、arch_type、loss、optimizer 默认保持固定；可用显式空间或条件搜索单独探索。
    return {
        "d_block": {"type": "categorical", "choices": widths},
        "n_blocks": {"type": "int", "low": 2 if name == "TabM" else 1, "high": 4},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    }
