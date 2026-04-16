"""随机工具.

提供随机种子设置等功能。
"""

import os
import random
import numpy as np


def seed_everything(seed: int, freeze_torch: bool = False):
    """固定当前环境随机种子，以保证后续实验可重复。

    :param seed: 随机种子
    :param freeze_torch: 是否固定 pytorch 的随机种子

    **参考样例**

    >>> seed_everything(42)
    >>> seed_everything(42, freeze_torch=True)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if freeze_torch:
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        except ImportError:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")
