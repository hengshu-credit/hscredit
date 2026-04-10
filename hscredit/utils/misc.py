"""杂项工具.

提供各种辅助函数。
"""

import os
import re
import random
import numpy as np
import pandas as pd


def round_float(num, decimal: int = 4):
    """调整数值分箱的上下界小数点精度，如未超出精度保持原样输出。

    :param num: 分箱的上界或者下界
    :param decimal: 小数点保留的精度
    :return: 精度调整后的数值

    示例:
        >>> round_float(3.14159265, decimal=4)
        3.1416
    """
    if decimal is None:
        return num

    if isinstance(decimal, (bool, np.bool_)) or not isinstance(decimal, (int, np.integer)) or int(decimal) < 0:
        raise ValueError("decimal 必须是大于等于 0 的整数")

    if pd.isna(num) or isinstance(num, (bool, np.bool_)):
        return num

    if isinstance(num, (int, np.integer)):
        return int(num)

    if isinstance(num, (float, np.floating)):
        return round(float(num), int(decimal))

    return num
