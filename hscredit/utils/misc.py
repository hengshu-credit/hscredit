"""杂项工具.

提供各种辅助函数。
"""

import os
import re
import sys
import types
import random
import numpy as np
import pandas as pd
import importlib.util


def round_float(num, decimal: int = 4):
    """调整数值分箱的上下界小数点精度，如未超出精度保持原样输出。

    :param num: 分箱的上界或者下界
    :param decimal: 小数点保留的精度
    :return: 精度调整后的数值

    **参考样例**

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


def force_reload_module(module_name):
    """在jupyter中强制重载模块，忽略所有缓存
    
    :param module_name: 模块名称
    :return: 重新导入的模块

    **参考样例**

    >>> import hscredit.utils.misc
    >>> hscredit.utils.misc.force_reload_module('hscredit.utils.misc')
    <module 'hscredit.utils.misc' from '...'>
    """
    if module_name in sys.modules:
        # 获取旧模块
        old_module = sys.modules[module_name]
        
        # 删除模块的所有子模块
        submodules = [name for name in sys.modules if name.startswith(f"{module_name}.")]
        for submodule in submodules:
            del sys.modules[submodule]
        
        # 删除主模块
        del sys.modules[module_name]
        
        # 清除可能存在的 .pyc 缓存
        import py_compile
        try:
            if hasattr(old_module, '__file__') and old_module.__file__:
                pyc_file = old_module.__file__ + 'c'
                import os
                if os.path.exists(pyc_file):
                    os.remove(pyc_file)
        except:
            pass
    
    # 重新导入
    return importlib.import_module(module_name)
