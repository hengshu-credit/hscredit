"""环境初始化.

提供 hscredit 全局环境配置函数，包括警告屏蔽、pandas 显示、
系统字体安装、matplotlib 字体、随机种子等一站式设置。
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .fonts import FONT_NAME, get_bundled_font_path, initialize_bundled_font


def init_setting(font_path=None, seed=None, freeze_torch=False, logger=False, **kwargs):
    """初始化环境配置。

    去除警告信息、修改 pandas 默认配置、固定随机种子。

    :param font_path: 画图时图像使用的字体，支持系统已注册字体名称或本地 ``.ttf``
        字体文件路径；为 None 时自动安装并使用包内置字体，安装不可用时回退到“楷体”
    :param seed: 随机种子，默认为 None（不固定）。非 None 时调用
        :func:`~hscredit.utils.seed_everything`
    :param freeze_torch: 是否同时固定 PyTorch 随机种子，默认 False（仅 seed 非 None 时生效）
    :param logger: 是否返回一个日志器，默认为 False
    :param kwargs: 当 logger 为 True 时传给 ``logging.getLogger`` 的参数
    :return: 当 logger 为 True 时返回 ``logging.Logger``，否则返回 None

    **注意**

    本函数在 ``import hscredit`` 时被自动调用，会尝试将内置字体安装到当前用户字体目录，
    全局执行 ``warnings.filterwarnings("ignore")`` 屏蔽所有警告，并修改 pandas/matplotlib 全局配置。
    字体安装失败不会阻断导入，系统不存在品牌字体时将回退到“楷体”。

    **参考样例**

    >>> from hscredit.utils import init_setting
    >>> init_setting()                       # 默认配置（内置中文字体）
    >>> init_setting(seed=42)                # 同时固定随机种子
    >>> init_setting(font_path='SimHei')     # 指定系统字体
    >>> logger = init_setting(logger=True)   # 返回日志器
    """
    warnings.filterwarnings("ignore")

    default_font_name = initialize_bundled_font()

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option("display.max_colwidth", 300)
    pd.set_option('expand_frame_repr', False)

    if "seaborn-ticks" in plt.style.available:
        plt.style.use('seaborn-ticks')
    else:
        plt.style.use('seaborn-v0_8-ticks')

    resolved_font_name = default_font_name
    resolved_font_path = None
    if font_path is None:
        if default_font_name == FONT_NAME:
            resolved_font_path = os.fspath(get_bundled_font_path())
    else:
        candidate = os.fspath(font_path)
        if os.path.isfile(candidate):
            resolved_font_path = candidate
        else:
            resolved_font_name = candidate

    if resolved_font_path is not None and os.path.isfile(resolved_font_path):
        try:
            font_manager.fontManager.addfont(resolved_font_path)
            resolved_font_name = font_manager.FontProperties(fname=resolved_font_path).get_name()
            # 使用粗体字
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'
        except Exception:
            if font_path is not None:
                raise

    plt.rcParams['font.family'] = resolved_font_name

    plt.rcParams['axes.unicode_minus'] = False

    if seed:
        from .random import seed_everything
        seed_everything(seed, freeze_torch=freeze_torch)

    if logger:
        import logging
        return logging.getLogger(**kwargs)
