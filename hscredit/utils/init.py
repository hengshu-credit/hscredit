"""环境初始化.

提供 hscredit 全局环境配置函数，包括警告屏蔽、pandas 显示、
matplotlib 字体、随机种子等一站式设置。
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def init_setting(font_path=None, seed=None, freeze_torch=False, logger=False, **kwargs):
    """初始化环境配置。

    去除警告信息、修改 pandas 默认配置、固定随机种子。

    :param font_path: 画图时图像使用的字体，支持系统已注册字体名称或本地 ``.ttf``
        字体文件路径；为 None 时使用包内置中文字体 ``resources/fonts/font.ttf``
    :param seed: 随机种子，默认为 None（不固定）。非 None 时调用
        :func:`~hscredit.utils.seed_everything`
    :param freeze_torch: 是否同时固定 PyTorch 随机种子，默认 False（仅 seed 非 None 时生效）
    :param logger: 是否返回一个日志器，默认为 False
    :param kwargs: 当 logger 为 True 时传给 ``logging.getLogger`` 的参数
    :return: 当 logger 为 True 时返回 ``logging.Logger``，否则返回 None

    **注意**

    本函数在 ``import hscredit`` 时被自动调用，会全局执行
    ``warnings.filterwarnings("ignore")`` 屏蔽所有警告，并修改 pandas/matplotlib 全局配置。

    **参考样例**

    >>> from hscredit.utils import init_setting
    >>> init_setting()                       # 默认配置（内置中文字体）
    >>> init_setting(seed=42)                # 同时固定随机种子
    >>> init_setting(font_path='SimHei')     # 指定系统字体
    >>> logger = init_setting(logger=True)   # 返回日志器
    """
    warnings.filterwarnings("ignore")

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option("display.max_colwidth", 300)
    pd.set_option('expand_frame_repr', False)

    if "seaborn-ticks" in plt.style.available:
        plt.style.use('seaborn-ticks')
    else:
        plt.style.use('seaborn-v0_8-ticks')

    if font_path is not None and font_path.lower() in [font.fname.lower() for font in font_manager.fontManager.ttflist]:
        plt.rcParams['font.family'] = font_path
    else:
        # 使用resources目录下的字体文件
        if font_path is None:
            font_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'resources', 'fonts', 'font.ttf'
            )

        if os.path.isfile(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = font_name
            # 使用粗体字
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'

    plt.rcParams['axes.unicode_minus'] = False

    if seed:
        from .random import seed_everything
        seed_everything(seed, freeze_torch=freeze_torch)

    if logger:
        import logging
        return logging.getLogger(**kwargs)
