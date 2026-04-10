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

    :param font_path: 画图时图像使用的字体，支持系统字体名称、本地字体文件路径
    :param seed: 随机种子，默认为 None
    :param freeze_torch: 是否固定 pytorch 环境
    :param logger: 是否需要初始化日志器，默认为 False
    :param kwargs: 日志初始化传入的相关参数
    :return: 当 logger 为 True 时返回 logging.Logger
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
