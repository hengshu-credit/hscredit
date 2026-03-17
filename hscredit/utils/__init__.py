"""工具函数模块.

提供常用的工具函数，包括随机种子设置、数据IO、特征描述、分箱表美化展示、环境初始化、日志管理等。
"""

from .random import seed_everything
from .io import load_pickle, save_pickle
from .describe import feature_describe, groupby_feature_describe
from .datasets import germancredit
from .misc import init_setting, round_float
from .logger import init_logger, get_logger
from .bin_table_display import (
    style_bin_table,
    BinTableDisplay,
    enable_dataframe_show,
)

__all__ = [
    # 随机种子
    'seed_everything',
    # 数据IO
    'load_pickle',
    'save_pickle',
    # 特征描述
    'feature_describe',
    'groupby_feature_describe',
    # 数据集
    'germancredit',
    # 杂项工具
    'round_float',
    'init_setting',
    # 日志工具
    'init_logger',
    'get_logger',
    # 分箱表展示
    'style_bin_table',
    'BinTableDisplay',
    'enable_dataframe_show',
]
