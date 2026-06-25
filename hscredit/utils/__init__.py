"""工具函数模块.

提供常用的工具函数，包括随机种子设置、数据IO、特征描述、分箱表美化展示、环境初始化、日志管理等。
"""

from .random import seed_everything
from .io import load_pickle, save_pickle
from .serialization import (
    ArtifactSerializableMixin,
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
)
from .parallel import resolve_n_jobs
from .describe import feature_describe, groupby_feature_describe
from .datasets import germancredit
from .misc import round_float, force_reload_module, trapz
from .init import init_setting
from .logger import init_logger, get_logger
from .pandas_extensions import (
    style_bin_table,
    style_rule_table,
    BinTableDisplay,
    register_extensions,
)
from .input_utils import (
    check_xy_inputs,
    convert_to_dataframe,
    extract_target_from_df,
    check_array_1d,
    get_feature_dtypes,
    check_missing_values,
)

__all__ = [
    # 随机种子
    'seed_everything',
    # 数据IO
    'load_pickle',
    'save_pickle',
    'ArtifactSerializableMixin',
    'ARTIFACT_FORMAT',
    'ARTIFACT_VERSION',
    'resolve_n_jobs',
    # 特征描述
    'feature_describe',
    'groupby_feature_describe',
    # 数据集
    'germancredit',
    # 杂项工具
    'round_float',
    'force_reload_module',
    'trapz',
    'init_setting',
    # 日志工具
    'init_logger',
    'get_logger',
    # 分箱表展示
    'style_bin_table',
    'style_rule_table',
    'BinTableDisplay',
    'register_extensions',
    # 输入处理工具
    'check_xy_inputs',
    'convert_to_dataframe',
    'extract_target_from_df',
    'check_array_1d',
    'get_feature_dtypes',
    'check_missing_values',
]
