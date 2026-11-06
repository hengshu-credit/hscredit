"""工具函数模块.

提供常用的工具函数，包括随机种子设置、数据IO、特征描述、分箱表美化展示、环境初始化、日志管理等。
"""

import os as _os

if _os.environ.get("HSCREDIT_LIGHT_IMPORT") == "1":
    from importlib import import_module as _import_module
    from .._public_api import EXPORTS

    _exports = EXPORTS[__name__]
    __all__ = list(_exports)

    def __getattr__(name):
        if name not in _exports:
            raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
        value = getattr(_import_module(_exports[name]), name)
        globals()[name] = value
        return value

    def __dir__():
        return sorted(set(globals()) | set(__all__))

else:
    from .random import seed_everything
    from .environment import check_environment
    from .io import load_pickle, save_pickle
    from .serialization import (
        ArtifactSerializableMixin,
        ARTIFACT_FORMAT,
        ARTIFACT_VERSION,
    )
    from .parallel import (
        ParallelBudget,
        ParallelExecutionPlan,
        ParallelWorkload,
        ParallelizableMixin,
        get_physical_cpu_count,
        parallel_execute,
        plan_parallel_execution,
        resolve_n_jobs,
        resolve_native_workers,
        split_parallel_budget,
        validate_parallel_config,
    )
    from .describe import feature_describe, groupby_feature_describe
    from .datasets import germancredit, hscredit_dataset
    from .misc import reload, round_float, trapz
    from .init import init_setting
    from .fonts import FONT_NAME, get_bundled_font_path, install_bundled_font
    from .logger import init_logger, get_logger
    from .pandas_extensions import (
        style_bin_table,
        style_rule_table,
        BinTableDisplay,
        register_extensions,
    )
    from .pandas_parallel import HSCreditApplyProxy, create_hscredit_apply_proxy
    from .input_utils import (
        check_xy_inputs,
        convert_to_dataframe,
        extract_target_from_df,
        check_array_1d,
        get_feature_dtypes,
        check_missing_values,
        normalize_dpd_values,
    )

    __all__ = [
        "check_environment",
        # 随机种子
        "seed_everything",
        # 数据IO
        "load_pickle",
        "save_pickle",
        "ArtifactSerializableMixin",
        "ARTIFACT_FORMAT",
        "ARTIFACT_VERSION",
        "ParallelBudget",
        "ParallelExecutionPlan",
        "ParallelWorkload",
        "ParallelizableMixin",
        "parallel_execute",
        "plan_parallel_execution",
        "resolve_n_jobs",
        "resolve_native_workers",
        "get_physical_cpu_count",
        "split_parallel_budget",
        "validate_parallel_config",
        # 特征描述
        "feature_describe",
        "groupby_feature_describe",
        # 数据集
        "germancredit",
        "hscredit_dataset",
        # 杂项工具
        "round_float",
        "reload",
        "trapz",
        "init_setting",
        "FONT_NAME",
        "get_bundled_font_path",
        "install_bundled_font",
        # 日志工具
        "init_logger",
        "get_logger",
        # 分箱表展示
        "style_bin_table",
        "style_rule_table",
        "BinTableDisplay",
        "register_extensions",
        "HSCreditApplyProxy",
        "create_hscredit_apply_proxy",
        # 输入处理工具
        "check_xy_inputs",
        "convert_to_dataframe",
        "extract_target_from_df",
        "check_array_1d",
        "get_feature_dtypes",
        "check_missing_values",
        "normalize_dpd_values",
    ]
