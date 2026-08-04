"""并行任务参数工具。"""

import math
import numbers
import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

import numpy as np
from joblib import cpu_count as joblib_cpu_count

from ..exceptions import ValidationError


NJobs = Optional[Union[int, float]]

_PARALLEL_CONFIG_KEYS = {
    "batch_size",
    "pre_dispatch",
    "max_nbytes",
    "mmap_mode",
    "temp_folder",
    "prefer",
    "require",
    "verbose",
    "timeout",
    "backend_kwargs",
}


def get_physical_cpu_count() -> int:
    """返回可用的物理 CPU 数，无法识别时使用保守回退值。"""
    try:
        count = joblib_cpu_count(only_physical_cores=True)
    except TypeError:
        count = None
    except Exception:
        count = None
    if count:
        return max(1, int(count))

    try:
        count = joblib_cpu_count()
    except Exception:
        count = None
    if count:
        return max(1, int(count))

    return max(1, int(os.cpu_count() or 1))


def resolve_n_jobs(
    n_jobs: NJobs,
    task_count: Optional[int] = None,
    *,
    cpu_count: Optional[int] = None,
    available_budget: Optional[int] = None,
) -> Optional[int]:
    """解析并行工作数。

    ``-1`` 大约使用物理 CPU 的 80%，并在多核环境中保留一个 CPU。
    正整数表示固定工作数，``0`` 到 ``1`` 之间的小数表示物理 CPU 比例。
    """
    if available_budget is not None:
        if isinstance(available_budget, (bool, np.bool_)) or not isinstance(available_budget, numbers.Integral):
            raise ValidationError("available_budget 必须为正整数")
        if available_budget < 1:
            raise ValidationError("available_budget 必须为正整数")
        available_budget = int(available_budget)

    if n_jobs is None:
        return None
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, numbers.Real):
        raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")

    cpus = max(1, int(cpu_count or get_physical_cpu_count()))

    if isinstance(n_jobs, numbers.Integral):
        value = int(n_jobs)
        if value == -1:
            workers = 1 if cpus == 1 else min(cpus - 1, math.ceil(cpus * 0.8))
            if available_budget is not None:
                workers = min(workers, available_budget)
        elif value >= 1:
            workers = value
        else:
            raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")
    else:
        value = float(n_jobs)
        if not math.isfinite(value):
            raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")
        if value == -1:
            workers = 1 if cpus == 1 else min(cpus - 1, math.ceil(cpus * 0.8))
            if available_budget is not None:
                workers = min(workers, available_budget)
        elif value.is_integer() and value >= 1:
            workers = int(value)
        elif 0 < value < 1:
            workers = math.ceil(cpus * value)
        else:
            raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")

    if task_count is not None:
        workers = min(workers, max(1, int(task_count)))
    return max(1, workers)


def validate_parallel_config(
    parallel_backend: Optional[str], parallel_config: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """校验 joblib 并行配置并返回独立的配置字典。

    工作数和后端由公共参数统一管理，不能在 ``parallel_config`` 中重复声明。
    ``backend_kwargs`` 用于承载后端专属参数，且会复制为独立字典。
    """
    if parallel_config is None:
        return {}
    if not isinstance(parallel_config, Mapping):
        raise ValidationError("parallel_config 必须为字典")

    config = dict(parallel_config)
    if any(not isinstance(key, str) for key in config):
        raise ValidationError("parallel_config 的配置项名称必须为字符串")
    if "n_jobs" in config:
        raise ValidationError("parallel_config 不能包含 n_jobs，请使用 n_jobs 参数")
    if "backend" in config:
        raise ValidationError("parallel_config 不能包含 backend，请使用 parallel_backend 参数")

    unknown_keys = set(config).difference(_PARALLEL_CONFIG_KEYS)
    if unknown_keys:
        raise ValidationError(f"parallel_config 包含不支持的配置项: {sorted(unknown_keys)}")

    backend_kwargs = config.get("backend_kwargs")
    if backend_kwargs is not None:
        if not isinstance(backend_kwargs, Mapping):
            raise ValidationError("parallel_config.backend_kwargs 必须为字典")
        config["backend_kwargs"] = dict(backend_kwargs)

    return config
