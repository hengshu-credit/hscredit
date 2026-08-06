"""并行任务参数工具。"""

import math
import numbers
import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from joblib import Parallel, delayed, parallel_backend as joblib_parallel_backend
from joblib import cpu_count as joblib_cpu_count

from ..exceptions import ParallelExecutionError, ValidationError


NJobs = Optional[Union[int, float]]
Task = TypeVar("Task")
Result = TypeVar("Result")

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
    "inner_max_num_threads",
    "backend_kwargs",
}

# 各 joblib 内置后端支持的 backend_kwargs 白名单。
# joblib 的 LokyBackend.configure 等实现会通过 **kwargs 静默吞掉未知参数，
# 需要在校验阶段拦截，保证配置错误以中文公共异常暴露。
_BACKEND_OPTION_KEYS = {
    "loky": {
        "inner_max_num_threads",
        "prefer",
        "require",
        "idle_worker_timeout",
        "temp_folder",
        "temp_folder_root",
        "max_nbytes",
        "mmap_mode",
        "context",
        "timeout",
    },
    "multiprocessing": {
        "prefer",
        "require",
        "temp_folder",
        "temp_folder_root",
        "max_nbytes",
        "mmap_mode",
        "context",
    },
    "threading": {"prefer", "require"},
    "sequential": {"prefer", "require"},
}


def _validate_backend_options(backend: str, backend_options: Mapping[str, Any]) -> None:
    """校验后端专属参数，未知参数统一转换为中文公共校验异常。"""
    if not backend_options:
        return
    allowed = _BACKEND_OPTION_KEYS.get(backend)
    if allowed is None:
        # 外部注册的自定义后端不做白名单校验
        return
    unknown = sorted(set(backend_options).difference(allowed))
    if unknown:
        raise ValidationError(f"并行后端配置无效: 后端 '{backend}' 不支持的参数 {unknown}")


def _validate_integer(value: Any, name: str, minimum: int) -> int:
    """校验并规范化公开并行预算中的整数参数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        requirement = "正整数" if minimum == 1 else "非负整数"
        raise ValidationError(f"{name} 必须为{requirement}")
    value = int(value)
    if value < minimum:
        requirement = "正整数" if minimum == 1 else "非负整数"
        raise ValidationError(f"{name} 必须为{requirement}")
    return value


@dataclass(frozen=True)
class ParallelBudget:
    """并行执行上下文中的可用预算。

    **参数**
        available: 当前调用可使用的最大并行预算。
        depth: 当前调用所在的嵌套深度。

    **属性**
        available: 正整数并行预算。
        depth: 非负嵌套深度。

    **参考样例**
        ``ParallelBudget(available=4, depth=1)`` 表示第一层 worker 可使用 4 个工作预算。
    """

    available: int
    depth: int

    def __post_init__(self) -> None:
        """校验并规范化预算边界。"""
        object.__setattr__(
            self, "available", _validate_integer(self.available, "available", 1)
        )
        object.__setattr__(self, "depth", _validate_integer(self.depth, "depth", 0))


_ACTIVE_BUDGET: ContextVar[Optional[ParallelBudget]] = ContextVar(
    "hscredit_parallel_budget", default=None
)


class _WorkerExecutionError(Exception):
    """在线程和进程 worker 间传递原始异常的内部包装。"""

    def __init__(self, label: Any, original_exception: Exception) -> None:
        super().__init__(label, original_exception)
        self.label = label
        self.original_exception = original_exception


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


def split_parallel_budget(
    available: int, task_count: int, has_parallel_children: bool
) -> Tuple[int, int]:
    """计算当前层和真实并行子层的工作预算。"""
    available = _validate_integer(available, "available", 1)
    task_count = _validate_integer(task_count, "task_count", 0)
    if not isinstance(has_parallel_children, (bool, np.bool_)):
        raise ValidationError("has_parallel_children 必须为布尔值")
    if not has_parallel_children:
        return min(available, max(1, task_count)), 1
    outer = min(max(1, task_count), math.ceil(math.sqrt(available)))
    return outer, max(1, available // outer)


def _run_with_budget(
    function: Callable[[Task], Result],
    task: Task,
    label: Any,
    child_budget: int,
    depth: int,
) -> Result:
    """在显式预算上下文中执行单个可序列化任务。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(child_budget, depth))
    try:
        return function(task)
    except Exception as exc:
        raise _WorkerExecutionError(label, exc) from exc
    finally:
        _ACTIVE_BUDGET.reset(token)


def _raise_parallel_execution_error(error: _WorkerExecutionError) -> None:
    """在父调用中恢复统一的中文错误和原始直接异常链。"""
    raise ParallelExecutionError(
        f"并行任务 '{error.label}' 执行失败: {error.original_exception}"
    ) from error.original_exception


def _current_parallel_budget() -> ParallelBudget:
    """返回当前预算；根调用使用统一的自动并行预算。"""
    active_budget = _ACTIVE_BUDGET.get()
    if active_budget is not None:
        return active_budget
    return ParallelBudget(resolve_n_jobs(-1) or 1, 0)


def _create_joblib_parallel(workers: int, options: Mapping[str, Any]) -> Parallel:
    """创建 joblib 执行器，并将公开配置错误转换为中文异常。"""
    try:
        return Parallel(n_jobs=workers, **dict(options))
    except Exception as exc:
        raise ValidationError(f"并行后端配置无效: {exc}") from exc


def parallel_execute(
    function: Callable[[Task], Result],
    tasks: Iterable[Task],
    *,
    n_jobs: NJobs = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Mapping[str, Any]] = None,
    task_labels: Optional[Iterable[Any]] = None,
    default_backend: Optional[str] = None,
    has_parallel_children: bool = False,
) -> List[Result]:
    """按提交顺序执行任务，并在线程和进程间传播并行预算。"""
    task_list = list(tasks)
    if task_labels is None:
        labels: Sequence[Any] = list(range(len(task_list)))
    else:
        labels = list(task_labels)
        if len(labels) != len(task_list):
            raise ValidationError("task_labels 的数量必须与 tasks 一致")

    config = validate_parallel_config(parallel_backend, parallel_config)
    current_budget = _current_parallel_budget()
    requested_workers = resolve_n_jobs(
        n_jobs,
        task_count=len(task_list),
        available_budget=current_budget.available,
    )
    if not task_list:
        return []

    workers = requested_workers or 1
    if has_parallel_children:
        automatic_limit, _ = split_parallel_budget(
            current_budget.available, len(task_list), True
        )
        if n_jobs == -1:
            workers = min(workers, automatic_limit)
        child_budget = max(1, current_budget.available // workers)
    else:
        child_budget = current_budget.available
    depth = current_budget.depth + 1

    calls = (
        (function, task, label, child_budget, depth)
        for task, label in zip(task_list, labels)
    )
    if workers == 1:
        try:
            return [_run_with_budget(*call) for call in calls]
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc)

    parallel_options = dict(config)
    backend_options = parallel_options.pop("backend_kwargs", {}) or {}
    inner_max_num_threads = parallel_options.pop("inner_max_num_threads", None)

    backend = parallel_backend or default_backend
    if inner_max_num_threads is not None:
        if backend == "threading":
            raise ValidationError("threading 后端不支持 inner_max_num_threads")
        backend_options["inner_max_num_threads"] = inner_max_num_threads

    submitted = (
        delayed(_run_with_budget)(*call)
        for call in calls
    )
    if backend is None and not backend_options:
        executor = _create_joblib_parallel(workers, parallel_options)
        try:
            return executor(submitted)
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc)

    backend = backend or "loky"
    _validate_backend_options(backend, backend_options)
    try:
        backend_context = joblib_parallel_backend(backend, **backend_options)
    except Exception as exc:
        raise ValidationError(f"并行后端配置无效: {exc}") from exc
    with backend_context:
        executor = _create_joblib_parallel(workers, parallel_options)
        try:
            return executor(submitted)
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc)


class ParallelizableMixin:
    """为估计器提供统一并行执行入口的内部混入类。"""

    def _parallel_execute(
        self,
        function: Callable[[Task], Result],
        tasks: Iterable[Task],
        **kwargs: Any,
    ) -> List[Result]:
        """使用实例保存的公共并行配置执行任务。"""
        return parallel_execute(
            function,
            tasks,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
            **kwargs,
        )
