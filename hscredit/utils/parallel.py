"""并行任务参数工具。"""

import math
import numbers
import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from joblib import Parallel, delayed, parallel_backend as joblib_parallel_backend
from joblib import cpu_count as joblib_cpu_count

from ..exceptions import ParallelExecutionError, ValidationError


NJobs = Optional[Union[int, float]]
Task = TypeVar("Task")
Result = TypeVar("Result")

_PARALLEL_CONFIG_KEYS = {
    "adaptive",
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

_PARALLEL_CAPABILITIES = {
    "vectorized",
    "thread_safe",
    "process_safe",
    "serial_only",
}

_THREAD_MIN_WORK = 2_000_000.0
_PROCESS_MIN_WORK = 1_000_000.0
_THREAD_WORK_PER_WORKER = 1_000_000.0
_PROCESS_WORK_PER_WORKER = 2_000_000.0
_PROCESS_AUTO_MEMORY_BUDGET = 512 * 1024**2

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
        object.__setattr__(self, "available", _validate_integer(self.available, "available", 1))
        object.__setattr__(self, "depth", _validate_integer(self.depth, "depth", 0))


@dataclass(frozen=True)
class ParallelWorkload:
    """描述一次批量运算的规模和安全执行能力。"""

    task_count: int
    rows: int = 1
    columns: int = 1
    data_bytes: int = 0
    cost_per_item: float = 1.0
    capability: str = "process_safe"
    releases_gil: bool = False
    has_parallel_children: bool = False
    auto_max_workers: Optional[int] = None
    operation: str = "批量任务"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_count", _validate_integer(self.task_count, "task_count", 0))
        object.__setattr__(self, "rows", _validate_integer(self.rows, "rows", 0))
        object.__setattr__(self, "columns", _validate_integer(self.columns, "columns", 0))
        object.__setattr__(self, "data_bytes", _validate_integer(self.data_bytes, "data_bytes", 0))
        if isinstance(self.cost_per_item, (bool, np.bool_)) or not isinstance(self.cost_per_item, numbers.Real) or not math.isfinite(float(self.cost_per_item)) or float(self.cost_per_item) <= 0:
            raise ValidationError("cost_per_item 必须为有限正数")
        object.__setattr__(self, "cost_per_item", float(self.cost_per_item))
        if self.capability not in _PARALLEL_CAPABILITIES:
            raise ValidationError(f"capability 必须为以下值之一: {sorted(_PARALLEL_CAPABILITIES)}")
        if not isinstance(self.releases_gil, (bool, np.bool_)):
            raise ValidationError("releases_gil 必须为布尔值")
        if not isinstance(self.has_parallel_children, (bool, np.bool_)):
            raise ValidationError("has_parallel_children 必须为布尔值")
        if self.auto_max_workers is not None:
            object.__setattr__(
                self,
                "auto_max_workers",
                _validate_integer(self.auto_max_workers, "auto_max_workers", 1),
            )
        if not isinstance(self.operation, str):
            raise ValidationError("operation 必须为字符串")

    @property
    def estimated_work(self) -> float:
        """返回不依赖运行时计时的确定性工作量估计。"""
        return float(self.rows * self.columns) * self.cost_per_item


@dataclass(frozen=True)
class ParallelExecutionPlan:
    """一次并行调用的只读有效执行计划。"""

    requested_workers: int
    workers: int
    backend: Optional[str]
    adaptive: bool
    estimated_work: float
    data_bytes: int
    child_budget: int
    operation: str


_ACTIVE_BUDGET: ContextVar[Optional[ParallelBudget]] = ContextVar("hscredit_parallel_budget", default=None)


class _WorkerExecutionError(Exception):
    """在线程和进程 worker 间传递原始异常的内部包装。"""

    def __init__(self, label: Any, original_exception: BaseException) -> None:
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

    if available_budget is not None:
        workers = min(workers, available_budget)
    if task_count is not None:
        workers = min(workers, max(1, int(task_count)))
    return max(1, workers)


def validate_parallel_config(parallel_backend: Optional[str], parallel_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
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

    adaptive = config.get("adaptive")
    if adaptive is not None and not isinstance(adaptive, (bool, np.bool_)):
        raise ValidationError("parallel_config.adaptive 必须为布尔值")

    return config


def plan_parallel_execution(
    n_jobs: NJobs,
    workload: ParallelWorkload,
    *,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Mapping[str, Any]] = None,
    default_backend: Optional[str] = None,
    cpu_count: Optional[int] = None,
    available_budget: Optional[int] = None,
) -> ParallelExecutionPlan:
    """根据用户预算和工作负载生成确定性的有效执行计划。"""
    if not isinstance(workload, ParallelWorkload):
        raise ValidationError("workload 必须为 ParallelWorkload")

    config = validate_parallel_config(parallel_backend, parallel_config)
    requested_workers = (
        resolve_n_jobs(
            n_jobs,
            cpu_count=cpu_count,
            available_budget=available_budget,
        )
        or 1
    )
    if available_budget is not None:
        requested_workers = min(requested_workers, int(available_budget))

    backend = parallel_backend or default_backend
    if backend is None:
        if workload.capability == "thread_safe" or (
            workload.capability == "process_safe" and workload.releases_gil
        ):
            backend = "threading"
        elif workload.capability == "process_safe":
            backend = "loky"

    adaptive = (
        bool(config.get("adaptive", True))
        and parallel_backend is None
        and n_jobs == -1
    )
    estimated_work = workload.estimated_work
    if workload.capability in {"serial_only", "vectorized"}:
        workers = 1
    elif not adaptive:
        workers = requested_workers
    else:
        if workload.capability == "thread_safe":
            minimum_work = _THREAD_MIN_WORK
            work_per_worker = _THREAD_WORK_PER_WORKER
        else:
            minimum_work = _PROCESS_MIN_WORK
            work_per_worker = _PROCESS_WORK_PER_WORKER

        if estimated_work < minimum_work or workload.task_count < 2:
            workers = 1
        else:
            profitable_workers = max(2, int(math.ceil(estimated_work / work_per_worker)))
            workers = min(requested_workers, workload.task_count, profitable_workers)

        if workload.auto_max_workers is not None:
            workers = min(workers, workload.auto_max_workers)
        if backend == "loky" and workload.data_bytes > 0:
            memory_workers = max(1, _PROCESS_AUTO_MEMORY_BUDGET // workload.data_bytes)
            workers = min(workers, memory_workers)

    workers = max(1, min(workers, max(1, workload.task_count)))
    total_budget = int(available_budget or requested_workers)
    if workload.has_parallel_children:
        # ``n_jobs`` 是整个调用树的总预算，而不是每一层都可重复使用的
        # worker 数。即使用户显式指定后端或关闭自适应，也必须为真实并行
        # 子任务保留预算，否则会形成 outer × inner 的过度订阅。
        outer_limit, _ = split_parallel_budget(total_budget, workload.task_count, True)
        workers = min(workers, outer_limit)
        child_budget = max(1, total_budget // workers)
    else:
        child_budget = total_budget

    return ParallelExecutionPlan(
        requested_workers=requested_workers,
        workers=workers,
        backend=backend,
        adaptive=adaptive,
        estimated_work=estimated_work,
        data_bytes=workload.data_bytes,
        child_budget=child_budget,
        operation=workload.operation,
    )


def split_parallel_budget(available: int, task_count: int, has_parallel_children: bool) -> Tuple[int, int]:
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
    except KeyboardInterrupt as exc:
        # loky 不会把 worker 内的 BaseException 当作普通任务失败及时回传；
        # 包装后由主进程恢复原始 KeyboardInterrupt，并立即取消剩余任务。
        raise _WorkerExecutionError(label, exc) from exc
    except Exception as exc:
        raise _WorkerExecutionError(label, exc) from exc
    finally:
        _ACTIVE_BUDGET.reset(token)


def _run_serial_batch(calls: Sequence[Tuple[Any, ...]]) -> List[Any]:
    """在一个可超时的 joblib 批次中严格顺序执行多个任务。"""
    return [_run_with_budget(*call) for call in calls]


def _raise_parallel_execution_error(
    error: _WorkerExecutionError,
    *,
    preserve_exceptions: bool = False,
) -> NoReturn:
    """按调用策略恢复原始异常或统一的中文并行错误。"""
    if preserve_exceptions:
        raise error.original_exception
    raise ParallelExecutionError(f"并行任务 '{error.label}' 执行失败: {error.original_exception}") from error.original_exception


def _current_parallel_budget() -> ParallelBudget:
    """返回当前预算；根调用使用统一的自动并行预算。"""
    active_budget = _ACTIVE_BUDGET.get()
    if active_budget is not None:
        return active_budget
    return ParallelBudget(resolve_n_jobs(-1) or 1, 0)


def _resolve_current_n_jobs(
    n_jobs: NJobs,
    task_count: Optional[int] = None,
) -> Optional[int]:
    """解析当前调用的 worker 数，仅在真实嵌套上下文中施加父级预算。"""
    active_budget = _ACTIVE_BUDGET.get()
    if active_budget is None:
        return resolve_n_jobs(n_jobs, task_count=task_count)
    if n_jobs == -1:
        # 父级已经把自动并行度解析为可下放预算；子级继承该值，不能再次按容器可见 CPU 缩减。
        n_jobs = active_budget.available
    return resolve_n_jobs(
        n_jobs,
        task_count=task_count,
        cpu_count=active_budget.available,
        available_budget=active_budget.available,
    )


def resolve_native_workers(
    n_jobs: NJobs,
    native_workers: Optional[int] = None,
) -> int:
    """统一解析 ``thread_count``、``num_workers`` 等原生线程参数。"""
    if native_workers is not None:
        native_workers = _validate_integer(native_workers, "native_workers", 1)
    budget = _current_parallel_budget()
    active = _ACTIVE_BUDGET.get() is not None
    requested = _resolve_current_n_jobs(n_jobs) or 1
    if active:
        requested = min(requested, budget.available)
    if n_jobs == -1 and native_workers is not None:
        return max(1, min(native_workers, budget.available))
    return max(1, requested)


def _create_joblib_parallel(workers: int, options: Mapping[str, Any]) -> Parallel:
    """创建 joblib 执行器，并将公开配置错误转换为中文异常。"""
    try:
        return Parallel(n_jobs=workers, **dict(options))
    except Exception as exc:
        raise ValidationError(f"并行后端配置无效: {exc}") from exc


def _validate_parallel_backend(backend: Optional[str], backend_options: Mapping[str, Any]) -> None:
    """在任务数量触发串行短路前验证有效 joblib 后端。"""
    if backend is None:
        return
    _validate_backend_options(backend, backend_options)
    try:
        with joblib_parallel_backend(backend, **dict(backend_options)):
            pass
    except Exception as exc:
        raise ValidationError(f"并行后端配置无效: {exc}") from exc


def _infer_legacy_workload(
    tasks: Sequence[Any],
    *,
    default_backend: Optional[str],
    has_parallel_children: bool,
) -> ParallelWorkload:
    """从旧调用的任务载荷保守推断规模，避免把标量任务当成重计算。"""
    seen_payloads = set()
    data_bytes = 0
    total_rows = 0

    def inspect_payload(value: Any, depth: int = 0) -> Tuple[int, int]:
        nonlocal data_bytes
        if depth > 2:
            return 0, 0

        if isinstance(value, np.ndarray):
            rows = int(value.shape[0]) if value.ndim else 1
            object_id = id(value)
            if object_id not in seen_payloads:
                seen_payloads.add(object_id)
                data_bytes += int(value.nbytes)
            return rows, int(value.size)

        shape = getattr(value, "shape", None)
        memory_usage = getattr(value, "memory_usage", None)
        if shape is not None and callable(memory_usage):
            try:
                rows = int(shape[0]) if len(shape) else 1
                usage = memory_usage(deep=True)
                size = int(usage.sum()) if hasattr(usage, "sum") else int(usage)
            except (TypeError, ValueError, AttributeError):
                rows, size = 0, 0
            object_id = id(value)
            if object_id not in seen_payloads:
                seen_payloads.add(object_id)
                data_bytes += max(0, size)
            return rows, max(0, size)

        if isinstance(value, Mapping):
            metrics = [inspect_payload(item, depth + 1) for item in value.values()]
        elif isinstance(value, (tuple, list)):
            metrics = [inspect_payload(item, depth + 1) for item in value]
        else:
            return 0, 0
        return (
            max((rows for rows, _ in metrics), default=0),
            sum(size for _, size in metrics),
        )

    for task in tasks:
        task_rows, _ = inspect_payload(task)
        total_rows += max(1, task_rows)

    capability = "thread_safe" if default_backend == "threading" else "process_safe"
    return ParallelWorkload(
        task_count=len(tasks),
        rows=total_rows,
        columns=1,
        data_bytes=data_bytes,
        cost_per_item=4.0 if capability == "thread_safe" else 8.0,
        capability=capability,
        releases_gil=capability == "thread_safe",
        has_parallel_children=has_parallel_children,
        operation="旧调用自动推断任务",
    )


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
    workload: Optional[ParallelWorkload] = None,
    preserve_exceptions: bool = False,
) -> List[Result]:
    """按提交顺序执行任务，并在线程和进程间传播并行预算。"""
    config = validate_parallel_config(parallel_backend, parallel_config)
    current_budget = _current_parallel_budget()
    if workload is None:
        # 旧调用没有提供规模元数据，只能展开一次以获得确定任务数。
        # 新调用传入 workload 后保留生成器惰性，让 joblib 的 pre_dispatch
        # 真正限制待创建任务及大型参数对象的峰值内存。
        task_source: Iterable[Task] = list(tasks)
        task_count = len(task_source)  # type: ignore[arg-type]
        workload = _infer_legacy_workload(
            task_source,  # type: ignore[arg-type]
            default_backend=default_backend,
            has_parallel_children=has_parallel_children,
        )
    else:
        task_source = tasks
        task_count = workload.task_count
        if hasattr(tasks, "__len__") and len(tasks) != task_count:  # type: ignore[arg-type]
            raise ValidationError("workload.task_count 必须与 tasks 数量一致")

    if has_parallel_children and not workload.has_parallel_children:
        workload = replace(workload, has_parallel_children=True)

    if task_labels is None:
        label_source: Optional[Iterable[Any]] = None
    else:
        label_source = task_labels
        if hasattr(task_labels, "__len__") and len(task_labels) != task_count:  # type: ignore[arg-type]
            raise ValidationError("task_labels 的数量必须与 tasks 一致")

    def iter_task_calls(child_budget: int, depth: int):
        """惰性配对任务和标签，并在消费边界校验声明数量。"""
        task_iterator = iter(task_source)
        label_iterator = iter(label_source) if label_source is not None else None
        for position in range(task_count):
            try:
                task = next(task_iterator)
            except StopIteration as exc:
                raise ValidationError("workload.task_count 必须与 tasks 数量一致") from exc
            if label_iterator is None:
                label = position
            else:
                try:
                    label = next(label_iterator)
                except StopIteration as exc:
                    raise ValidationError("task_labels 的数量必须与 tasks 一致") from exc
            yield function, task, label, child_budget, depth

        try:
            next(task_iterator)
        except StopIteration:
            pass
        else:
            raise ValidationError("workload.task_count 必须与 tasks 数量一致")
        if label_iterator is not None:
            try:
                next(label_iterator)
            except StopIteration:
                pass
            else:
                raise ValidationError("task_labels 的数量必须与 tasks 一致")

    plan = plan_parallel_execution(
        n_jobs,
        workload,
        parallel_backend=parallel_backend,
        parallel_config=config,
        default_backend=default_backend,
        available_budget=(current_budget.available if _ACTIVE_BUDGET.get() is not None else None),
    )

    parallel_options = dict(config)
    parallel_options.pop("adaptive", None)
    backend_options = parallel_options.pop("backend_kwargs", {}) or {}
    inner_max_num_threads = parallel_options.pop("inner_max_num_threads", None)

    backend = plan.backend
    if inner_max_num_threads is not None:
        if backend == "threading":
            raise ValidationError("threading 后端不支持 inner_max_num_threads")
        backend_options["inner_max_num_threads"] = inner_max_num_threads
    if backend is None and backend_options:
        backend = "loky"
    _validate_parallel_backend(backend, backend_options)

    if task_count == 0:
        # 对无长度生成器仍检查 workload=0 是否如实声明为空。
        try:
            next(iter(task_source))
        except StopIteration:
            pass
        else:
            raise ValidationError("workload.task_count 必须与 tasks 数量一致")
        return []

    workers = plan.workers
    child_budget = plan.child_budget
    depth = current_budget.depth + 1
    enforce_timeout = parallel_options.get("timeout") is not None
    if enforce_timeout and backend == "sequential":
        raise ValidationError("sequential 后端不支持 timeout")

    calls = iter_task_calls(child_budget, depth)
    if workers == 1 and not enforce_timeout:
        try:
            return [_run_with_budget(*call) for call in calls]
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc, preserve_exceptions=preserve_exceptions)

    serial_timeout_batch = workers == 1 and enforce_timeout
    submitted: Iterable[Any]
    if serial_timeout_batch:
        # joblib 的 n_jobs=1 会绕过后端并忽略 timeout；用一个批次交给
        # n_jobs=2 的执行器可保留 timeout，同时批次内部仍严格串行。
        submitted = [delayed(_run_serial_batch)(list(calls))]
    else:
        submitted = (delayed(_run_with_budget)(*call) for call in calls)
    executor_workers = 2 if serial_timeout_batch else workers
    if backend is None and not backend_options:
        executor = _create_joblib_parallel(executor_workers, parallel_options)
        try:
            result = executor(submitted)
            return result[0] if serial_timeout_batch else result
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc, preserve_exceptions=preserve_exceptions)
        except ValidationError:
            raise
        except Exception as exc:
            if preserve_exceptions:
                raise
            raise ParallelExecutionError(f"并行任务执行失败: {exc}") from exc

    backend = backend or "loky"
    try:
        backend_context = joblib_parallel_backend(backend, **backend_options)
    except Exception as exc:
        raise ValidationError(f"并行后端配置无效: {exc}") from exc
    with backend_context:
        executor = _create_joblib_parallel(executor_workers, parallel_options)
        try:
            result = executor(submitted)
            return result[0] if serial_timeout_batch else result
        except _WorkerExecutionError as exc:
            _raise_parallel_execution_error(exc, preserve_exceptions=preserve_exceptions)
        except ValidationError:
            raise
        except Exception as exc:
            if preserve_exceptions:
                raise
            raise ParallelExecutionError(f"并行任务执行失败: {exc}") from exc


class ParallelizableMixin:
    """为估计器提供统一并行执行入口的内部混入类。"""

    n_jobs: NJobs
    parallel_backend: Optional[str]
    parallel_config: Optional[Mapping[str, Any]]

    def _validate_parallel_configuration(self, default_backend: Optional[str] = None) -> None:
        """校验实例并行配置，但不创建 worker。"""
        config = validate_parallel_config(self.parallel_backend, self.parallel_config)
        backend_options = dict(config.get("backend_kwargs", {}) or {})
        inner_max_num_threads = config.get("inner_max_num_threads")
        backend = self.parallel_backend or default_backend
        if inner_max_num_threads is not None:
            if backend == "threading":
                raise ValidationError("threading 后端不支持 inner_max_num_threads")
            backend = backend or "loky"
            backend_options["inner_max_num_threads"] = inner_max_num_threads
        _validate_parallel_backend(backend, backend_options)

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
