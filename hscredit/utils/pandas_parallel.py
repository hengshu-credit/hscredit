"""pandas apply 并行代理。

集中管理 DataFrame、Series 和 GroupBy 对象的链式并行配置，并在不改变
pandas 原生 ``apply`` 参数边界的前提下调度具体执行器。
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from joblib.externals import cloudpickle

from ..exceptions import ValidationError
from .parallel import (
    ParallelWorkload,
    _validate_parallel_backend,
    parallel_execute,
    resolve_n_jobs,
    validate_parallel_config,
)


_THREAD_SAFE_CALLABLES = (
    np.all,
    np.any,
    np.max,
    np.mean,
    np.median,
    np.min,
    np.prod,
    np.std,
    np.sum,
    np.var,
)


class _ApplyProgressReporter:
    """主进程中线程安全的 apply 进度报告器。"""

    def __init__(self, enabled: bool, total: int, description: str):
        self.enabled = bool(enabled)
        # 禁用进度时该对象仍会随 loky 任务序列化；不要携带不可 pickle 的线程锁。
        self._lock = threading.Lock() if self.enabled else None
        self._bar = None
        if self.enabled:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=int(total), desc=description, unit="项", dynamic_ncols=False)

    def advance(self, count: int = 1) -> None:
        """累计已经真实完成的逻辑任务。"""
        if not self.enabled:
            return
        assert self._lock is not None
        with self._lock:
            assert self._bar is not None
            self._bar.update(int(count))

    def close(self) -> None:
        """关闭 tqdm 资源。"""
        if not self.enabled:
            return
        assert self._lock is not None
        with self._lock:
            assert self._bar is not None
            self._bar.close()


class _QueueApplyProgressReporter:
    """供 loky worker 向主进程报告完成数量的可序列化代理。"""

    def __init__(self, event_queue):
        self._event_queue = event_queue

    def advance(self, count: int = 1) -> None:
        self._event_queue.put(("advance", int(count)))


def _monitor_apply_progress(event_queue, reporter: _ApplyProgressReporter) -> None:
    """消费 worker 进度事件，直到收到显式停止标记。"""
    while True:
        action, count = event_queue.get()
        if action == "stop":
            return
        if action == "advance":
            reporter.advance(count)


@contextmanager
def _apply_progress_context(enabled: bool, total: int, description: str, backend: Optional[str]):
    """创建适合当前后端的 reporter，并保证所有退出路径都清理资源。"""
    reporter = _ApplyProgressReporter(enabled, total, description)
    manager = None
    event_queue = None
    monitor = None
    task_reporter: Any = reporter
    try:
        if enabled and backend in {"loky", "multiprocessing"}:
            from joblib.externals.loky.backend.context import get_context

            manager = get_context().Manager()
            event_queue = manager.Queue()
            task_reporter = _QueueApplyProgressReporter(event_queue)
            monitor = threading.Thread(
                target=_monitor_apply_progress,
                args=(event_queue, reporter),
                daemon=True,
            )
            monitor.start()
        yield task_reporter
    finally:
        if event_queue is not None:
            try:
                event_queue.put(("stop", 0))
            except Exception:
                pass
        if monitor is not None:
            monitor.join(timeout=2.0)
        reporter.close()
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass


@dataclass(frozen=True)
class HSCreditApplyProxy:
    """保存一次 pandas apply 调用的并行配置。

    **参数**
        _obj: DataFrame、Series 或相应的 GroupBy 对象。
        n_jobs: 总并行预算，默认 ``-1`` 表示自动使用约 80% 的物理核心。
        bar: 是否显示按真实完成项累计的进度条。
        parallel_backend: 可选的 joblib 后端；未指定时按 callable 能力静态选择。
        parallel_config: ``batch_size``、``timeout`` 等统一并行运行参数。

    **属性**
        所有配置均只属于当前代理；创建代理不会修改原 pandas 对象。

    **参考样例**
        ``df.hscredit(n_jobs=-1, bar=True).apply(func, axis=1)``
    """

    _obj: Any
    n_jobs: Any = -1
    bar: bool = True
    parallel_backend: Optional[str] = None
    parallel_config: Optional[Mapping[str, Any]] = None

    def apply(self, func, *args, **kwargs):
        """使用已配置的 hscredit 执行策略调用 pandas apply。"""
        return _apply_object(self, func, args, kwargs)


def create_hscredit_apply_proxy(
    self,
    n_jobs=-1,
    bar: bool = True,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Mapping[str, Any]] = None,
) -> HSCreditApplyProxy:
    """为当前 pandas 对象创建独立的 apply 配置代理。

    此函数会绑定为 pandas 对象的 ``hscredit`` 方法。它只保存配置，不读取样本、
    不调用用户函数，也不会改变原对象。
    """
    return HSCreditApplyProxy(
        self,
        n_jobs=n_jobs,
        bar=bar,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
    )


def _bind_native_apply(obj, func, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """按当前 pandas 版本的原生签名拆分控制参数和 UDF 参数。"""
    signature = inspect.signature(obj.apply)
    bound = signature.bind(func, *args, **dict(kwargs))
    bound.apply_defaults()
    return dict(bound.arguments)


def _native_apply(obj, func, args: Tuple[Any, ...], kwargs: Mapping[str, Any]):
    """在明确的兼容回退路径调用 pandas 原生 apply。"""
    return obj.apply(func, *args, **dict(kwargs))


def _native_apply_with_progress(
    proxy: HSCreditApplyProxy,
    obj,
    func,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    total: int,
    description: str,
):
    """为不可拆分的原生 apply 显示一次真实完成更新。"""
    with _apply_progress_context(proxy.bar, total, description, None) as reporter:
        result = _native_apply(obj, func, args, kwargs)
        reporter.advance(total)
        return result


def _classify_callable(func) -> str:
    """不执行 callable，静态判断适合的执行能力。"""
    if isinstance(func, np.ufunc):
        return "vectorized"
    if inspect.isbuiltin(func) or any(func is candidate for candidate in _THREAD_SAFE_CALLABLES):
        return "thread_safe"
    return "process_safe"


def _is_cloudpickle_serializable(*values: Any) -> bool:
    """只检查序列化能力，不调用用户函数。"""
    try:
        cloudpickle.dumps(values)
    except Exception:
        return False
    return True


def _resolve_default_backend(proxy: HSCreditApplyProxy, capability: str, func, udf_args, udf_kwargs) -> Optional[str]:
    """合并显式后端和静态 callable 分类。"""
    if proxy.parallel_backend is not None:
        return None
    if capability == "thread_safe":
        return "threading"
    if capability == "process_safe":
        if _is_cloudpickle_serializable(func, udf_args, udf_kwargs):
            return "loky"
        return "threading"
    return None


def _call_apply_item(task):
    """执行一个已编号的 pandas apply 逻辑任务。"""
    position, func, value, udf_args, udf_kwargs, reporter = task
    result = func(value, *udf_args, **udf_kwargs)
    if isinstance(result, pd.Series):
        result = result.copy(deep=False)
    reporter.advance(1)
    return position, result


def _is_indexed_like(result, group_axes, axis: int) -> bool:
    """兼容 pandas GroupBy 对“结果索引未改变”的判断。"""
    if isinstance(result, pd.Series):
        return len(group_axes) == 1 and result.index.equals(group_axes[axis])
    if isinstance(result, pd.DataFrame):
        return result.axes[axis].equals(group_axes[axis])
    return False


def _normalize_group_key(key):
    """把 NumPy 分组键还原为 pandas 原生 apply 暴露的 Python 标量。"""
    if isinstance(key, tuple):
        return tuple(_normalize_group_key(value) for value in key)
    if isinstance(key, np.generic):
        return key.item()
    return key


def _call_group_apply_item(task):
    """执行一个 GroupBy 分组并返回 pandas 装配所需的索引变化标记。"""
    position, key, group, func, udf_args, udf_kwargs, axis, reporter = task
    object.__setattr__(group, "name", _normalize_group_key(key))
    group_axes = group.axes
    result = func(group, *udf_args, **udf_kwargs)
    if isinstance(result, pd.Series):
        result = result.copy(deep=False)
    reporter.advance(1)
    return position, result, not _is_indexed_like(result, group_axes, axis)


def _execute_items(
    proxy: HSCreditApplyProxy,
    tasks,
    *,
    capability: str,
    func,
    udf_args,
    udf_kwargs,
    rows: int,
    columns: int,
    data_bytes: int,
    operation: str,
):
    """通过 hscredit 统一预算执行已物化的 apply 任务。"""
    default_backend = _resolve_default_backend(proxy, capability, func, udf_args, udf_kwargs)
    effective_backend = proxy.parallel_backend or default_backend
    with _apply_progress_context(proxy.bar, len(tasks), operation, effective_backend) as reporter:
        progress_tasks = [(*task, reporter) for task in tasks]
        return parallel_execute(
            _call_apply_item,
            progress_tasks,
            n_jobs=proxy.n_jobs,
            parallel_backend=proxy.parallel_backend,
            parallel_config=proxy.parallel_config,
            default_backend=default_backend,
            task_labels=range(len(tasks)),
            workload=ParallelWorkload(
                task_count=len(tasks),
                rows=rows,
                columns=columns,
                data_bytes=data_bytes,
                cost_per_item=128.0,
                capability=capability,
                releases_gil=capability == "thread_safe",
                operation=operation,
            ),
            preserve_exceptions=True,
        )


def _execute_group_items(
    proxy: HSCreditApplyProxy,
    tasks,
    *,
    capability: str,
    func,
    udf_args,
    udf_kwargs,
    rows: int,
    columns: int,
    data_bytes: int,
):
    """通过统一预算执行按组任务。"""
    default_backend = _resolve_default_backend(proxy, capability, func, udf_args, udf_kwargs)
    effective_backend = proxy.parallel_backend or default_backend
    description = "GroupBy 分组计算"
    with _apply_progress_context(proxy.bar, len(tasks), description, effective_backend) as reporter:
        progress_tasks = [(*task, reporter) for task in tasks]
        return parallel_execute(
            _call_group_apply_item,
            progress_tasks,
            n_jobs=proxy.n_jobs,
            parallel_backend=proxy.parallel_backend,
            parallel_config=proxy.parallel_config,
            default_backend=default_backend,
            task_labels=[task[1] for task in tasks],
            workload=ParallelWorkload(
                task_count=len(tasks),
                rows=rows,
                columns=columns,
                data_bytes=data_bytes,
                cost_per_item=256.0,
                capability=capability,
                releases_gil=capability == "thread_safe",
                operation=description,
            ),
            preserve_exceptions=True,
        )


def _frame_apply_adapter(df, func, parameters: Mapping[str, Any]):
    """构造当前 pandas 版本的 FrameApply，但不执行用户函数。"""
    from pandas.core.apply import frame_apply

    candidate = {
        "func": func,
        "axis": df._get_axis_number(parameters.get("axis", 0)),
        "raw": parameters.get("raw", False),
        "result_type": parameters.get("result_type"),
        "by_row": parameters.get("by_row", "compat"),
        "engine": parameters.get("engine", "python"),
        "engine_kwargs": parameters.get("engine_kwargs"),
        "args": parameters.get("args", ()),
        "kwargs": parameters.get("kwargs", {}),
    }
    supported = inspect.signature(frame_apply).parameters
    return frame_apply(df, **{name: value for name, value in candidate.items() if name in supported})


def _execute_dataframe_apply(
    proxy: HSCreditApplyProxy,
    func,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
):
    """并行执行 DataFrame apply 并交回 pandas 统一装配。"""
    df = proxy._obj
    parameters = _bind_native_apply(df, func, args, kwargs)
    capability = _classify_callable(func) if callable(func) else "vectorized"
    axis = df._get_axis_number(parameters.get("axis", 0))
    raw = bool(parameters.get("raw", False))
    result_type = parameters.get("result_type")
    engine = parameters.get("engine", "python")
    logical_tasks = len(df.columns) if axis == 0 else len(df.index)
    resolve_n_jobs(proxy.n_jobs, task_count=logical_tasks)

    if (
        not callable(func)
        or capability == "vectorized"
        or proxy.parallel_backend == "sequential"
        or raw
        or result_type == "broadcast"
        or engine not in (None, "python")
        or not all(df.shape)
    ):
        return _native_apply_with_progress(
            proxy,
            df,
            func,
            args,
            kwargs,
            total=logical_tasks,
            description="DataFrame 行计算" if axis == 1 else "DataFrame 列计算",
        )

    operation = _frame_apply_adapter(df, func, parameters)
    values = []
    for value in operation.series_generator:
        values.append(value.copy(deep=axis == 1))

    udf_args = tuple(parameters.get("args", ()))
    udf_kwargs = dict(parameters.get("kwargs", {}))
    tasks = [(position, func, value, udf_args, udf_kwargs) for position, value in enumerate(values)]
    completed = _execute_items(
        proxy,
        tasks,
        capability=capability,
        func=func,
        udf_args=udf_args,
        udf_kwargs=udf_kwargs,
        rows=len(df),
        columns=len(df.columns),
        data_bytes=int(df.memory_usage(deep=True).sum()),
        operation="DataFrame 行计算" if axis == 1 else "DataFrame 列计算",
    )
    results = {position: result for position, result in completed}
    return operation.wrap_results(results, operation.result_index)


def _series_apply_adapter(series, func, parameters: Mapping[str, Any]):
    """构造当前 pandas 版本的 SeriesApply 控制对象。"""
    from pandas._libs import lib
    from pandas.core.apply import SeriesApply

    candidate = {
        "convert_dtype": parameters.get("convert_dtype", lib.no_default),
        "by_row": parameters.get("by_row", "compat"),
        "args": parameters.get("args", ()),
        "kwargs": parameters.get("kwargs", {}),
    }
    supported = inspect.signature(SeriesApply).parameters
    return SeriesApply(series, func, **{name: value for name, value in candidate.items() if name in supported})


def _boxed_series_apply_values(series: pd.Series, convert_dtype: bool):
    """按 pandas 原生 map 路径生成传给 UDF 的 Python/扩展标量。"""
    values = series._values
    if isinstance(values, np.ndarray):
        # pandas.core.algorithms.map_array 同样先转换为 object，lib.map_infer
        # 再把这些 Python 标量交给用户函数。
        return values.astype(object, copy=False).tolist()

    boxed = []

    def capture(value):
        boxed.append(value)
        return value

    # ExtensionArray 的装箱规则各不相同（如 Int64 会把缺失值转为 float NaN，
    # DatetimeArray 则传 Timestamp）。复用当前 pandas 的映射入口只做装箱，
    # 不执行用户函数，也不基于返回值探测或重试。
    series._map_values(capture, convert=convert_dtype)
    return boxed


def _execute_series_apply(
    proxy: HSCreditApplyProxy,
    func,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
):
    """并行执行 Series apply 并按全局首个结果恢复类型。"""
    series = proxy._obj
    parameters = _bind_native_apply(series, func, args, kwargs)
    capability = _classify_callable(func) if callable(func) else "vectorized"
    resolve_n_jobs(proxy.n_jobs, task_count=len(series))
    operation = _series_apply_adapter(series, func, parameters) if callable(func) else None
    whole_series_call = operation is not None and operation.by_row is False

    if (
        not callable(func)
        or capability == "vectorized"
        or proxy.parallel_backend == "sequential"
        or whole_series_call
        or pd.api.types.is_categorical_dtype(series.dtype)
    ):
        return _native_apply_with_progress(
            proxy,
            series,
            func,
            args,
            kwargs,
            total=len(series),
            description="Series 元素计算",
        )

    assert operation is not None
    udf_args = tuple(parameters.get("args", ()))
    udf_kwargs = dict(parameters.get("kwargs", {}))
    values = _boxed_series_apply_values(series, operation.convert_dtype)
    tasks = [(position, func, value, udf_args, udf_kwargs) for position, value in enumerate(values)]
    completed = _execute_items(
        proxy,
        tasks,
        capability=capability,
        func=func,
        udf_args=udf_args,
        udf_kwargs=udf_kwargs,
        rows=len(series),
        columns=1,
        data_bytes=int(series.memory_usage(deep=True)),
        operation="Series 元素计算",
    )
    mapped = [result for _, result in completed]
    if mapped and isinstance(mapped[0], pd.Series):
        return series._constructor_expanddim(mapped, index=series.index)

    if not operation.convert_dtype:
        mapped = np.asarray(mapped, dtype=object)
    return series._constructor(mapped, index=series.index).__finalize__(series, method="apply")


def _groupby_data(grouped, parameters: Mapping[str, Any]):
    """按当前 pandas include_groups 语义选择传给 UDF 的数据。"""
    if "include_groups" in parameters and not parameters["include_groups"]:
        return grouped._obj_with_exclusions
    return grouped._selected_obj


def _groupby_memory_usage(data) -> int:
    """计算选择后 GroupBy 数据的深层字节数。"""
    usage = data.memory_usage(deep=True)
    return int(usage.sum()) if hasattr(usage, "sum") else int(usage)


def _execute_groupby_apply(
    proxy: HSCreditApplyProxy,
    func,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
):
    """严格单次执行 GroupBy apply 并复用 pandas 原生结果装配。"""
    from pandas.core import common as com

    grouped = proxy._obj
    parameters = _bind_native_apply(grouped, func, args, kwargs)
    if isinstance(func, str) or not callable(func):
        return _native_apply_with_progress(
            proxy,
            grouped,
            func,
            args,
            kwargs,
            total=int(getattr(grouped, "ngroups", 0)),
            description="GroupBy 分组计算",
        )

    effective_func = getattr(com, "is_builtin_func")(func)
    data = _groupby_data(grouped, parameters)
    grouper = getattr(grouped, "_grouper", None)
    if grouper is None:
        grouper = grouped.grouper
    axis = int(getattr(grouped, "axis", 0))
    keys = list(grouper.group_keys_seq)
    splitter = grouper._get_splitter(data, axis=axis)
    groups = list(splitter)
    if len(groups) != len(keys):
        raise RuntimeError("pandas GroupBy 分组键与数据分组数量不一致")
    if not groups:
        with _apply_progress_context(proxy.bar, 0, "GroupBy 分组计算", None):
            return grouped._wrap_applied_output(data, [], False, False)

    udf_args = tuple(parameters.get("args", ()))
    udf_kwargs = dict(parameters.get("kwargs", {}))
    capability = _classify_callable(effective_func)
    tasks = [
        (position, key, group, effective_func, udf_args, udf_kwargs, axis)
        for position, (key, group) in enumerate(zip(keys, groups))
    ]
    completed = _execute_group_items(
        proxy,
        tasks,
        capability=capability,
        func=effective_func,
        udf_args=udf_args,
        udf_kwargs=udf_kwargs,
        rows=len(data),
        columns=(len(data.columns) if isinstance(data, pd.DataFrame) else 1),
        data_bytes=_groupby_memory_usage(data),
    )
    completed.sort(key=lambda item: item[0])
    values = [result for _, result, _ in completed]
    not_indexed_same = any(mutated for _, _, mutated in completed)
    return grouped._wrap_applied_output(data, values, not_indexed_same, False)


def _apply_object(
    proxy: HSCreditApplyProxy,
    func,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
):
    """按 pandas 对象类型分派并行 apply。"""
    if not isinstance(proxy.bar, (bool, np.bool_)):
        raise ValidationError("bar 必须为布尔值")
    config = validate_parallel_config(proxy.parallel_backend, proxy.parallel_config)
    backend_options = dict(config.get("backend_kwargs", {}) or {})
    inner_max_num_threads = config.get("inner_max_num_threads")
    backend = proxy.parallel_backend
    if inner_max_num_threads is not None:
        if backend == "threading":
            raise ValidationError("threading 后端不支持 inner_max_num_threads")
        backend = backend or "loky"
        backend_options["inner_max_num_threads"] = inner_max_num_threads
    _validate_parallel_backend(backend, backend_options)

    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

    if isinstance(proxy._obj, (DataFrameGroupBy, SeriesGroupBy)):
        return _execute_groupby_apply(proxy, func, args, kwargs)
    if isinstance(proxy._obj, pd.DataFrame):
        return _execute_dataframe_apply(proxy, func, args, kwargs)
    if isinstance(proxy._obj, pd.Series):
        return _execute_series_apply(proxy, func, args, kwargs)
    raise TypeError(f"暂不支持对象类型: {type(proxy._obj).__name__}")


__all__ = ["HSCreditApplyProxy", "create_hscredit_apply_proxy"]
