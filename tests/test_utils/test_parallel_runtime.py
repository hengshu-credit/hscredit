"""并行运行时配置测试。"""

from time import sleep
from unittest.mock import patch

import pytest

from hscredit.exceptions import ParallelExecutionError, ValidationError
from hscredit.utils import (
    ParallelBudget,
    ParallelizableMixin,
    get_physical_cpu_count,
    parallel_execute,
    resolve_n_jobs,
    split_parallel_budget,
    validate_parallel_config,
)
from hscredit.utils.parallel import _ACTIVE_BUDGET


def _square(value):
    return value * value


def _read_active_budget(_):
    return _ACTIVE_BUDGET.get()


def _fail_on_two(value):
    if value == 2:
        raise KeyError("boom")
    return value


_FAIL_FAST_STARTED = []


def _fail_first_and_record_started(value):
    _FAIL_FAST_STARTED.append(value)
    if value == 0:
        raise KeyError("boom")
    sleep(0.05)
    return value


def _run_inner_parallel(_):
    return parallel_execute(
        _read_active_budget,
        range(3),
        n_jobs=2,
        parallel_backend="threading",
    )


class _ParallelExecutor(ParallelizableMixin):
    n_jobs = 2
    parallel_backend = "threading"
    parallel_config = None


def test_physical_cpu_count_falls_back_to_joblib_without_keyword_support():
    """旧版 joblib 不支持物理核参数时仍应使用其 CPU 查询结果。"""
    def cpu_count(*args, **kwargs):
        if kwargs:
            raise TypeError("unexpected keyword argument")
        return 4

    with patch("hscredit.utils.parallel.joblib_cpu_count", side_effect=cpu_count):
        with patch("hscredit.utils.parallel.os.cpu_count", return_value=1):
            assert get_physical_cpu_count() == 4


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [(1, 1), (2, 1), (8, 7), (16, 13)],
)
def test_auto_workers_use_eighty_percent_and_leave_one_cpu(cpus, expected):
    """自动并行度应保留至少一个 CPU。"""
    assert resolve_n_jobs(-1, cpu_count=cpus) == expected


@pytest.mark.parametrize(
    ("value", "cpus", "expected"),
    [(1, 8, 1), (1.0, 8, 1), (2.0, 8, 2), (0.25, 8, 2), (0.26, 8, 3)],
)
def test_explicit_worker_forms(value, cpus, expected):
    """显式并行度支持正整数和 CPU 比例。"""
    assert resolve_n_jobs(value, cpu_count=cpus) == expected


def test_task_count_caps_workers():
    """任务数不足时不创建空闲工作进程。"""
    assert resolve_n_jobs(-1, task_count=2, cpu_count=16) == 2


@pytest.mark.parametrize(
    ("cpus", "available_budget", "expected"),
    [(8, 8, 7), (16, 8, 8)],
)
def test_available_budget_caps_automatic_workers(cpus, available_budget, expected):
    """嵌套预算只能收紧自动工作数，不能绕过保守 CPU 限制。"""
    assert resolve_n_jobs(-1, cpu_count=cpus, available_budget=available_budget) == expected


@pytest.mark.parametrize("available_budget", [True, 0, -1, 1.5, "2"])
def test_invalid_available_budget_raises_chinese_validation_error(available_budget):
    """嵌套预算必须为正整数。"""
    with pytest.raises(ValidationError, match="available_budget"):
        resolve_n_jobs(-1, cpu_count=8, available_budget=available_budget)


def test_serial_n_jobs_still_validates_available_budget():
    """兼容的串行模式也不能绕过嵌套预算校验。"""
    with pytest.raises(ValidationError, match="available_budget"):
        resolve_n_jobs(None, available_budget=0)


def test_large_integral_n_jobs_preserves_exact_worker_count():
    """大整数工作数不能因浮点转换而丢失精度。"""
    large_n_jobs = 2**53 + 1
    assert resolve_n_jobs(large_n_jobs, cpu_count=8) == large_n_jobs


@pytest.mark.parametrize("value", [True, 0, -2, 1.5, "2", object()])
def test_invalid_n_jobs_raises_chinese_validation_error(value):
    """无效并行度应抛出统一校验异常。"""
    with pytest.raises(ValidationError, match="n_jobs"):
        resolve_n_jobs(value, cpu_count=8)


def test_parallel_config_rejects_duplicate_worker_and_backend_sources():
    """并行配置不能重复声明工作数或后端。"""
    with pytest.raises(ValidationError, match="n_jobs"):
        validate_parallel_config(None, {"n_jobs": 2})
    with pytest.raises(ValidationError, match="backend"):
        validate_parallel_config("loky", {"backend": "threading"})


def test_parallel_config_preserves_supported_joblib_values():
    """校验后的配置应保留允许值且不修改调用方对象。"""
    source = {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}
    assert validate_parallel_config("loky", source) == source
    assert source == {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}


def test_parallel_config_rejects_non_string_keys_with_chinese_validation_error():
    """混合类型的未知配置键也必须返回统一中文校验错误。"""
    with pytest.raises(ValidationError, match="parallel_config"):
        validate_parallel_config(None, {"unknown": 1, 1: 2})


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_execute_preserves_submission_order(backend):
    """并行结果必须保持任务提交顺序，而不是完成顺序。"""
    assert parallel_execute(
        _square, [3, 1, 2], n_jobs=2, parallel_backend=backend
    ) == [9, 1, 4]


def test_serial_and_parallel_call_the_same_worker():
    """串并行调度必须复用同一个单任务函数。"""
    serial = parallel_execute(_square, range(8), n_jobs=1)
    parallel = parallel_execute(
        _square, range(8), n_jobs=2, parallel_backend="threading"
    )
    assert parallel == serial


def test_worker_failure_has_chinese_context_and_original_cause():
    """任务失败应补充中文标签并保留原始异常链。"""
    def fail(_):
        raise KeyError("boom")

    with pytest.raises(ParallelExecutionError, match="特征A") as error:
        parallel_execute(fail, [1], task_labels=["特征A"])
    assert isinstance(error.value.__cause__, KeyError)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_real_parallel_worker_failure_keeps_original_direct_cause(backend):
    """真实多 worker 失败必须与串行路径一样直接链接原始异常。"""
    with pytest.raises(ParallelExecutionError, match="特征B") as error:
        parallel_execute(
            _fail_on_two,
            [1, 2, 3],
            n_jobs=2,
            parallel_backend=backend,
            task_labels=["特征A", "特征B", "特征C"],
        )
    assert isinstance(error.value.__cause__, KeyError)
    assert error.value.__cause__.args == ("boom",)


def test_threading_worker_failure_stops_before_all_tasks_start():
    """首个线程任务失败后，joblib 应停止调度尚未启动的副作用任务。"""
    _FAIL_FAST_STARTED.clear()
    with pytest.raises(ParallelExecutionError, match="任务0"):
        parallel_execute(
            _fail_first_and_record_started,
            range(20),
            n_jobs=2,
            parallel_backend="threading",
            parallel_config={"batch_size": 1, "pre_dispatch": 2},
            task_labels=[f"任务{index}" for index in range(20)],
        )

    assert len(_FAIL_FAST_STARTED) < 20


@pytest.mark.parametrize(
    ("parallel_backend", "default_backend"),
    [("threading", None), (None, "threading")],
)
def test_threading_backend_rejects_inner_thread_limit_in_chinese(
    parallel_backend, default_backend
):
    """显式或模块默认 threading 后端都不能泄漏 joblib 英文断言。"""
    with pytest.raises(ValidationError, match="threading.*inner_max_num_threads"):
        parallel_execute(
            _square,
            [1, 2],
            n_jobs=2,
            parallel_backend=parallel_backend,
            default_backend=default_backend,
            parallel_config={"inner_max_num_threads": 1},
        )


@pytest.mark.parametrize(
    ("parallel_backend", "default_backend"),
    [("loky", None), (None, "loky"), (None, None)],
)
def test_loky_and_implicit_default_accept_inner_thread_limit(
    parallel_backend, default_backend
):
    """loky 和未指定模块后端时应通过 joblib 1.0 后端上下文设置线程上限。"""
    assert parallel_execute(
        _square,
        [1, 2],
        n_jobs=2,
        parallel_backend=parallel_backend,
        default_backend=default_backend,
        parallel_config={"inner_max_num_threads": 1},
    ) == [1, 4]


@pytest.mark.parametrize(
    ("parallel_backend", "default_backend", "parallel_config"),
    [
        ("missing-backend", None, None),
        (None, "missing-backend", None),
        ("loky", None, {"backend_kwargs": {"unknown_option": True}}),
    ],
)
def test_invalid_backend_configuration_raises_chinese_validation_error(
    parallel_backend, default_backend, parallel_config
):
    """无效显式/默认后端及后端参数应统一为中文公共校验异常。"""
    with pytest.raises(ValidationError, match="并行后端配置无效"):
        parallel_execute(
            _square,
            [1, 2],
            n_jobs=2,
            parallel_backend=parallel_backend,
            default_backend=default_backend,
            parallel_config=parallel_config,
        )


@pytest.mark.parametrize(
    ("available", "task_count", "has_parallel_children", "expected"),
    [
        (13, 100, True, (4, 3)),
        (13, 100, False, (13, 1)),
    ],
)
def test_split_parallel_budget_uses_square_root_only_for_real_nesting(
    available, task_count, has_parallel_children, expected
):
    """只有真实同时嵌套的任务才按平方根切分预算。"""
    assert split_parallel_budget(
        available, task_count, has_parallel_children
    ) == expected


@pytest.mark.parametrize("available", [True, 0, -1, 1.5, "13"])
def test_parallel_budget_rejects_invalid_available_with_chinese_error(available):
    """预算总量必须是正整数，不能接受 bool 或隐式字符串转换。"""
    with pytest.raises(ValidationError, match="available"):
        ParallelBudget(available, 0)


@pytest.mark.parametrize("depth", [True, -1, 1.5, "0"])
def test_parallel_budget_rejects_invalid_depth_with_chinese_error(depth):
    """预算深度必须是非负整数，不能接受 bool 或隐式字符串转换。"""
    with pytest.raises(ValidationError, match="depth"):
        ParallelBudget(1, depth)


@pytest.mark.parametrize(
    ("available", "task_count", "has_parallel_children", "field"),
    [
        (True, 1, False, "available"),
        ("13", 1, False, "available"),
        (0, 1, False, "available"),
        (1, True, False, "task_count"),
        (1, "2", False, "task_count"),
        (1, -1, False, "task_count"),
        (1, 1, "yes", "has_parallel_children"),
    ],
)
def test_split_parallel_budget_rejects_invalid_inputs_in_chinese(
    available, task_count, has_parallel_children, field
):
    """公开预算切分参数必须由 HSCredit 统一校验，不能泄漏原生转换错误。"""
    with pytest.raises(ValidationError, match=field):
        split_parallel_budget(available, task_count, has_parallel_children)


def test_one_outer_task_with_children_receives_the_full_budget():
    """只有一个外层任务时，其并行子任务可获得完整当前预算。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(13, 0))
    try:
        result = parallel_execute(
            _read_active_budget,
            [None],
            n_jobs=-1,
            has_parallel_children=True,
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert result == [ParallelBudget(13, 1)]


def test_sequential_composite_and_stepwise_labels_do_not_split_budget():
    """阶段名称不能让顺序 Composite/Stepwise 调用被误判为真实嵌套。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(13, 0))
    try:
        result = parallel_execute(
            _read_active_budget,
            [None, None],
            n_jobs=2,
            parallel_backend="threading",
            task_labels=["Composite", "Stepwise"],
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert result == [ParallelBudget(13, 1), ParallelBudget(13, 1)]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_real_workers_receive_divided_child_budget(backend):
    """线程和进程 worker 都必须收到显式传播的真实子预算。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(13, 0))
    try:
        result = parallel_execute(
            _read_active_budget,
            range(4),
            n_jobs=4,
            parallel_backend=backend,
            has_parallel_children=True,
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert result == [ParallelBudget(3, 1)] * 4


def test_sequential_parent_then_inner_parallel_call_keeps_full_budget():
    """串行外层阶段随后启动内部并行时仍应看到完整当前预算。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(13, 0))
    try:
        result = parallel_execute(_run_inner_parallel, [None], n_jobs=1)
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert result == [[ParallelBudget(13, 2)] * 3]


def test_parallelizable_mixin_delegates_instance_configuration():
    """估计器 mixin 应把实例公共配置交给同一执行器。"""
    assert _ParallelExecutor()._parallel_execute(_square, [3, 1, 2]) == [9, 1, 4]
