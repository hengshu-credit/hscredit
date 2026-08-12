"""并行运行时配置测试。"""

from time import sleep
from unittest.mock import patch

import pytest
import threading

from hscredit.exceptions import ParallelExecutionError, ValidationError
from hscredit.utils import (
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
from hscredit.utils.parallel import _ACTIVE_BUDGET


def _square(value):
    return value * value


def _read_active_budget(_):
    return _ACTIVE_BUDGET.get()


def _thread_identity(_):
    return threading.get_ident()


def _sleep_thread_identity(_):
    sleep(0.01)
    return threading.get_ident()


def _fail_on_two(value):
    if value == 2:
        raise KeyError("boom")
    return value


def _raise_lookup_error(value):
    raise LookupError(f"bad-{value}")


def _raise_keyboard_interrupt(value):
    raise KeyboardInterrupt(f"stop-{value}")


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


@pytest.mark.parametrize(("backend", "n_jobs"), [("sequential", 1), ("loky", 2)])
def test_parallel_execute_can_preserve_original_task_exception(backend, n_jobs):
    """pandas apply 需要保留 UDF 原始异常类型，不能统一包装成运行时异常。"""
    with pytest.raises(LookupError, match="bad-1"):
        parallel_execute(
            _raise_lookup_error,
            [1],
            n_jobs=n_jobs,
            parallel_backend=backend,
            preserve_exceptions=True,
        )


@pytest.mark.parametrize(("backend", "n_jobs"), [("sequential", 1), ("loky", 2)])
def test_parallel_execute_can_preserve_keyboard_interrupt(backend, n_jobs):
    """worker 内的中断必须及时回到调用线程，不能等待其他并行任务。"""
    with pytest.raises(KeyboardInterrupt, match="stop-1"):
        parallel_execute(
            _raise_keyboard_interrupt,
            [1],
            n_jobs=n_jobs,
            parallel_backend=backend,
            preserve_exceptions=True,
        )


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


def test_active_budget_caps_explicit_workers_to_prevent_nested_oversubscription():
    """用户显式预算在根调用优先，但进入父级 worker 后不得突破剩余总预算。"""
    assert resolve_n_jobs(8, cpu_count=16, available_budget=2) == 2
    assert resolve_n_jobs(0.75, cpu_count=16, available_budget=3) == 3


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
    assert parallel_execute(_square, [3, 1, 2], n_jobs=2, parallel_backend=backend) == [9, 1, 4]


def test_serial_and_parallel_call_the_same_worker():
    """串并行调度必须复用同一个单任务函数。"""
    serial = parallel_execute(_square, range(8), n_jobs=1)
    parallel = parallel_execute(_square, range(8), n_jobs=2, parallel_backend="threading")
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


def test_explicit_workload_keeps_task_generator_lazy_for_fail_fast():
    """已知任务规模时不得预先展开生成器，否则 pre_dispatch 无法限制内存。"""
    yielded = []

    def tasks():
        for value in range(100):
            yielded.append(value)
            yield value

    workload = ParallelWorkload(
        task_count=100,
        rows=100,
        columns=1,
        cost_per_item=1.0,
        capability="thread_safe",
    )

    with pytest.raises(ParallelExecutionError, match="任务0"):
        parallel_execute(
            _fail_first_and_record_started,
            tasks(),
            n_jobs=2,
            parallel_backend="threading",
            parallel_config={"batch_size": 1, "pre_dispatch": 2},
            task_labels=[f"任务{index}" for index in range(100)],
            workload=workload,
        )

    assert len(yielded) < 100


def test_timeout_does_not_parallelize_serial_only_workload():
    """为单 worker 强制 timeout 时也不能偷偷并发执行 serial_only 任务。"""
    lock = threading.Lock()
    state = {"active": 0, "maximum": 0}

    def record_concurrency(value):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        sleep(0.01)
        with lock:
            state["active"] -= 1
        return value

    result = parallel_execute(
        record_concurrency,
        range(6),
        n_jobs=4,
        parallel_backend="threading",
        parallel_config={"timeout": 2},
        workload=ParallelWorkload(
            task_count=6,
            capability="serial_only",
            operation="状态型串行任务",
        ),
    )

    assert result == list(range(6))
    assert state["maximum"] == 1


@pytest.mark.parametrize(
    ("parallel_backend", "default_backend"),
    [("threading", None), (None, "threading")],
)
def test_threading_backend_rejects_inner_thread_limit_in_chinese(parallel_backend, default_backend):
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
def test_loky_and_implicit_default_accept_inner_thread_limit(parallel_backend, default_backend):
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
def test_invalid_backend_configuration_raises_chinese_validation_error(parallel_backend, default_backend, parallel_config):
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
def test_split_parallel_budget_uses_square_root_only_for_real_nesting(available, task_count, has_parallel_children, expected):
    """只有真实同时嵌套的任务才按平方根切分预算。"""
    assert split_parallel_budget(available, task_count, has_parallel_children) == expected


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
def test_split_parallel_budget_rejects_invalid_inputs_in_chinese(available, task_count, has_parallel_children, field):
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


def test_generic_small_automatic_tasks_stay_serial_without_workload_metadata(monkeypatch):
    """旧调用未传 workload 时也不能把轻量标量任务误判成重进程任务。"""
    monkeypatch.setattr("hscredit.utils.parallel.get_physical_cpu_count", lambda: 4)

    thread_ids = parallel_execute(
        _sleep_thread_identity,
        range(20),
        n_jobs=-1,
        default_backend="threading",
    )

    assert len(set(thread_ids)) == 1


def test_tiny_automatic_workload_uses_serial_plan():
    """小任务的自动预算不得为了占满 CPU 创建高开销 worker。"""
    workload = ParallelWorkload(
        task_count=32,
        rows=20,
        columns=32,
        data_bytes=8_192,
        cost_per_item=1.0,
        capability="thread_safe",
        operation="轻量字段统计",
    )

    plan = plan_parallel_execution(-1, workload, cpu_count=16)

    assert plan == ParallelExecutionPlan(
        requested_workers=13,
        workers=1,
        backend="threading",
        adaptive=True,
        estimated_work=640.0,
        data_bytes=8_192,
        child_budget=13,
        operation="轻量字段统计",
    )


def test_large_automatic_workload_uses_multiple_workers():
    """计算量足够大时自动计划必须真正使用多核。"""
    workload = ParallelWorkload(
        task_count=64,
        rows=50_000,
        columns=64,
        data_bytes=25_600_000,
        cost_per_item=4.0,
        capability="process_safe",
        operation="互信息",
    )

    plan = plan_parallel_execution(-1, workload, cpu_count=16)

    assert 1 < plan.workers <= 13
    assert plan.backend == "loky"
    assert plan.requested_workers == 13
    assert plan.estimated_work == 12_800_000.0


def test_explicit_n_jobs_is_a_hard_upper_bound_for_adaptive_plan():
    """用户正整数预算优先，自动策略只能向下收缩，不能越界。"""
    workload = ParallelWorkload(
        task_count=100,
        rows=1_000_000,
        columns=100,
        cost_per_item=10.0,
        capability="process_safe",
    )

    plan = plan_parallel_execution(3, workload, cpu_count=16)

    assert plan.requested_workers == 3
    assert plan.workers == 3


def test_explicit_backend_preserves_requested_workers_for_tiny_work():
    """显式后端表达强制执行意图，不得被自动成本阈值改成串行。"""
    workload = ParallelWorkload(
        task_count=8,
        rows=2,
        columns=8,
        cost_per_item=1.0,
        capability="thread_safe",
    )

    plan = plan_parallel_execution(
        4,
        workload,
        parallel_backend="threading",
        cpu_count=16,
    )

    assert plan.workers == 4
    assert plan.backend == "threading"
    assert plan.adaptive is False


def test_explicit_n_jobs_preserves_requested_workers_for_tiny_work_without_backend():
    """用户显式 worker 数必须生效，不能被自动成本阈值缩减。"""
    workload = ParallelWorkload(
        task_count=8,
        rows=2,
        columns=8,
        cost_per_item=1.0,
        capability="thread_safe",
    )

    plan = plan_parallel_execution(4, workload, cpu_count=16)

    assert plan.workers == 4
    assert plan.backend == "threading"
    assert plan.adaptive is False


def test_automatic_worker_cap_does_not_override_explicit_n_jobs():
    """workload 自动上限只约束 -1 自动预算，不改写显式 n_jobs。"""
    workload = ParallelWorkload(
        task_count=32,
        rows=100_000,
        columns=32,
        cost_per_item=8.0,
        capability="thread_safe",
        releases_gil=True,
        auto_max_workers=4,
    )

    automatic = plan_parallel_execution(-1, workload, cpu_count=16)
    explicit = plan_parallel_execution(8, workload, cpu_count=16)

    assert automatic.workers == 4
    assert explicit.workers == 8
    assert explicit.adaptive is False


def test_large_process_payload_caps_automatic_workers():
    """自动 loky 计划必须限制大型序列化载荷的并发副本数。"""
    workload = ParallelWorkload(
        task_count=16,
        rows=1_000_000,
        columns=16,
        data_bytes=300 * 1024**2,
        cost_per_item=10.0,
        capability="process_safe",
        releases_gil=False,
    )

    plan = plan_parallel_execution(-1, workload, cpu_count=16)

    assert plan.backend == "loky"
    assert plan.workers == 1


def test_releases_gil_prefers_threads_only_without_backend_choice():
    """释放 GIL 的进程安全任务仅在无 backend 选择时自动使用线程。"""
    workload = ParallelWorkload(
        task_count=8,
        rows=100_000,
        columns=8,
        cost_per_item=8.0,
        capability="process_safe",
        releases_gil=True,
    )

    automatic = plan_parallel_execution(-1, workload, cpu_count=8)
    loky_default = plan_parallel_execution(-1, workload, default_backend="loky", cpu_count=8)

    assert automatic.backend == "threading"
    assert loky_default.backend == "loky"


def test_single_outer_task_preserves_total_budget_for_parallel_children():
    """任务数只限制当前层 worker，不能吞掉唯一任务可下放的总预算。"""
    workload = ParallelWorkload(
        task_count=1,
        rows=100_000,
        columns=1,
        cost_per_item=10.0,
        capability="process_safe",
        has_parallel_children=True,
    )

    plan = plan_parallel_execution(
        -1,
        workload,
        parallel_backend="threading",
        cpu_count=5,
    )

    assert plan.requested_workers == 4
    assert plan.workers == 1
    assert plan.child_budget == 4


def test_adaptive_false_preserves_requested_workers_without_explicit_backend():
    """关闭自适应后仍使用能力推荐后端，但 worker 数遵循用户预算。"""
    workload = ParallelWorkload(
        task_count=8,
        rows=2,
        columns=8,
        cost_per_item=1.0,
        capability="process_safe",
    )

    plan = plan_parallel_execution(
        4,
        workload,
        parallel_config={"adaptive": False},
        cpu_count=16,
    )

    assert plan.workers == 4
    assert plan.backend == "loky"
    assert plan.adaptive is False


def test_serial_only_capability_wins_over_forced_parallel_configuration():
    """无法安全并行的状态型任务不能因显式后端破坏正确性。"""
    workload = ParallelWorkload(
        task_count=8,
        rows=100_000,
        columns=8,
        cost_per_item=10.0,
        capability="serial_only",
    )

    plan = plan_parallel_execution(
        4,
        workload,
        parallel_backend="threading",
        cpu_count=16,
    )

    assert plan.workers == 1
    assert plan.child_budget == 4


def test_native_worker_aliases_share_active_budget_and_public_n_jobs_priority():
    """CP-SAT/CatBoost 等原生线程池不得绕过当前总预算。"""
    token = _ACTIVE_BUDGET.set(ParallelBudget(4, 0))
    try:
        assert resolve_native_workers(-1) == 4
        assert resolve_native_workers(-1, native_workers=2) == 2
        assert resolve_native_workers(3, native_workers=99) == 3
        assert resolve_native_workers(1, native_workers=99) == 1
    finally:
        _ACTIVE_BUDGET.reset(token)


def test_single_worker_timeout_is_enforced_instead_of_serial_shortcut():
    """显式 timeout 不能因有效 worker 为 1 而被静默忽略。"""
    with pytest.raises(ParallelExecutionError, match="并行任务执行失败") as exc_info:
        parallel_execute(
            sleep,
            [0.2],
            n_jobs=1,
            parallel_backend="loky",
            parallel_config={"timeout": 0.01},
        )

    assert "Timeout" in type(exc_info.value.__cause__).__name__
