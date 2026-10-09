"""并行资源预检、慢消费者背压和有界任务窗口。"""

import numpy as np
import pytest

from hscredit.exceptions import ValidationError
from hscredit.utils.parallel import ParallelWorkload, parallel_execute, plan_parallel_execution


def _square(value):
    return value * value


@pytest.mark.parametrize(
    "name,value",
    [("memory_budget_bytes", 0), ("max_inflight_tasks", True), ("max_output_rows", -1), ("result_retention", "full")],
)
def test_invalid_resource_configuration(name, value):
    with pytest.raises(ValidationError):
        parallel_execute(_square, [1], n_jobs=1, parallel_config={name: value})


def test_explicit_worker_memory_budget_is_checked_before_running():
    workload = ParallelWorkload(10, data_bytes=100, working_bytes_per_task=40, result_bytes_per_task=10)
    with pytest.raises(ValidationError, match="预计峰值内存"):
        plan_parallel_execution(4, workload, parallel_backend="threading", parallel_config={"memory_budget_bytes": 300})
    plan = plan_parallel_execution(
        4, workload, parallel_backend="threading", parallel_config={"memory_budget_bytes": 400}
    )
    assert plan.estimated_peak_bytes == 360


def test_automatic_worker_plan_can_reduce_memory_usage():
    workload = ParallelWorkload(
        10,
        rows=1000000,
        columns=10,
        data_bytes=100,
        working_bytes_per_task=40,
        result_bytes_per_task=10,
        capability="thread_safe",
    )
    plan = plan_parallel_execution(-1, workload, cpu_count=8, parallel_config={"memory_budget_bytes": 240})
    assert plan.workers == 1
    assert plan.estimated_peak_bytes <= 240


@pytest.mark.parametrize("workers,backend", [(1, None), (2, "threading"), (2, "loky")])
def test_iterator_keeps_input_consumption_bounded(workers, backend):
    consumed = []
    closed = []

    def inputs():
        try:
            for item in range(10):
                consumed.append(item)
                yield item
        finally:
            closed.append(True)

    results = parallel_execute(
        _square,
        inputs(),
        n_jobs=workers,
        parallel_backend=backend,
        workload=ParallelWorkload(10),
        parallel_config={"max_inflight_tasks": 2, "result_retention": "iterator"},
    )
    assert consumed == []
    assert next(results) == 0
    assert consumed == [0, 1]
    assert next(results) == 1
    assert consumed == [0, 1]
    assert next(results) == 4
    assert consumed == [0, 1, 2, 3]
    results.close()
    assert consumed == [0, 1, 2, 3]
    assert closed == [True]


def test_bounded_list_keeps_default_order_and_reports_iterator_length_errors():
    assert parallel_execute(
        _square, range(9), n_jobs=2, parallel_backend="threading", parallel_config={"max_inflight_tasks": 3}
    ) == [i * i for i in range(9)]
    result = parallel_execute(
        _square,
        iter([1, 2]),
        n_jobs=1,
        workload=ParallelWorkload(3),
        parallel_config={"result_retention": "iterator", "max_inflight_tasks": 1},
    )
    assert next(result) == 1
    assert next(result) == 4
    with pytest.raises(ValidationError, match="数量一致"):
        next(result)


def test_unknown_size_iterators_require_workload():
    with pytest.raises(ValidationError, match="workload"):
        parallel_execute(_square, iter([1, 2]), parallel_config={"result_retention": "iterator"})


def test_output_limit_checked_in_plan_and_during_consumption():
    with pytest.raises(ValidationError, match="预计输出行数"):
        plan_parallel_execution(
            1, ParallelWorkload(3, output_rows_per_task=10), parallel_config={"max_output_rows": 20}
        )
    result = parallel_execute(
        lambda n: np.ones(n),
        [2, 3],
        n_jobs=1,
        parallel_config={"max_output_rows": 4, "max_inflight_tasks": 1, "result_retention": "iterator"},
    )
    assert len(next(result)) == 2
    with pytest.raises(ValidationError, match="实际输出行数"):
        next(result)


def test_iterator_estimate_does_not_include_every_result():
    workload = ParallelWorkload(100, data_bytes=10, working_bytes_per_task=1, result_bytes_per_task=100)
    plan = plan_parallel_execution(
        2,
        workload,
        parallel_backend="threading",
        parallel_config={"result_retention": "iterator", "max_inflight_tasks": 2, "memory_budget_bytes": 300},
    )
    assert plan.estimated_peak_bytes == 212


def test_nonstreaming_components_reject_single_use_result_iterators():
    from hscredit.utils.parallel import ParallelizableMixin

    class Component(ParallelizableMixin):
        n_jobs = 1
        parallel_backend = None
        parallel_config = {"result_retention": "iterator"}

    with pytest.raises(ValidationError, match="完整批量结果"):
        Component()._parallel_execute(_square, [1, 2])
    assert list(Component()._parallel_execute(_square, [1, 2], stream_results=True)) == [1, 4]


def test_adaptive_memory_plan_recomputes_implicit_result_window():
    workload = ParallelWorkload(100, rows=10000000, capability="thread_safe", result_bytes_per_task=100)
    plan = plan_parallel_execution(
        -1, workload, cpu_count=8, parallel_config={"result_retention": "iterator", "memory_budget_bytes": 500}
    )
    assert plan.workers == 2
    assert plan.estimated_peak_bytes == 400


def test_bounded_execution_preserves_nested_worker_budget():
    from hscredit.utils.parallel import _ACTIVE_BUDGET

    def budget(_):
        return _ACTIVE_BUDGET.get().available

    workload = ParallelWorkload(4, has_parallel_children=True)
    expected = parallel_execute(budget, range(4), n_jobs=4, parallel_backend="threading", workload=workload)
    actual = list(
        parallel_execute(
            budget,
            range(4),
            n_jobs=4,
            parallel_backend="threading",
            workload=workload,
            parallel_config={"result_retention": "iterator"},
        )
    )
    assert expected == actual == [2, 2, 2, 2]
