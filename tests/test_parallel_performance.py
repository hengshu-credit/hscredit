"""统一并行执行的可复现慢速性能验收。"""

import os
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from hscredit.core.binning import QuantileBinning
from hscredit.core.rules import Rule, RuleFlow
from hscredit.core.selectors import MutualInfoSelector
from hscredit.utils.parallel import get_physical_cpu_count, parallel_execute


pytestmark = pytest.mark.slow
REPEATS = 3
PARALLEL_WORKERS = 4


def _worker_process_identity(task):
    """用真实 CPU 循环让低核环境也能观察到显式进程 worker。"""
    checksum = 0
    for value in range(300_000):
        checksum = (checksum + value * (task + 1)) % 1_000_003
    return os.getpid(), checksum


@pytest.fixture(scope="module")
def benchmark_data():
    """确定性的宽数值/类别、多规则和三标签性能数据。"""
    rng = np.random.RandomState(20260805)
    rows = 12_000
    numeric_count = 48
    categorical_count = 8
    raw = rng.normal(size=(rows, numeric_count))
    data = pd.DataFrame(raw, columns=[f"n{index}" for index in range(numeric_count)])
    categories = np.asarray(["A", "B", "C", "D"], dtype=object)
    for index in range(categorical_count):
        data[f"c{index}"] = categories[rng.randint(0, len(categories), rows)]

    logits = raw[:, :6].sum(axis=1) + rng.normal(scale=1.5, size=rows)
    data["target"] = (logits > 0).astype(np.int8)
    for index, threshold in enumerate((0.2, 0.0, -0.2)):
        data[f"label{index}"] = (logits > threshold).astype(np.int8)
    data["amount"] = rng.uniform(100, 10_000, rows)
    return data


@pytest.fixture(autouse=True)
def limit_third_party_thread_pools():
    """避免 BLAS/OpenMP 内层线程污染 HSCredit 外层并行基准。"""
    with threadpool_limits(limits=1):
        yield


def _measure_medians(name, serial_call, parallel_call, metadata):
    """各预热一次，再交错测量三次并返回可诊断中位数。"""
    serial_warm = serial_call()
    parallel_warm = parallel_call()

    serial_times = []
    parallel_times = []
    for run in range(REPEATS):
        calls = (
            (("serial", serial_call), ("parallel", parallel_call))
            if run % 2 == 0
            else (("parallel", parallel_call), ("serial", serial_call))
        )
        for mode, call in calls:
            started = perf_counter()
            call()
            elapsed = perf_counter() - started
            (serial_times if mode == "serial" else parallel_times).append(elapsed)

    serial_median = median(serial_times)
    parallel_median = median(parallel_times)
    speedup = serial_median / parallel_median
    diagnostic = (
        f"BENCHMARK {name} {metadata} physical_cpus={get_physical_cpu_count()} "
        f"serial_runs={serial_times} parallel_runs={parallel_times} "
        f"serial_median={serial_median:.6f}s parallel_median={parallel_median:.6f}s "
        f"speedup={speedup:.3f}x"
    )
    print(diagnostic)
    return serial_warm, parallel_warm, serial_median, parallel_median, speedup, diagnostic


def _assert_speed_gate(serial_median, parallel_median, speedup, diagnostic, require_speedup=False):
    """四核以上执行速度门槛；低核仅跳过门槛而不跳过工作流和一致性。"""
    if get_physical_cpu_count() < 4:
        return
    assert parallel_median <= serial_median * 1.05, diagnostic
    if require_speedup:
        assert speedup >= 1.20, diagnostic


def test_explicit_parallel_workers_participate_even_when_speed_gate_is_unavailable():
    """速度门槛以外仍验证共享执行器确实使用多个进程。"""
    results = parallel_execute(
        _worker_process_identity,
        range(12),
        n_jobs=2,
        parallel_backend="loky",
        parallel_config={"batch_size": 1, "inner_max_num_threads": 1},
    )
    assert len({process_id for process_id, _ in results}) >= 2


def test_cpu_heavy_wide_selector_reaches_speedup_gate(benchmark_data):
    """宽表互信息是主要 CPU-heavy 加速门槛，不测试裸 joblib。"""
    features = [f"n{index}" for index in range(48)]
    X = benchmark_data[features]
    y = benchmark_data["target"]
    config = {"batch_size": 1, "inner_max_num_threads": 1}

    def serial_call():
        return MutualInfoSelector(
            random_state=20260805, n_jobs=1, parallel_config=config
        ).fit(X, y)

    def parallel_call():
        return MutualInfoSelector(
            random_state=20260805,
            n_jobs=PARALLEL_WORKERS,
            parallel_backend="loky",
            parallel_config=config,
        ).fit(X, y)

    serial, parallel, serial_median, parallel_median, speedup, diagnostic = _measure_medians(
        "mutual_info_wide",
        serial_call,
        parallel_call,
        "rows=12000 numeric_features=48 backend=loky workers=4",
    )
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_, check_exact=True)
    assert serial.selected_features_ == parallel.selected_features_
    _assert_speed_gate(
        serial_median, parallel_median, speedup, diagnostic, require_speedup=True
    )


def test_wide_numeric_categorical_binning_parallel_overhead_gate(benchmark_data):
    """宽数值+类别分箱不得因线程调度慢于串行超过 5%。"""
    features = [f"n{index}" for index in range(48)] + [f"c{index}" for index in range(8)]
    X = benchmark_data[features]
    y = benchmark_data["target"]
    config = {"batch_size": 1}

    def serial_call():
        return QuantileBinning(
            max_n_bins=8, random_state=20260805, n_jobs=1, parallel_config=config
        ).fit(X, y)

    def parallel_call():
        return QuantileBinning(
            max_n_bins=8,
            random_state=20260805,
            n_jobs=PARALLEL_WORKERS,
            parallel_backend="threading",
            parallel_config=config,
        ).fit(X, y)

    serial, parallel, serial_median, parallel_median, speedup, diagnostic = _measure_medians(
        "quantile_binning_wide",
        serial_call,
        parallel_call,
        "rows=12000 numeric_features=48 categorical_features=8 backend=threading workers=4",
    )
    pd.testing.assert_frame_equal(serial.transform(X), parallel.transform(X), check_exact=True)
    for feature in features:
        pd.testing.assert_frame_equal(
            serial.get_bin_table(feature), parallel.get_bin_table(feature), check_exact=True
        )
    _assert_speed_gate(serial_median, parallel_median, speedup, diagnostic)


def _benchmark_rules():
    return [
        Rule(f"n{index % 48} > {(index % 7 - 3) / 4}", name=f"规则{index}")
        for index in range(72)
    ]


def test_many_rule_flow_parallel_overhead_gate(benchmark_data):
    """72 条真实规则的有序命中输出不得因并行慢于串行超过 5%。"""
    config = {"batch_size": 1}

    def serial_call():
        return RuleFlow(
            _benchmark_rules(), mode="parallel", n_jobs=1, parallel_config=config
        ).predict(benchmark_data)

    def parallel_call():
        return RuleFlow(
            _benchmark_rules(),
            mode="parallel",
            n_jobs=PARALLEL_WORKERS,
            parallel_backend="threading",
            parallel_config=config,
        ).predict(benchmark_data)

    serial, parallel, serial_median, parallel_median, speedup, diagnostic = _measure_medians(
        "rule_flow_many_rules",
        serial_call,
        parallel_call,
        "rows=12000 rules=72 backend=threading workers=4",
    )
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    _assert_speed_gate(serial_median, parallel_median, speedup, diagnostic)


def test_three_label_rule_report_parallel_overhead_gate(benchmark_data):
    """三个独立标签口径并行报告必须精确，且不得慢于串行超过 5%。"""
    label_data = pd.concat([benchmark_data] * 8, ignore_index=True)
    config = {"batch_size": 1}

    def serial_call():
        return Rule("n0 > 0", n_jobs=1, parallel_config=config).report(
            label_data,
            overdue=["label0", "label1", "label2"],
            dpds=[0],
            amount="amount",
        )

    def parallel_call():
        return Rule(
            "n0 > 0",
            n_jobs=3,
            parallel_backend="threading",
            parallel_config=config,
        ).report(
            label_data,
            overdue=["label0", "label1", "label2"],
            dpds=[0],
            amount="amount",
        )

    serial, parallel, serial_median, parallel_median, speedup, diagnostic = _measure_medians(
        "rule_report_three_labels",
        serial_call,
        parallel_call,
        "rows=96000 labels=3 backend=threading workers=3",
    )
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    _assert_speed_gate(serial_median, parallel_median, speedup, diagnostic)
