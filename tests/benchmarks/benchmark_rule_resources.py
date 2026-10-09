"""规则热点的可复现实验；仅合成数据，报告时间及进程 RSS/分配峰值。

运行: python tests/benchmarks/benchmark_rule_resources.py
这是聚合/任务载荷内核基准，不将结果外推为整条建模流程加速。
"""

import copy
import gc
import json
import os
import platform
import threading
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil


def measure(function, repeats=3):
    observations = []
    process = psutil.Process()
    for _ in range(repeats):
        gc.collect()
        baseline = process.memory_info().rss
        peak = [baseline]
        stopped = threading.Event()

        def sample():
            while not stopped.wait(0.002):
                rss = process.memory_info().rss
                rss += sum(child.memory_info().rss for child in process.children(recursive=True) if child.is_running())
                peak[0] = max(peak[0], rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        tracemalloc.start()
        start = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - start
        _, allocated_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak[0] = max(peak[0], process.memory_info().rss)
        stopped.set()
        sampler.join()
        observations.append(
            {
                "seconds": elapsed,
                "rss_peak_bytes": peak[0],
                "rss_increase_bytes": peak[0] - baseline,
                "allocation_peak_bytes": allocated_peak,
            }
        )
        del value
    return observations


def main():
    rng = np.random.RandomState(20261008)
    rows, bins = 1000000, 25
    x = rng.randint(0, bins, rows).astype(np.int16)
    z = rng.randint(0, bins, rows).astype(np.int16)
    y = rng.randint(0, 2, rows).astype(np.int8)

    def scan_cells():
        counts = np.zeros((bins, bins), dtype=np.int64)
        bad = np.zeros_like(counts)
        for i in range(bins):
            for j in range(bins):
                mask = (x == i) & (z == j)
                counts[i, j] = mask.sum()
                bad[i, j] = y[mask].sum()
        return counts, bad

    def joint_counts():
        joint = x.astype(np.int64) * bins + z
        return (
            np.bincount(joint, minlength=bins * bins).reshape(bins, bins),
            np.bincount(joint, weights=y, minlength=bins * bins).reshape(bins, bins),
        )

    expected, actual = scan_cells(), joint_counts()  # 预热兼正确性对照。
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    source = SimpleNamespace(X_=pd.DataFrame(rng.normal(size=(100000, 20))), y_=np.zeros(100000), cross_results_={})

    def old_tasks():
        return [(copy.deepcopy(source), i) for i in range(10)]

    def shared_tasks():
        return [(source, i) for i in range(10)]

    output = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "seed": 20261008,
            "repeats": 3,
            "workers": 1,
            "disk_output_bytes": 0,
        },
        "grid": {
            "rows": rows,
            "shape": [bins, bins],
            "dtype": "int16/int16/int8",
            "equal": True,
            "before": measure(scan_cells),
            "after": measure(joint_counts),
        },
        "task_payloads": {
            "rows": 100000,
            "columns": 20,
            "tasks": 10,
            "dtype": "float64",
            "before": measure(old_tasks),
            "after": measure(shared_tasks),
        },
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
