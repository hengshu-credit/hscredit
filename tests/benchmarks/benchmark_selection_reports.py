"""独立进程的IV核对照与百万行筛选报告容量基准；输出仅合成数据汇总JSON。"""

import argparse
import gc
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from hscredit.core.selectors import IVSelector, NullSelector, VarianceSelector, collect_selection_report
from hscredit.core.selectors.iv_selector import _compute_iv_single


def measure(function):
    gc.collect()
    process = psutil.Process()
    initial = process.memory_info().rss
    peak = [initial]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample)
    thread.start()
    started = perf_counter()
    try:
        result = function()
    finally:
        elapsed = perf_counter() - started
        peak[0] = max(peak[0], process.memory_info().rss)
        stopped.set()
        thread.join()
    return result, {
        "耗时秒": elapsed,
        "起始RSS字节": initial,
        "抽样峰值RSS字节": peak[0],
        "RSS增量字节": peak[0] - initial,
    }


def legacy_iv(x, y):
    values = np.unique(x)
    events = np.array([np.sum(y[x == value] == 1) for value in values], dtype=float)
    nonevents = np.array([np.sum(y[x == value] == 0) for value in values], dtype=float)
    a = (events + 1) / (events.sum() + len(values))
    b = (nonevents + 1) / (nonevents.sum() + len(values))
    return float(np.sum((a - b) * np.log(a / b)))


def worker(case, rows, columns):
    rng = np.random.default_rng(42)
    with threadpool_limits(limits=1):
        if case in {"iv_old", "iv_new"}:
            x, y = np.arange(5000, dtype=float), np.arange(5000) % 2
            function = (lambda: legacy_iv(x, y)) if case == "iv_old" else (lambda: _compute_iv_single(x, y))
            function()
            samples = [measure(function) for _ in range(5)]
            return {
                "行数": len(x),
                "唯一值数": len(x),
                "结果": samples[-1][0],
                "中位耗时秒": median(item[1]["耗时秒"] for item in samples),
                "测量": [item[1] for item in samples],
            }
        if psutil.virtual_memory().available < rows * columns * 16 + 512 * 1024 * 1024:
            raise RuntimeError("可用内存不足以安全运行容量基准，请减小 --rows/--columns")
        frame = pd.DataFrame(
            rng.standard_normal((rows, columns), dtype=np.float32), columns=[f"字段{i}" for i in range(columns)]
        )
        y = rng.integers(0, 2, size=rows, dtype=np.int8)
        pipeline = Pipeline(
            [("缺失", NullSelector(n_jobs=1)), ("方差", VarianceSelector(n_jobs=1)), ("IV", IVSelector(n_jobs=1))]
        )
        _, fit = measure(lambda: pipeline.fit(frame, y))
        report, collect = measure(lambda: collect_selection_report(pipeline))
        with tempfile.TemporaryDirectory(prefix="hscredit-selector-benchmark-") as temporary:
            receipt, save = measure(lambda: report.save(Path(temporary) / "容量报告", formats=("json",)))
            byte_count = Path(receipt["路径"]["json"]).stat().st_size
        return {
            "行数": rows,
            "字段数": columns,
            "数据字节": int(frame.memory_usage(deep=True).sum()),
            "拟合": fit,
            "收集报告": collect,
            "保存JSON": save,
            "报告文件字节": byte_count,
            "阶段行数": len(report.summary),
            "明细行数": len(report.details),
            "指标行数": len(report.metrics),
            "最终字段数": len(pipeline[-1].selected_features_),
            "报告完整": report.metadata["完整"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["iv_old", "iv_new", "workflow"])
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--columns", type=int, default=32)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(worker(args.case, args.rows, args.columns), ensure_ascii=True, allow_nan=False))
    else:
        output = {
            "环境": {
                "Python": platform.python_version(),
                "NumPy": np.__version__,
                "pandas": pd.__version__,
                "BLAS线程": 1,
            },
            "范围": "IV核1次预热后测量5次；Null+Variance+IV百万行流程仅当前实现单次容量检查；RSS每5ms抽样，不是生产容量保证",
        }
        for case in ["iv_old", "iv_new", "workflow"]:
            command = [
                sys.executable,
                __file__,
                "--case",
                case,
                "--rows",
                str(args.rows),
                "--columns",
                str(args.columns),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
            output[case] = json.loads(completed.stdout)
        np.testing.assert_allclose(output["iv_old"]["结果"], output["iv_new"]["结果"], rtol=1e-12, atol=1e-12)
        output["IV核结果等价"] = True
        output["IV核中位加速比"] = output["iv_old"]["中位耗时秒"] / output["iv_new"]["中位耗时秒"]
        print(json.dumps(output, ensure_ascii=True, allow_nan=False, indent=2))
