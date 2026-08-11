"""CorrSelector 精确相关并行基准。"""

import argparse
from contextlib import nullcontext
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hscredit.core.selectors.corr_selector as corr_module  # noqa: E402
from hscredit.core.selectors import CorrSelector  # noqa: E402
from hscredit.utils.parallel import get_physical_cpu_count  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description="CorrSelector 超宽表精确并行基准")
    parser.add_argument("--rows", type=int, default=6194, help="样本行数")
    parser.add_argument("--features", type=int, default=67793, help="特征数")
    parser.add_argument(
        "--correlated-ratio",
        type=float,
        default=0.5,
        help="成对构造高相关特征的比例，范围 0 到 1",
    )
    parser.add_argument("--n-jobs", type=int, default=12, help="并行总预算")
    parser.add_argument(
        "--backend",
        choices=("auto", "threading", "loky", "multiprocessing"),
        default="auto",
        help="joblib 后端；auto 使用 CorrSelector 默认共享内存线程",
    )
    parser.add_argument("--corr-block-size", type=int, default=512, help="相关矩阵块上限")
    parser.add_argument(
        "--method",
        choices=("pearson", "spearman", "kendall"),
        default="spearman",
        help="相关性方法",
    )
    parser.add_argument(
        "--compare-serial",
        action="store_true",
        help="额外执行 n_jobs=1 串行对照；实际超宽数据可能耗时很长",
    )
    return parser.parse_args()


def _validate_args(args):
    if args.rows < 2:
        raise ValueError("rows 必须至少为 2")
    if args.features < 2:
        raise ValueError("features 必须至少为 2")
    if not 0 <= args.correlated_ratio <= 1:
        raise ValueError("correlated-ratio 必须在 0 到 1 之间")
    if args.n_jobs < 1:
        raise ValueError("n-jobs 必须为正整数")
    if args.corr_block_size < 1:
        raise ValueError("corr-block-size 必须为正整数")


def _build_frame(rows, features, correlated_ratio):
    rng = np.random.RandomState(20260807)
    values = rng.normal(size=(rows, features))
    correlated_features = min(features, int(features * correlated_ratio))
    correlated_features -= correlated_features % 2
    for index in range(1, correlated_features, 2):
        values[:, index] = values[:, index - 1] * 0.995 + rng.normal(
            scale=0.005,
            size=rows,
        )
    columns = [f"特征{index}" for index in range(features)]
    frame = pd.DataFrame(values, columns=columns)
    weights = {column: float(features - index) for index, column in enumerate(columns)}
    return frame, weights


def _run_once(frame, weights, *, n_jobs, backend, method, block_size, identities):
    original_rank_worker = corr_module._rank_corr_column
    original_corr_worker = corr_module._corr_block_worker
    original_parallel_execute = CorrSelector._parallel_execute
    original_threadpool_limits = corr_module.threadpool_limits
    phase_seconds = {"排名": 0.0, "相关块": 0.0}
    native_limits = []
    if backend in ("loky", "multiprocessing"):

        def record_identity():
            identity = (os.getpid(), threading.get_ident())
            identities[str(identity)] = True

    else:
        local_identities = set()
        identity_lock = threading.Lock()

        def record_identity():
            identity = (os.getpid(), threading.get_ident())
            with identity_lock:
                if identity in local_identities:
                    return
                local_identities.add(identity)
            identities[str(identity)] = True

    def rank_worker(task):
        record_identity()
        return original_rank_worker(task)

    def corr_worker(task):
        record_identity()
        return original_corr_worker(task)

    def timed_parallel_execute(self, function, tasks, **kwargs):
        started = perf_counter()
        try:
            return original_parallel_execute(self, function, tasks, **kwargs)
        finally:
            elapsed = perf_counter() - started
            if function is rank_worker:
                phase_seconds["排名"] += elapsed
            elif function is corr_worker:
                phase_seconds["相关块"] += elapsed

    def recording_threadpool_limits(*, limits):
        native_limits.append(int(limits))
        return original_threadpool_limits(limits=limits)

    corr_module._rank_corr_column = rank_worker
    corr_module._corr_block_worker = corr_worker
    CorrSelector._parallel_execute = timed_parallel_execute
    corr_module.threadpool_limits = recording_threadpool_limits
    started = perf_counter()
    try:
        selector = CorrSelector(
            threshold=0.7,
            method=method,
            weights=weights,
            binning_params=None,
            n_jobs=n_jobs,
            parallel_backend=backend,
            corr_block_size=block_size,
        ).fit(frame)
    finally:
        elapsed = perf_counter() - started
        corr_module._rank_corr_column = original_rank_worker
        corr_module._corr_block_worker = original_corr_worker
        CorrSelector._parallel_execute = original_parallel_execute
        corr_module.threadpool_limits = original_threadpool_limits

    other_seconds = max(0.0, elapsed - sum(phase_seconds.values()))
    return selector, {
        "总耗时秒": elapsed,
        "排名任务墙钟秒": phase_seconds["排名"],
        "相关块任务墙钟秒": phase_seconds["相关块"],
        "准备及报告秒": other_seconds,
        "原生线程限制序列": native_limits,
    }


def main():
    args = _parse_args()
    _validate_args(args)
    backend = None if args.backend == "auto" else args.backend
    physical_cpus = get_physical_cpu_count()
    pair_count = args.features * (args.features - 1) // 2
    dense_gib = args.features * args.features * 8 / (1024**3)
    input_gib = args.rows * args.features * 8 / (1024**3)
    print(
        json.dumps(
            {
                "数据形状": [args.rows, args.features],
                "唯一特征对": pair_count,
                "输入float64约GiB": round(input_gib, 3),
                "完整相关矩阵约GiB": round(dense_gib, 3),
                "相关矩阵块数": math.ceil(args.features / args.corr_block_size),
                "后端": args.backend,
                "n_jobs总预算": args.n_jobs,
                "物理CPU数": physical_cpus,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    frame, weights = _build_frame(args.rows, args.features, args.correlated_ratio)
    process_backend = backend in ("loky", "multiprocessing")
    manager_context = multiprocessing.Manager() if process_backend else nullcontext(None)
    with manager_context as manager:
        parallel_identities = manager.dict() if manager is not None else {}
        parallel, parallel_metrics = _run_once(
            frame,
            weights,
            n_jobs=args.n_jobs,
            backend=backend,
            method=args.method,
            block_size=args.corr_block_size,
            identities=parallel_identities,
        )
        parallel_metrics["worker身份"] = sorted(parallel_identities.keys())

        result = {
            "并行": parallel_metrics,
            "选中特征数": len(parallel.selected_features_),
            "剔除特征数": len(parallel.dropped_),
        }
        if args.compare_serial:
            serial_identities = manager.dict() if manager is not None else {}
            serial, serial_metrics = _run_once(
                frame,
                weights,
                n_jobs=1,
                backend=backend,
                method=args.method,
                block_size=args.corr_block_size,
                identities=serial_identities,
            )
            serial_metrics["worker身份"] = sorted(serial_identities.keys())
            result["串行"] = serial_metrics
            result["加速比"] = serial_metrics["总耗时秒"] / parallel_metrics["总耗时秒"]
            result["结果一致"] = (
                serial.selected_features_ == parallel.selected_features_
                and serial.dropped_.equals(parallel.dropped_)
            )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
