"""独立进程、三次重复的MDLP候选扫描/VIF全列评分资源差分；非容量认证。"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import json
import platform
import subprocess
import threading
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from hscredit.core.binning import MDLPBinning
from hscredit.core.selectors import VIFSelector
from hscredit.core.selectors.vif_selector import _compute_vif_single
from tests.test_binning.test_mdlp_prefix_counts import LegacyScanMDLP


def measure(function):
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()
    def sample():
        while not stopped.wait(0.002):
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
    return result, {"seconds": elapsed, "rss_before": baseline, "sampled_peak_rss": peak[0], "sampled_rss_increment": peak[0] - baseline}


def worker(case, options):
    rng = np.random.default_rng(42)
    if case.startswith("mdlp"):
        x = np.sort(rng.normal(size=options.rows_mdlp))
        y = rng.integers(0, 2, len(x))
        binner = (LegacyScanMDLP if case.endswith("legacy") else MDLPBinning)(max_n_bins=5, n_jobs=1)
        candidates = binner._find_all_candidates_v3(x, y)
        operation = lambda: binner._force_additional_splits_v3(x, y, [], candidates)
        metadata = {"rows": len(x), "candidate_count": len(candidates), "max_bins": 5, "input_bytes": x.nbytes + y.nbytes}
    else:
        values = rng.normal(size=(options.rows_vif, options.features))
        frame = pd.DataFrame(values)
        operation = (lambda: [_compute_vif_single(values, index) for index in range(values.shape[1])]) if case.endswith("legacy") else (lambda: VIFSelector(n_jobs=1)._compute_vif_all(frame).tolist())
        metadata = {"rows": len(values), "features": values.shape[1], "input_bytes": values.nbytes}
    # 各算法独立进程；计时前完成解释器/依赖导入，限制BLAS为单线程。
    with threadpool_limits(limits=1):
        operation()
        samples = []
        for _ in range(options.repeats):
            result, observed = measure(operation)
            samples.append(observed)
    return {"case": case, **metadata, "samples": samples, "result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["mdlp_legacy", "mdlp_prefix", "vif_legacy", "vif_qr"])
    parser.add_argument("--rows-mdlp", type=int, default=4000)
    parser.add_argument("--rows-vif", type=int, default=8000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    if options.case:
        print(json.dumps(worker(options.case, options)))
    else:
        report = {"environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "blas_threads": 1, "warmup_runs": 1},
                  "scope": "MDLP force candidate kernel and one-pass VIF, not full fitting; RSS sampled every 2ms in separate processes"}
        for case in ["mdlp_legacy", "mdlp_prefix", "vif_legacy", "vif_qr"]:
            command = [sys.executable, __file__, "--case", case, "--rows-mdlp", str(options.rows_mdlp), "--rows-vif", str(options.rows_vif), "--features", str(options.features), "--repeats", str(options.repeats)]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            report[case] = json.loads(completed.stdout)
        np.testing.assert_array_equal(report["mdlp_legacy"]["result"], report["mdlp_prefix"]["result"])
        np.testing.assert_allclose(report["vif_legacy"]["result"], report["vif_qr"]["result"], rtol=1e-8, atol=1e-10)
        report["equivalence"] = True
        for prefix, old, new in [("mdlp", "mdlp_legacy", "mdlp_prefix"), ("vif", "vif_legacy", "vif_qr")]:
            report[prefix + "_median_speedup"] = float(np.median([x["seconds"] for x in report[old]["samples"]]) / np.median([x["seconds"] for x in report[new]["samples"]]))
        print(json.dumps(report, ensure_ascii=True, indent=2))
