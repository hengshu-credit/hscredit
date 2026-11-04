"""在独立进程记录冷启动、指标、报告和流式聚合的耗时及进程树峰值内存。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from tempfile import TemporaryDirectory


def worker(case, rows):
    import psutil

    process = psutil.Process()
    peak = [process.memory_info().rss]
    stopped = threading.Event()

    def sample():
        children = []
        refresh_at = 0.0
        while not stopped.wait(0.05):
            try:
                now = time.perf_counter()
                if now >= refresh_at:
                    children = process.children(recursive=True)
                    refresh_at = now + 0.25
                rss = process.memory_info().rss
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak[0] = max(peak[0], rss)
            except psutil.Error:
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    artifact_bytes = 0
    try:
        import hscredit as h
        import numpy as np
        import pandas as pd

        if case == "import":
            result = {"version": h.__version__, "auc": h.auc([0, 1], [0.1, 0.9])}
        elif case == "metrics":
            rng = np.random.RandomState(42)
            y = rng.randint(0, 2, rows)
            p = rng.uniform(size=rows)
            result = {"auc": h.auc(y, p), "ks": h.ks(y, p)}
        elif case in ("report", "excel"):
            rng = np.random.RandomState(42)
            X = pd.DataFrame(rng.normal(size=(rows, 8)), columns=[f"x{i}" for i in range(8)])
            y = (X.x0 + X.x1 > 0).astype(int)
            model = h.RandomForest(n_estimators=10, max_depth=4, n_jobs=1, random_state=42).fit(X, y)
            report = h.ModelReport(model, X_train=X, y_train=y, cache_policy="none", n_jobs=1)
            result = {
                "metrics": report.get_metrics().to_dict("records"),
                "bins": report.get_bin_table().to_dict("records"),
            }
            if case == "excel":
                from openpyxl import load_workbook

                with TemporaryDirectory(prefix="hscredit_benchmark_") as directory:
                    path = Path(directory) / "report.xlsx"
                    report.to_excel(path, with_plots=False, include_sample_records=False, mode="strict")
                    artifact_bytes = path.stat().st_size
                    book = load_workbook(path, read_only=True)
                    result["sheets"] = book.sheetnames
                    book.close()
        else:
            from hscredit.database.stream import QueryStream

            class Source:
                columns = ["bin", "target"]
                position = 0

                def fetchmany(self, size):
                    indices = np.arange(self.position, min(rows, self.position + size))
                    self.position += len(indices)
                    return pd.DataFrame({"bin": indices % 10, "target": (indices % 3 == 0).astype(int)})

                def close(self):
                    pass

            stream = QueryStream(Source(), chunksize=4096, retain=False)
            table = stream.aggregate(h.BinStatsAccumulator(), lambda batch: (batch.bin, batch.target))
            assert stream.resource_usage()["保留行数"] == 0
            result = table.to_dict("records")
        from hscredit.utils.json import json_safe
        from hscredit.utils.environment import runtime_version

        signature = hashlib.sha256(
            json.dumps(json_safe(result), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return {
            "场景": case,
            "样本数": rows,
            "耗时秒": time.perf_counter() - started,
            "进程树采样峰值RSS": max(peak[0], process.memory_info().rss),
            "结果摘要": signature,
            "依赖": {name: runtime_version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
            "导入模式": "light" if os.environ.get("HSCREDIT_LIGHT_IMPORT") == "1" else "full",
            "产物字节": artifact_bytes,
        }
    finally:
        stopped.set()
        sampler.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = ["import", "metrics", "report", "excel", "stream"]
    parser.add_argument("--worker", choices=cases)
    parser.add_argument("--cases", nargs="+", choices=cases, default=cases)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(".codex-temp/workflow-benchmark.json"))
    args = parser.parse_args()
    if args.rows < 20 or args.repeats < 1:
        parser.error("rows 至少为20，repeats 至少为1")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if args.worker:
        print(json.dumps(worker(args.worker, args.rows), ensure_ascii=False))
        return
    results = []
    for mode in ("full", "light"):
        for case in args.cases:
            for repetition in range(args.repeats):
                environment = dict(
                    os.environ,
                    HSCREDIT_LIGHT_IMPORT="1" if mode == "light" else "0",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                )
                command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(Path(__file__).resolve()),
                    "--worker",
                    case,
                    "--rows",
                    str(args.rows),
                ]
                completed = subprocess.run(
                    command, env=environment, cwd=root, text=True, encoding="utf-8", capture_output=True, check=True
                )
                result = json.loads(completed.stdout.splitlines()[-1])
                result["重复序号"] = repetition + 1
                results.append(result)
    for case in args.cases:
        if len({r["结果摘要"] for r in results if r["场景"] == case}) != 1:
            raise RuntimeError(f"{case} 的执行模式或重复运行产生不同结果")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"结果": results, "说明": "独立进程；耗时包含冷启动；RSS按50毫秒采样、子进程列表250毫秒刷新；不隔离整机负载，不是硬内存限制"}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
