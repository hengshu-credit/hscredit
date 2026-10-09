"""固定分箱统计的全量/分块差分基准；子进程峰值包含解释器与库基线。"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def worker(mode, rows):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from hscredit.core.metrics import BinStatsAccumulator, compute_bin_stats
    def arrays(start, stop):
        index = np.arange(start, stop, dtype=np.int64)
        return (index * 17) % 31, ((index * 7) % 13 > 8).astype(np.int8)
    compute_bin_stats(*arrays(0, 1000))
    start = time.perf_counter()
    if mode == 'whole':
        result = compute_bin_stats(*arrays(0, rows), round_digits=False)
    else:
        accumulator = BinStatsAccumulator()
        for offset in range(0, rows, 10_000):
            accumulator.update(*arrays(offset, min(offset + 10_000, rows)))
        result = accumulator.finalize(round_digits=False)
    elapsed = time.perf_counter() - start
    info = psutil.Process().memory_info()
    print(json.dumps({'mode': mode, 'rows': rows, 'seconds': elapsed,
                      'rss_bytes': info.rss, 'peak_rss_bytes': getattr(info, 'peak_wset', None),
                      'table': result.to_dict('list')}, ensure_ascii=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['whole', 'stream'])
    parser.add_argument('--rows', type=int, default=1_000_000)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.rows)
        return
    all_results = []
    for rows in [10_000, 100_000, args.rows]:
        cases = []
        for _ in range(3):
            for mode in ['whole', 'stream']:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                    '--rows', str(rows)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                    env=dict(os.environ, PYTHONUTF8='1', MPLBACKEND='Agg'), timeout=120, check=True)
                cases.append(json.loads(completed.stdout))
        expected = cases[0]['table']
        assert all(case.pop('table') == expected for case in cases)
        all_results.append({'rows': rows, 'identical': True, 'runs': cases,
            'median_seconds': {mode: statistics.median(c['seconds'] for c in cases if c['mode'] == mode) for mode in ['whole', 'stream']}})
    result = {'scope': '固定31箱、非训练、非完整报告；峰值来自独立子进程，非所有平台都提供peak_wset',
              'python': sys.version, 'results': all_results}
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()
