"""独立、中等规模、单进程算法差分计时；不是生产容量认证。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import json
import platform
import threading
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from hscredit.core.encoders import WOEEncoder, CatBoostEncoder


def measure(fn):
    gc.collect()
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stopped = threading.Event()
    def sample():
        while not stopped.wait(.005):
            peak[0] = max(peak[0], proc.memory_info().rss)
    worker = threading.Thread(target=sample)
    worker.start()
    started = perf_counter()
    try:
        result = fn()
    finally:
        elapsed = perf_counter() - started
        peak[0] = max(peak[0], proc.memory_info().rss)
        stopped.set()
        worker.join()
    return result, {'秒': elapsed, '采样峰值RSS字节': peak[0]}


def legacy_woe(encoder, x, y):
    total_good, total_bad = (y == 0).sum(), (y == 1).sum()
    mapping = {}
    for category in x.unique():
        mask = x == category
        mapping[category] = encoder._compute_woe((y[mask] == 0).sum(), (y[mask] == 1).sum(), total_good, total_bad)
    return mapping, encoder._compute_iv_categorical(x, y, total_good, total_bad)


def legacy_catboost(encoder, x, y, order):
    result = pd.Series(index=x.index, dtype=float)
    counts, sums = {}, {}
    for idx in order:
        category = x.iloc[idx]
        count = counts.get(category, 0)
        result.iloc[idx] = (sums.get(category, 0) + encoder.global_mean_) / (count + 1)
        counts[category] = count + 1
        sums[category] = sums.get(category, 0) + y.iloc[idx]
    return result


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    x = pd.Series(rng.randint(0, 300, 100000)).astype(str)
    y = pd.Series(rng.randint(0, 2, len(x)))
    woe = WOEEncoder(n_jobs=1)
    cb = CatBoostEncoder(n_jobs=1, random_state=42)
    cb.global_mean_ = float(y.mean())
    order = rng.permutation(30000)
    stats = {'环境': {'Python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__}, 'WOE_n':len(x), '类别':300, 'CatBoost_n':len(order)}
    # 预热与正确性对比
    old = legacy_woe(woe, x.iloc[:100], y.iloc[:100])
    new = woe._fit_categorical(x.iloc[:100], y.iloc[:100], (y.iloc[:100] == 0).sum(), (y.iloc[:100] == 1).sum())
    assert np.isclose(old[1], new[1])
    for name, fn in {
        'WOE_旧两轮类别扫描': lambda:legacy_woe(woe,x,y),
        'WOE_factorize': lambda:woe._fit_categorical(x,y,(y==0).sum(),(y==1).sum()),
        'CatBoost_旧逐行iloc': lambda:legacy_catboost(cb,x.iloc[:30000],y.iloc[:30000],order),
        'CatBoost_分组累计': lambda:cb._transform_ordered(x.iloc[:30000],y.iloc[:30000],{},random_order=order),
    }.items():
        observations=[]
        for _ in range(3):
            result, observation=measure(fn)
            observations.append(observation)
        stats[name]=observations
        if name == 'WOE_旧两轮类别扫描':
            reference_woe=result
        elif name == 'WOE_factorize':
            assert np.isclose(reference_woe[1],result[1])
            assert all(np.isclose(value,result[0][key]) for key,value in reference_woe[0].items())
        elif name == 'CatBoost_旧逐行iloc':
            reference_cb=result
        else:
            np.testing.assert_allclose(reference_cb,result,atol=1e-12)
    print(json.dumps(stats,ensure_ascii=False))
