"""验证 cloudpickle 使用公开依赖，不依赖 joblib 的内部打包布局。"""

import os
from pathlib import Path
import subprocess
import sys


def test_import_and_callable_serialization_without_joblib_vendored_cloudpickle():
    """新版 joblib 移除内置 cloudpickle 后，导入、apply 和报告序列化仍可用。"""
    script = """
import pickle
import sys
import threading

import joblib.externals

# 先完成旧版 joblib 自身初始化，再模拟新版不再暴露 vendored cloudpickle。
sys.modules["joblib.externals.cloudpickle"] = None
if hasattr(joblib.externals, "cloudpickle"):
    del joblib.externals.cloudpickle

import hscredit
import pandas as pd
from sklearn.dummy import DummyClassifier

from hscredit.report import ModelReport
from hscredit.utils.pandas_parallel import _is_cloudpickle_serializable

factor = 2.0
transform = lambda value: value * factor
assert _is_cloudpickle_serializable(transform)
assert not _is_cloudpickle_serializable(threading.Lock())

frame = pd.DataFrame({"金额": [1.0, 2.0, 3.0]})
actual = frame.hscredit(n_jobs=1, bar=False).apply(transform)
pd.testing.assert_frame_equal(actual, frame * factor)

report = ModelReport(
    DummyClassifier(),
    X_train=frame,
    y_train=pd.Series([0, 1, 0], name="标签"),
    method=lambda self, x: x["金额"].to_numpy() * factor,
    n_jobs=1,
)
restored = pickle.loads(pickle.dumps(report))
assert restored.method(restored, frame).tolist() == [2.0, 4.0, 6.0]
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
