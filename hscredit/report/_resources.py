"""报告缓存与逐行数据的显式生命周期，默认保留完整交互能力。"""

import sys
import numpy as np
import pandas as pd
from ..exceptions import StateError


def retained_bytes(values):
    """按对象身份去重估算 Python 可见数据，不冒充操作系统 RSS。"""
    seen = set()

    def measure(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, (pd.Series, pd.DataFrame)):
            size = value.memory_usage(index=True, deep=True)
            return int(size.sum() if hasattr(size, "sum") else size)
        if isinstance(value, np.ndarray):
            return value.nbytes + (sum(measure(item) for item in value.flat) if value.dtype.hasobject else 0)
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(measure(key) + measure(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(measure(item) for item in value)
        if type(value).__module__.startswith("hscredit.core.binning") and hasattr(value, "__dict__"):
            return measure(vars(value))
        if isinstance(value, (str, bytes, int, float, bool)) or value is None:
            return sys.getsizeof(value)
        return 0

    return measure(values)


class ReportResourceMixin:
    """显式释放派生缓存或逐行数据，释放后不静默重训或重新读取。"""

    def _init_resource_state(self):
        from threading import RLock

        self._operation_lock = RLock()
        self._active_operations = 0

    def _start_report_operation(self):
        with self._operation_lock:
            self._active_operations += 1

    def _finish_report_operation(self):
        with self._operation_lock:
            self._active_operations -= 1
            if self._active_operations == 0 and self.cache_policy == "none" and not getattr(self, "_data_released", False):
                self.release_cache()

    def release_cache(self):
        """释放派生表、图形数据和解释缓存；训练分箱规则与原始数据保留。"""
        with self._operation_lock:
            if self._active_operations:
                raise StateError("报告任务仍在执行，不能释放缓存")
            for name, value in list(vars(self).items()):
                if name.endswith("_cache"):
                    setattr(self, name, {} if isinstance(value, dict) else None)

    def resource_usage(self):
        """报告数据的估算保留量，不包括外部模型原生内存。"""
        rows = [item for dataset in self._datasets.values() for item in (dataset.X, dataset.y, dataset.score, dataset.y_dict)]
        caches = {key: value for key, value in vars(self).items() if key.endswith("_cache")}
        return {
            "缓存策略": self.cache_policy,
            "数据已释放": getattr(self, "_data_released", False),
            "数据集数": len(self._datasets),
            "保留行数": sum(len(d.y) for d in self._datasets.values()),
            "估算保留字节": retained_bytes([rows, caches, self.init_params_, self._reference_binners]),
            "估算缓存字节": retained_bytes(caches),
            "估算规则字节": retained_bytes(self._reference_binners),
        }

    def release_data(self):
        """保留整体指标与摘要，释放报告引用的逐行数据；明细和导出须重建报告。"""
        if getattr(self, "_data_released", False):
            return
        self._assert_model_current()
        summary = self.summary()
        metrics = {label: self.get_metrics(label) for label in [None, *self._label_names]}
        self.release_cache()
        self._metrics_cache = metrics
        self._summary_cache = summary
        self._datasets.clear()
        self._reference_binners = {}
        self._reference_locks = {}
        for key in ("datasets", "X_train", "y_train", "X_test", "y_test", "X_oot", "y_oot", "target", "sample_weight"):
            self.init_params_[key] = None
        self._target_cfg = None
        self.sample_weight = None
        self.metrics = []
        self.label_metrics_ = []
        self.kwargs = {}
        self.prediction_kwargs = {}
        self._training_process_snapshot = {}
        self.explain_config = dict(self.explain_config)
        for key in ("data", "background_data", "X_train", "y_train", "X_validation"):
            self.explain_config[key] = None
        self.init_params_["explain_config"] = self.explain_config
        self.init_params_["metrics"] = None
        self.init_params_["prediction_kwargs"] = None
        self._data_released = True

    def _require_report_data(self):
        if getattr(self, "_data_released", False):
            raise StateError("报告逐行数据已释放；仅可读取已缓存的整体指标与摘要，请重新创建报告后生成明细")
