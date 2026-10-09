"""报告任务提供可检查的工作区/结果/输出规模估算，不把估算当RSS硬界限。"""

import numpy as np
import pandas as pd

from hscredit.report.feature_analyzer import _feature_report_workload
from tests.test_report.test_audit_report_contracts import small_model_report


def test_feature_resource_estimate_tracks_dtype_labels_rules_and_long_format():
    data = pd.DataFrame({"x": np.arange(100.), "y": np.tile([0, 1], 50)})
    parameters = {"overdue": ["y"], "dpds": [7, 3, 0], "rules": list(range(12)), "max_n_bins": 3, "margins": True}
    wide = _feature_report_workload(data, 1, has_parallel_children=False, operation="测试", parameters=[parameters])
    long = _feature_report_workload(data, 1, has_parallel_children=False, operation="测试", parameters=[{**parameters, "long_format": True}])
    assert wide.output_rows_per_task == 17  # 13个数值箱+3个保留箱+合计
    assert long.output_rows_per_task == wide.output_rows_per_task * 3
    assert long.result_bytes_per_task > wide.result_bytes_per_task > 0
    assert wide.working_bytes_per_task > data.memory_usage(deep=True).sum()
    objects = data.assign(x=["x" * 200] * len(data))
    larger = _feature_report_workload(objects, 1, has_parallel_children=False, operation="测试", parameters=[parameters])
    assert larger.working_bytes_per_task > wide.working_bytes_per_task


def test_model_report_parallel_workloads_describe_output_and_working_buffers(monkeypatch):
    import hscredit.report.model_report as module
    captured = []
    original = module.parallel_execute
    def capture(function, tasks, **kwargs):
        workload = kwargs.get("workload")
        if workload is not None:
            captured.append(workload)
        return original(function, tasks, **kwargs)
    monkeypatch.setattr(module, "parallel_execute", capture)
    report = small_model_report()
    report.get_metrics()
    report.get_feature_importance()
    assert len(captured) >= 3
    assert all(item.working_bytes_per_task > 0 and item.result_bytes_per_task > 0 and item.output_rows_per_task > 0 for item in captured)
    prediction = next(item for item in captured if item.operation == "模型报告数据集预测")
    assert prediction.output_rows_per_task == 4
    assert prediction.result_bytes_per_task >= 4 * 8
