"""hsbin 表格分析操作测试。"""

from pathlib import Path

from openpyxl import load_workbook

from hscredit.skills_runtime import execute_skill


def _request(tmp_path, operation, parameters, name):
    return {
        "version": "1",
        "operation": operation,
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": parameters,
        "output": {"directory": str(tmp_path), "name": name, "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }


def _workbook(result):
    artifact = next(item for item in result["artifacts"] if item["type"] == "excel")
    path = Path(artifact["path"])
    assert path.is_file()
    return load_workbook(path, read_only=True)


def test_feature_bin_stats_executes_real_hscredit_and_writes_excel(tmp_path, credit_frame):
    """防止适配器返回空壳结果或跳过真实分箱统计。"""
    request = _request(
        tmp_path,
        "feature_bin_stats",
        {
            "feature": "score",
            "target": "target",
            "method": "quantile",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "score_stats",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    assert result["status"] == "success"
    assert result["summary"]["rows"] >= 2
    assert result["summary"]["preview"]
    assert _workbook(result).sheetnames == ["分箱统计"]


def test_feature_bin_stats_combines_overdue_labels_and_exposes_combinations(tmp_path, credit_frame):
    """多标签分箱应单次执行，并在现有表格摘要中追加全部标签组合。"""
    data = credit_frame.copy()
    data["overdue_a"] = data["target"] * 10
    data["overdue_b"] = data["target"] * 20
    request = _request(
        tmp_path,
        "feature_bin_stats",
        {
            "feature": "score",
            "overdue": ["overdue_a", "overdue_b"],
            "dpds": [0, 7],
            "method": "quantile",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "multi_label_stats",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": data})

    assert result["summary"]["rows"] >= 2
    assert result["summary"]["preview"]
    assert result["summary"]["label_combinations"] == [
        {"overdue": "overdue_a", "dpd": 0},
        {"overdue": "overdue_a", "dpd": 7},
        {"overdue": "overdue_b", "dpd": 0},
        {"overdue": "overdue_b", "dpd": 7},
    ]
    assert _workbook(result).sheetnames == ["分箱统计"]


def test_benchmark_binning_methods_writes_method_comparison(tmp_path, credit_frame):
    """防止方法对比操作调用错误的目标列或遗漏结果。"""
    request = _request(
        tmp_path,
        "benchmark_binning_methods",
        {
            "feature": "score",
            "overdue_col": "MOB1",
            "dpds": [0],
            "hscredit_methods": ["quantile", "tree"],
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "binning_benchmark",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    assert result["summary"]["rows"] == 2
    assert _workbook(result).sheetnames == ["方法对比"]


def test_feature_binning_summary_writes_summary_and_detail_sheets(tmp_path, credit_frame):
    """防止跨方法摘要丢失嵌套分箱明细。"""
    request = _request(
        tmp_path,
        "feature_binning_summary",
        {
            "feature": ["score", "age"],
            "methods": ["quantile"],
            "target": "target",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "binning_summary",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    workbook = _workbook(result)

    assert result["summary"]["rows"] == 2
    assert workbook.sheetnames[0] == "分箱摘要"
    assert any(name.startswith("score-quantile") for name in workbook.sheetnames)
    assert any(name.startswith("age-quantile") for name in workbook.sheetnames)


def test_multi_label_binning_summary_operations_expose_combinations(tmp_path, credit_frame):
    """跨方法及分组摘要都应保留多标签组合元数据。"""
    data = credit_frame.copy()
    data["overdue_a"] = data["target"] * 10
    data["overdue_b"] = data["target"] * 20
    expected = [
        {"overdue": "overdue_a", "dpd": 0},
        {"overdue": "overdue_a", "dpd": 7},
        {"overdue": "overdue_b", "dpd": 0},
        {"overdue": "overdue_b", "dpd": 7},
    ]
    cases = [
        (
            "feature_binning_summary",
            {
                "feature": "score",
                "methods": "quantile",
                "overdue": ["overdue_a", "overdue_b"],
                "dpds": [0, 7],
                "max_n_bins": 4,
                "n_jobs": 1,
            },
        ),
        (
            "feature_group_binning_summary",
            {
                "feature": "score",
                "methods": "quantile",
                "group_col": "segment",
                "overdue": ["overdue_a", "overdue_b"],
                "dpds": [0, 7],
                "max_n_bins": 4,
                "n_jobs": 1,
            },
        ),
    ]

    for operation, parameters in cases:
        result = execute_skill(
            "hsbin",
            _request(tmp_path, operation, parameters, f"{operation}_multi_label"),
            objects={"data:credit": data},
        )

        assert result["summary"]["rows"] >= 1
        assert result["summary"]["label_combinations"] == expected
        assert len([artifact for artifact in result["artifacts"] if artifact["type"] == "excel"]) == 1


def test_feature_group_binning_summary_writes_group_details(tmp_path, credit_frame):
    """防止分组摘要复算分箱规则或丢失分组明细。"""
    request = _request(
        tmp_path,
        "feature_group_binning_summary",
        {
            "feature": "score",
            "methods": "quantile",
            "group_col": "segment",
            "target": "target",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "group_binning_summary",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    workbook = _workbook(result)

    assert result["summary"]["rows"] == 2
    assert workbook.sheetnames[0] == "分组摘要"
    assert any(name.startswith("score-quantile-") for name in workbook.sheetnames)
