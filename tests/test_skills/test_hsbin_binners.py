"""hsbin 分箱器生命周期操作测试。"""

from pathlib import Path

from hscredit.core.binning import OptimalBinning, OptimalBinning2D
from hscredit.skills_runtime import execute_skill


def _fit_request(tmp_path, operation, parameters, name):
    return {
        "version": "1",
        "operation": operation,
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": parameters,
        "output": {"directory": str(tmp_path), "name": name, "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }


def _artifact(result):
    path = Path(next(item["path"] for item in result["artifacts"] if item["type"] == "hscredit-artifact"))
    assert path.is_file()
    return path


def test_optimal_binning_fit_publishes_a_loadable_artifact(tmp_path, credit_frame):
    """防止 fitted binner 丢失真实分箱元数据。"""
    request = _fit_request(
        tmp_path,
        "optimal_binning_fit",
        {
            "features": ["score", "age"],
            "target": "target",
            "method": "quantile",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "binner",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    loaded = OptimalBinning.load_artifact(_artifact(result))

    assert list(loaded.feature_names_in_) == ["score", "age"]
    assert result["summary"]["rows"] == len(credit_frame)


def test_optimal_binning_transform_accepts_a_trusted_artifact(tmp_path, credit_frame):
    """防止文件型 binner 制品无法串联到后续 Skill 请求。"""
    fit_result = execute_skill(
        "hsbin",
        _fit_request(
            tmp_path,
            "optimal_binning_fit",
            {
                "features": ["score", "age"],
                "target": "target",
                "method": "quantile",
                "max_n_bins": 4,
                "n_jobs": 1,
            },
            "fitted_binner",
        ),
        objects={"data:credit": credit_frame},
    )
    artifact = _artifact(fit_result)
    request = {
        "version": "1",
        "operation": "optimal_binning_transform",
        "inputs": {
            "data": {"kind": "object_ref", "ref": "data:credit"},
            "binner": {"kind": "file", "path": str(artifact), "trusted": True},
        },
        "parameters": {"features": ["score", "age"], "metric": "woe"},
        "output": {"directory": str(tmp_path), "name": "woe_data", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    assert result["summary"]["rows"] == len(credit_frame)
    assert result["summary"]["columns"] == ["score", "age"]


def test_optimal_binning_2d_fit_publishes_a_loadable_artifact(tmp_path, credit_frame):
    """防止二维分箱只输出图片而无法复用拟合结果。"""
    request = _fit_request(
        tmp_path,
        "optimal_binning_2d_fit",
        {
            "features": ["score", "age"],
            "target": "target",
            "method": "quantile",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "binner_2d",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    loaded = OptimalBinning2D.load_artifact(_artifact(result))

    assert [loaded.feature_x_, loaded.feature_y_] == ["score", "age"]
    assert result["summary"]["rows"] == len(credit_frame)


def test_optimal_binning_2d_transform_uses_fitted_feature_names(tmp_path, credit_frame):
    """防止二维制品转换读取不存在的通用 feature_names 属性。"""
    fit_result = execute_skill(
        "hsbin",
        _fit_request(
            tmp_path,
            "optimal_binning_2d_fit",
            {
                "features": ["score", "age"],
                "target": "target",
                "method": "quantile",
                "max_n_bins": 4,
                "n_jobs": 1,
            },
            "fitted_binner_2d",
        ),
        objects={"data:credit": credit_frame},
    )
    request = {
        "version": "1",
        "operation": "optimal_binning_2d_transform",
        "inputs": {
            "data": {"kind": "object_ref", "ref": "data:credit"},
            "binner": {"kind": "file", "path": str(_artifact(fit_result)), "trusted": True},
        },
        "parameters": {"metric": "event_rate"},
        "output": {"directory": str(tmp_path), "name": "event_rate_2d", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    assert result["summary"]["rows"] == len(credit_frame)
