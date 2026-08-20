"""hsbin 分箱可视化和效率分析测试。"""

from pathlib import Path

import matplotlib
import pytest
from PIL import Image

matplotlib.use("Agg")

from hscredit.skills_runtime import execute_skill


def _request(tmp_path, operation, parameters, name):
    return {
        "version": "1",
        "operation": operation,
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": parameters,
        "output": {
            "directory": str(tmp_path),
            "name": name,
            "format": "png",
            "overwrite": False,
        },
        "environment": {"mode": "current", "install_missing": False},
    }


def _assert_images(result, minimum=1):
    images = [Path(item["path"]) for item in result["artifacts"] if item["type"] == "image"]
    assert len(images) >= minimum
    for path in images:
        assert path.is_file()
        with Image.open(path) as rendered:
            assert rendered.format == "PNG"
            assert rendered.width >= 500
            assert rendered.height >= 300


def test_bin_plot_renders_a_decodable_nonempty_png(tmp_path, credit_frame):
    """防止 bin_plot 只返回 Figure 而不交付图片。"""
    request = _request(
        tmp_path,
        "bin_plot",
        {
            "feature": "score",
            "target": "target",
            "method": "quantile",
            "n_bins": 4,
            "anchor": 0.17,
        },
        "score_bin_plot",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    _assert_images(result)


@pytest.mark.parametrize(
    ("operation", "parameters", "name"),
    [
        (
            "bin_trend_plot",
            {
                "feature": "score",
                "target": "target",
                "date_col": "apply_date",
                "date_freq": "M",
                "method": "quantile",
                "max_n_bins": 4,
            },
            "score_trend",
        ),
        (
            "bin_overdues_plot",
            {
                "feature": "score",
                "overdue": ["MOB1"],
                "dpds": [1],
                "method": "quantile",
                "max_n_bins": 4,
            },
            "score_overdue",
        ),
        (
            "bin_2d_plot",
            {
                "features": ["score", "age"],
                "target": "target",
                "method": "quantile",
                "max_n_bins": 4,
            },
            "score_age_2d",
        ),
    ],
)
def test_hsbin_plot_operations_render_real_images(
    tmp_path,
    credit_frame,
    operation,
    parameters,
    name,
):
    """防止趋势、逾期或二维绘图操作注册为空壳。"""
    result = execute_skill(
        "hsbin",
        _request(tmp_path, operation, parameters, name),
        objects={"data:credit": credit_frame},
    )

    _assert_images(result)


def test_feature_efficiency_analysis_publishes_tables_rules_and_images(tmp_path, credit_frame):
    """防止效率分析只保留图对象而丢失可交付产物。"""
    request = _request(
        tmp_path,
        "feature_efficiency_analysis",
        {
            "feature": "score",
            "target": "target",
            "auto_method": "quantile",
            "date_col": "apply_date",
            "date_freq": "M",
            "max_n_bins": 4,
            "n_jobs": 1,
        },
        "score_efficiency",
    )

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    _assert_images(result, minimum=3)
    assert any(item["type"] == "excel" for item in result["artifacts"])
    assert any(item["type"] == "json" for item in result["artifacts"])
    assert result["summary"]["feature"] == "score"
