"""两个独立 Skill launcher 的文件型端到端测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
EXECUTION_MODE = os.environ.get("HSCREDIT_SKILL_TEST_ENV", "current")


def _run_launcher(skill, request_path):
    result = subprocess.run(
        [sys.executable, f"skills/{skill}/scripts/run.py", str(request_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_hsbin_launcher_executes_a_file_request(tmp_path, credit_frame):
    """验证自包含 hsbin launcher 能从 CSV 交付真实 Excel。"""
    data_path = tmp_path / "credit.csv"
    credit_frame.to_csv(data_path, index=False, encoding="utf-8")
    request_path = tmp_path / "hsbin-request.json"
    request_path.write_text(
        json.dumps(
            {
                "version": "1",
                "operation": "feature_bin_stats",
                "inputs": {"data": {"kind": "file", "path": str(data_path)}},
                "parameters": {
                    "feature": "score",
                    "target": "target",
                    "method": "quantile",
                    "max_n_bins": 4,
                    "n_jobs": 1,
                },
                "output": {"directory": str(tmp_path), "name": "cli-bin-stats", "overwrite": False},
                "environment": {
                    "mode": EXECUTION_MODE,
                    "install_missing": EXECUTION_MODE == "isolated",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = _run_launcher("hsbin", request_path)

    assert response["status"] == "success"
    assert Path(response["artifacts"][0]["path"]).is_file()


def test_hsreport_launcher_executes_a_file_request(tmp_path, credit_frame):
    """验证自包含 hsreport launcher 能从 CSV 交付真实报告。"""
    data_path = tmp_path / "credit.csv"
    credit_frame.to_csv(data_path, index=False, encoding="utf-8")
    request_path = tmp_path / "hsreport-request.json"
    request_path.write_text(
        json.dumps(
            {
                "version": "1",
                "operation": "auto_feature_analysis",
                "inputs": {"data": {"kind": "file", "path": str(data_path)}},
                "parameters": {
                    "features": ["score"],
                    "target": "target",
                    "pictures": [],
                    "n_jobs": 1,
                },
                "output": {"directory": str(tmp_path), "name": "cli-feature-report", "overwrite": False},
                "environment": {
                    "mode": EXECUTION_MODE,
                    "install_missing": EXECUTION_MODE == "isolated",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = _run_launcher("hsreport", request_path)

    assert response["status"] == "success"
    assert Path(response["artifacts"][0]["path"]).is_file()
