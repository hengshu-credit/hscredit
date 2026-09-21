"""在独立 Jupyter kernel 中执行公共契约与资源保留教程。"""

import os
from pathlib import Path
import subprocess
import sys


def test_model_workflow_notebook_executes_all_semantic_checks(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "artifacts"
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        MPLBACKEND="Agg",
        HSCREDIT_EXAMPLE_INPUT=str(tmp_path / "use_synthetic_data.xlsx"),
        HSCREDIT_EXAMPLE_OUTPUT=str(output),
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_examples.py", "--pattern", "28_model_workflow.ipynb", "--timeout", "120"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "model_full.joblib").is_file()
    assert (output / "model_inference.joblib").is_file()
    assert (output / "model_inference.joblib").stat().st_size < (output / "model_full.joblib").stat().st_size
    assert (output / "tuner.joblib").is_file()
    assert len(list((output / "disk").rglob("*_fold_*.pkl"))) == 6
