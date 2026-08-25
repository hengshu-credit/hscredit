"""公开文档示例可执行性测试."""

import json
import os
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from hscredit.core.selectors import PSISelector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_markdown_python_blocks_execute(tmp_path):
    text = (PROJECT_ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    namespace = {"__name__": "__documentation_example__"}
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        for index, block in enumerate(blocks, start=1):
            exec(
                compile(block, f"docs/quickstart.md:代码块{index}", "exec"),
                namespace,
            )
    finally:
        os.chdir(original_cwd)
    assert len(blocks) == 8


def test_executable_quickstart_script(tmp_path):
    module = runpy.run_path(str(PROJECT_ROOT / "examples" / "00_quickstart.py"))
    result = module["run_quickstart"](tmp_path)

    assert result["train_rows"] == 300
    assert result["test_rows"] == 100
    assert result["calibration_rows"] == 60
    assert len(result["scores"]) == 100
    assert Path(result["artifact_path"]).exists()


def test_public_docs_do_not_use_removed_example_parameters():
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "quickstart.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "save_path=" not in text
    assert not re.search(r"extract_rules\([^)]*(top_n|metric)", text)


def test_active_docs_and_examples_use_new_model_subpackages():
    """公开入口不能继续推荐已删除的 evaluation 包或不存在的示例文件。"""
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "quickstart.md",
        PROJECT_ROOT / "docs" / "articles" / "model-interpretability.md",
        PROJECT_ROOT / "examples" / "00_quickstart.py",
        PROJECT_ROOT / "examples" / "27_model_interpretability.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "models.evaluation" not in text
    assert "examples/27_model_interpretability.py" in text
    assert "examples/27_model_interpretability.ipynb" in text
    assert (PROJECT_ROOT / "examples" / "27_model_interpretability.py").is_file()
    assert (PROJECT_ROOT / "examples" / "27_model_interpretability.ipynb").is_file()


def test_model_interpretability_notebook_uses_new_explainability_api():
    """模型解释 notebook 必须使用新子包且不保留已删除的反事实参数。"""
    notebook = json.loads(
        (PROJECT_ROOT / "examples" / "27_model_interpretability.ipynb").read_text(encoding="utf-8")
    )
    sources = [cell.get("source", []) for cell in notebook["cells"]]
    source_texts = ["".join(source) if isinstance(source, list) else source for source in sources]
    all_source = "\n".join(source_texts)
    counterfactual_source = "\n".join(text for text in source_texts if "CounterfactualExplainer(" in text)
    error_outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    output_text = "\n".join(
        "".join(output.get("text", [])) if isinstance(output.get("text", []), list) else output.get("text", "")
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )

    assert "hscredit.core.models.evaluation" not in all_source
    assert "hscredit.core.models.explainability" in all_source
    assert "random_state=" not in counterfactual_source
    assert error_outputs == []
    assert str(PROJECT_ROOT) not in output_text


def test_selectors_notebook_psi_example_compares_unequal_train_and_oot_splits():
    """PSI 教程应能直接比较样本量不同的训练集与 OOT 集。"""
    notebook_path = PROJECT_ROOT / "examples" / "03_selectors.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    psi_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "psi_selector.fit" in "".join(cell.get("source", []))
    ]
    numeric_features = ["数值特征", "稳定特征"]
    df_model = pd.DataFrame(
        {
            "数值特征": np.arange(264, dtype=float),
            "稳定特征": np.tile([0.0, 1.0], 132),
        }
    )
    namespace = {
        "PSISelector": PSISelector,
        "df_model": df_model,
        "numeric_features": numeric_features,
    }

    assert len(psi_cells) == 1
    exec(compile(psi_cells[0], str(notebook_path), "exec"), namespace)

    selector = namespace["psi_selector"]
    assert selector.n_features_in_ == 2
    assert selector.scores_.index.tolist() == numeric_features
    assert selector.oot_df.equals(df_model.iloc[158:][numeric_features])
    assert np.isfinite(selector.scores_.to_numpy()).all()
