"""确保演示 notebook 统一使用 hscredit 可视化 API。"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

_DRAW_METHODS = {
    "bar",
    "barh",
    "boxplot",
    "countplot",
    "errorbar",
    "fill_between",
    "heatmap",
    "hexbin",
    "hist",
    "imshow",
    "jointplot",
    "kdeplot",
    "matshow",
    "pairplot",
    "pie",
    "plot",
    "regplot",
    "scatter",
    "stackplot",
    "violinplot",
}
_HSCREDIT_OBJECT_PREFIXES = ("binner", "viz_")


def _attribute_name(node):
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def test_example_notebooks_do_not_draw_with_low_level_libraries():
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    violations = []

    for path in sorted(examples_dir.glob("*.ipynb")):
        if path.name == "建模参考代码.ipynb":
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                call_name = _attribute_name(node.func)
                if call_name.rsplit(".", 1)[-1] not in _DRAW_METHODS:
                    continue
                root_name = call_name.split(".", 1)[0]
                if call_name.endswith(".plot") and root_name.startswith(_HSCREDIT_OBJECT_PREFIXES):
                    continue
                violations.append(f"{path.name}: cell {cell_index}: {call_name}")

    assert not violations, "发现未使用 hscredit 可视化 API 的绘图调用:\n" + "\n".join(violations)


def test_models_notebook_compares_top_badcases_with_similar_correct_samples(monkeypatch):
    """模型教程应按坏样本率阈值选取两类 Top2，并对匹配样本成对绘制 SHAP。"""
    notebook_path = Path(__file__).resolve().parents[2] / "examples" / "04_models.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    section_indexes = [index for index, cell in enumerate(notebook["cells"]) if cell.get("cell_type") == "markdown" and "Badcase" in "".join(cell.get("source", []))]
    assert section_indexes, "04_models.ipynb 缺少 Badcase 分析章节"
    section_source = "\n\n".join("".join(cell.get("source", [])) for cell in notebook["cells"][section_indexes[-1] + 1 :] if cell.get("cell_type") == "code")

    probabilities = pd.Series(
        [0.95, 0.85, 0.20, 0.10, 0.15, 0.25, 0.05, 0.12, 0.80, 0.75],
        index=range(10),
    )

    class FakeModel:
        def predict_proba(self, values):
            positive = probabilities.loc[values.index].to_numpy()
            return np.column_stack([1.0 - positive, positive])

    X_test = pd.DataFrame(
        {
            "特征1": [0.0, 10.0, 0.1, 9.9, 5.0, 6.0, 20.0, 30.0, 20.1, 30.1],
            "特征2": [0.0, 10.0, 0.1, 10.1, 5.0, 6.0, 20.0, 30.0, 20.1, 30.0],
        }
    )
    X_train = pd.DataFrame(
        {
            "特征1": [-1.0, 0.0, 5.0, 10.0, 20.0, 30.0],
            "特征2": [-1.0, 0.0, 5.0, 10.0, 20.0, 30.0],
        }
    )
    y_test = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], index=X_test.index)
    plot_calls = []

    def fake_plot_model_sample_shap(**kwargs):
        plot_calls.append(kwargs)

    monkeypatch.setattr(
        "hscredit.core.viz.plot_model_sample_shap",
        fake_plot_model_sample_shap,
    )
    namespace = {
        "np": np,
        "pd": pd,
        "display": lambda *args, **kwargs: None,
        "xgboost_model": FakeModel(),
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    exec(compile(section_source, str(notebook_path), "exec"), namespace)

    assert np.isclose(namespace["badcase_threshold"], 0.45)
    assert namespace["selected_badcases"].index.tolist() == [0, 1, 6, 7]
    assert [(pair["badcase_index"], pair["similar_correct_index"]) for pair in namespace["badcase_pairs"]] == [(0, 2), (1, 3), (6, 8), (7, 9)]
    assert [call["sample"].index[0] for call in plot_calls] == [0, 2, 1, 3, 6, 8, 7, 9]
    assert all(call["background_data"].equals(X_train) for call in plot_calls)
    assert all(call["max_display"] is None for call in plot_calls)
