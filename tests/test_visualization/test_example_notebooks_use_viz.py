"""确保演示 notebook 统一使用 hscredit 可视化 API。"""

import ast
import json
from pathlib import Path


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
