"""教程段落必须显式自包含，且不以抽样或隐藏包装替代真实调用。"""

import ast
import builtins
from pathlib import Path
import symtable

import nbformat
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = ["01_binning", "02_encoders", "03_selectors", "04_models", "05_scorecard", "06_net", "07_rules", "08_rule_mining", "09_excel"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_sections_have_no_hidden_dependencies_or_sampling(name):
    notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells[0].source == "import sys\nsys.path.append('../')"
    sections = [cell for cell in code_cells if "independent-example" in cell.metadata.get("tags", [])]
    assert sections
    for cell in sections:
        tree = ast.parse(cell.source)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) for node in ast.walk(tree)), cell.metadata.example_id
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sample" for node in ast.walk(tree)), cell.metadata.example_id
        loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hscredit_dataset"]
        assert len(loads) == 1
        assert any(keyword.arg == "columns" and isinstance(keyword.value, ast.Name) and keyword.value.id == "HSCREDIT_EXAMPLE_COLUMNS" for keyword in loads[0].keywords)

        table = symtable.symtable(cell.source, name, "exec")
        defined = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned() or symbol.is_imported()} | set(dir(builtins))
        scopes = [table]
        while scopes:
            scope = scopes.pop()
            scopes.extend(scope.get_children())
            missing = {symbol.get_name() for symbol in scope.get_symbols() if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in defined}
            assert not missing, f"{cell.metadata.example_id} 依赖前文变量: {sorted(missing)}"
