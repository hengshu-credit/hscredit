"""公开参数文档与可执行 Notebook 的契约校验。"""

import ast
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _constructors():
    for package in ("binning", "encoders", "selectors"):
        for path in sorted((ROOT / "hscredit" / "core" / package).glob("*.py")):
            if path.name.startswith("_"):
                continue
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                    continue
                init = next(
                    (member for member in node.body if isinstance(member, ast.FunctionDef) and member.name == "__init__"),
                    None,
                )
                if init is not None:
                    yield node.name, node, init


@pytest.mark.parametrize("entry", list(_constructors()), ids=lambda entry: entry[0])
def test_constructor_parameters_visible_on_class(entry):
    """Sphinx 使用类 docstring，不能只在 __init__ 或内部实现中写参数。"""
    name, cls, init = entry
    doc = ast.get_docstring(cls) or ""
    arguments = init.args.posonlyargs + init.args.args + init.args.kwonlyargs
    names = [arg.arg for arg in arguments if arg.arg not in {"self", "cls"}]
    names += [arg.arg for arg in (init.args.vararg, init.args.kwarg) if arg is not None]
    missing = [arg for arg in names if not re.search(rf":param {re.escape(arg)}:", doc)]
    assert not missing, f"{name} 的公开类文档缺少参数：{missing}"


def test_binning_method_reference_matches_registry():
    from hscredit.core.binning import OptimalBinning

    text = (ROOT / "docs/api/conventions.rst").read_text(encoding="utf-8")
    method_section = text.split("当前 17 个值是", 1)[1].split("每种方法", 1)[0]
    documented = re.findall(r"``([a-z_]+)``", method_section)
    assert set(documented) - {"optimal"} == set(OptimalBinning.VALID_METHODS)
    assert "optimal" not in OptimalBinning.VALID_METHODS


def test_api_reference_notebook_with_real_data(tmp_path, monkeypatch):
    """按单元顺序执行同一份演示，验证统计行数、返回类型和导出产物。"""
    if not (ROOT / "examples/hscredit_yyp.xlsx").is_file():
        pytest.skip("真实放款数据为本地验证资源，不随仓库分发")
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("HSCREDIT_EXAMPLE_OUTPUT", str(tmp_path))
    notebook = json.loads((ROOT / "examples/29_api_reference.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__api_reference__"}
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            exec(compile(source, f"29_api_reference.ipynb:单元格{index}", "exec"), namespace)
    assert int(namespace["table"]["样本总数"].sum()) == len(namespace["data"])
    assert (tmp_path / "特征分析.xlsx").is_file()
