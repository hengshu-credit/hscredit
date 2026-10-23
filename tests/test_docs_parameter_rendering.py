"""参数枚举和 API 目录的实际解析回归。"""

import runpy
from pathlib import Path

from docutils import nodes
from docutils.core import publish_doctree

ROOT = Path(__file__).resolve().parents[1]
CONFIG = runpy.run_path(str(ROOT / "docs" / "conf.py"))
normalize = CONFIG["normalize_parameter_docs"]


def test_mixed_enum_values_render_as_literals_and_a_list():
    source = """说明。

:param mode: 取值"auto"、`fast`、``'safe'``，默认None。
    - 'auto': 自动
    - True 或 False: 开关
:type mode: str
"""
    lines = normalize(source.splitlines())
    document = publish_doctree("\n".join(lines))
    assert not list(document.findall(nodes.system_message))
    assert [node.astext() for node in document.findall(nodes.literal)] == [
        "'auto'",
        "fast",
        "'safe'",
        "None",
        "'auto'",
        "True",
        "False",
    ]
    assert len(list(document.findall(nodes.bullet_list))) == 1
    assert len(list(document.findall(nodes.list_item))) == 2
    assert normalize(lines) == lines


def test_references_examples_and_fixed_width_tables_are_preserved():
    source = """:param mode: 参见 :class:`Mode` 和 `外部文档 <https://example.com>`_。

    .. code-block:: python

        mode = "auto"
        enabled = True

    示例::

        {"mode": "auto", "enabled": False}

    >>> mode = "auto"
    >>> print(mode)
    auto

    ======= ========
    值      含义
    ======= ========
    'auto'  自动
    ======= ========

:type mode: Literal["auto"]
"""
    assert normalize(source.splitlines()) == source.splitlines()


def test_existing_attribute_literals_are_not_wrapped_twice():
    lines = [":param trials: 如 ``tuner.study_.best_trial.number`` 或裸属性 study_。"]
    CONFIG["_process_docstring"](None, "method", "test", None, None, lines)
    document = publish_doctree("\n".join(lines))
    assert not list(document.findall(nodes.system_message))
    assert [node.astext() for node in document.findall(nodes.literal)] == ["tuner.study_.best_trial.number", "study_"]


def test_real_enum_table_renders_without_column_width_errors():
    import inspect
    from hscredit.report.mining.manual_tree_extractor import DecisionTreeAnalyzer

    lines = inspect.getdoc(DecisionTreeAnalyzer.evaluate).splitlines()
    CONFIG["_process_docstring"](None, "method", "evaluate", None, None, lines)
    document = publish_doctree("\n".join(lines))
    assert not list(document.findall(nodes.system_message))
    table = next(document.findall(nodes.table))
    assert [node.astext() for node in table.findall(nodes.literal)] == ["'auc'", "'ks'", "'lift'", "'top'"]


def test_literal_with_accidental_inner_spaces_is_rendered():
    lines = normalize([":param rules: 支持 `` [{'type': 'cell',", "    'name': '金额'}]`` 。"])
    document = publish_doctree("\n".join(lines))
    assert not list(document.findall(nodes.system_message))
    assert [node.astext() for node in document.findall(nodes.literal)] == ["[{'type': 'cell', 'name': '金额'}]"]


def test_api_layout_checks_actual_object_parentage(tmp_path):
    from scripts.validate_api_layout import collect_api_layout_errors

    api = tmp_path / "api"
    api.mkdir()
    body = (
        '<dl class="py class"><dt id="hscredit.Example">类</dt><dd>'
        '<dl class="py method"><dt id="hscredit.Example.fit">方法</dt></dl></dd></dl>'
    )
    nav = (
        '<li class="toctree-l4"><a href="#hscredit.Example">类</a><ul>'
        '<li class="toctree-l5"><a href="#hscredit.Example.fit">方法</a></li></ul></li>'
    )
    path = api / "example.html"
    path.write_text(nav + body, encoding="utf-8")
    assert collect_api_layout_errors(tmp_path) == []
    path.write_text(nav.replace("toctree-l4", "toctree-l5") + body, encoding="utf-8")
    assert any("hscredit.Example 应位于第 4 层" in error for error in collect_api_layout_errors(tmp_path))
    path.write_text(body, encoding="utf-8")
    assert len(collect_api_layout_errors(tmp_path)) == 2
