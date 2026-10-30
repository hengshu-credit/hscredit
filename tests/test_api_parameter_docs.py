"""通过真实 Sphinx 渲染检查字段说明，防止只验证源代码中存在参数名。"""

import io
import runpy
from pathlib import Path

import pytest
from docutils import nodes
from sphinx.application import Sphinx

from scripts.validate_api_parameter_docs import collect_parameter_doc_errors, parameter_docs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rendered_api(tmp_path_factory):
    source = tmp_path_factory.mktemp("api-docs")
    (source / "conf.py").write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"config = runpy.run_path({str(ROOT / 'docs/conf.py')!r})\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']\n"
        "autodoc_typehints = 'description'\n"
        "language = 'zh_CN'\n"
        "def setup(app):\n"
        "    app.connect('object-description-transform', config['merge_api_field_lists'], priority=600)\n"
        "    app.connect('autodoc-process-docstring', config['_process_docstring'])\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "API\n===\n\n" ".. toctree::\n\n   api/report\n",
        encoding="utf-8",
    )
    (source / "api").mkdir()
    (source / "api/report.rst").write_text(
        "报告\n====\n\n"
        ".. autoclass:: hscredit.report.ReportResult\n   :members:\n\n"
        ".. autoclass:: hscredit.report.ReportSectionResult\n\n"
        ".. autoclass:: hscredit.report.ReportGenerationError\n\n"
        ".. autoclass:: hscredit.report.ModelReport\n   :class-doc-from: both\n\n"
        ".. autoclass:: hscredit.core.selectors.IVSelector\n\n"
        ".. autofunction:: hscredit.report.feature_bin_stats\n\n"
        ".. autofunction:: hscredit.report.feature_bin_stats_2d\n",
        encoding="utf-8",
    )
    warnings = io.StringIO()
    app = Sphinx(
        str(source),
        str(source),
        str(source / "html"),
        str(source / "doctrees"),
        "html",
        status=io.StringIO(),
        warning=warnings,
    )
    app.build(force_all=True)
    assert not warnings.getvalue(), warnings.getvalue()
    return source / "html"


def test_actual_report_parameters_have_descriptions_once(rendered_api):
    assert collect_parameter_doc_errors(rendered_api) == []
    html = (rendered_api / "api/report.html").read_text(encoding="utf-8")
    docs = parameter_docs(html)
    from hscredit.report import ReportResult, ReportSectionResult, feature_bin_stats, feature_bin_stats_2d
    import inspect

    for obj in (ReportResult, ReportSectionResult, feature_bin_stats, feature_bin_stats_2d):
        fields = docs[f"hscredit.report.{obj.__name__}"]
        assert set(dict(fields)) == set(inspect.signature(obj).parameters)
        assert all(description for _, description in fields)
    for field in ("mode", "metadata", "sections", "artifacts", "sheet_mapping"):
        assert field in html
    assert html.index("<strong>属性</strong>") < html.index("<strong>sheet_mapping</strong>")
    for parameter in ("data", "parallel_config", "overdue_operator"):
        assert f"<strong>{parameter}</strong> (" in html
    for method in ("plan", "run", "skip", "success", "failure"):
        fields = docs[f"hscredit.report.ReportResult.{method}"]
        assert fields and all(description for _, description in fields)


def test_validator_rejects_empty_and_duplicate_parameters(tmp_path):
    api = tmp_path / "api"
    api.mkdir()
    (api / "sample.html").write_text(
        '<dl class="py function"><dt id="hscredit.example">函数</dt><dd>'
        '<dl class="field-list"><dt>参数:</dt><dd><ul>'
        "<li><p><strong>x</strong> (<em>str</em>)</p></li>"
        "<li><p><strong>x</strong> -- 字段名</p></li>"
        "</ul></dd></dl></dd></dl>",
        encoding="utf-8",
    )
    errors = collect_parameter_doc_errors(tmp_path)
    assert len(errors) == 2
    assert any("缺少参数说明" in error for error in errors)
    assert any("参数重复展示" in error for error in errors)


@pytest.mark.parametrize("separator", [" -- ", " – ", " — ", "&nbsp;&ndash;&nbsp;"])
def test_parameter_descriptions_support_sphinx_smart_punctuation(separator):
    html = (
        '<dl class="py function"><dt id="hscredit.example">函数</dt><dd>'
        '<dl class="field-list"><dt>参数:</dt><dd><p><strong>x</strong> (<em>str</em>)'
        + separator
        + "<code>auto</code> 表示自动选择。</p></dd></dl></dd></dl>"
    )
    assert parameter_docs(html)["hscredit.example"] == [("x", "auto 表示自动选择。")]


def test_type_annotation_separator_does_not_count_as_a_description(tmp_path):
    api = tmp_path / "api"
    api.mkdir()
    (api / "example.html").write_text(
        '<dl class="py function"><dt id="hscredit.example">函数</dt><dd>'
        '<dl class="field-list"><dt>参数:</dt><dd><p><strong>x</strong>'
        " (<em>Literal[' -- ']</em>)</p></dd></dl></dd></dl>",
        encoding="utf-8",
    )
    assert collect_parameter_doc_errors(tmp_path) == ["hscredit.example.x: 缺少参数说明"]


def test_merging_fields_preserves_nested_examples_and_other_domains():
    merge = runpy.run_path(str(ROOT / "docs/_ext/parameter_docs.py"))["merge_api_field_lists"]
    content = nodes.container()
    first, extra, nested = nodes.field_list(), nodes.field_list(), nodes.field_list()
    first += nodes.field("", nodes.field_name("", "param x"), nodes.field_body("", nodes.paragraph("", "字段")))
    extra += nodes.field("", nodes.field_name("", "param y"), nodes.field_body("", nodes.paragraph("", "目标")))
    note = nodes.note("", nested)
    code = nodes.literal_block("", ":param x: 示例原文")
    content.extend([first, note, code, extra])
    merge(None, "js", "function", content)
    assert len(content) == 4
    merge(None, "py", "function", content)
    assert list(content) == [first, note, code]
    assert len(first) == 2
    assert note[0] is nested and code.astext() == ":param x: 示例原文"
