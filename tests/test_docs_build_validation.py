"""文档导航与搜索构建产物的回归测试。"""

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
from docutils import nodes
from docutils.core import publish_doctree

from hscredit.database.client import Database

from scripts.validate_docs_build import collect_search_runtime_errors, collect_validation_errors

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10 compatibility
    import tomli as tomllib


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_accepts_complete_docs_artifacts(tmp_path):
    """完整产物应通过导航、搜索运行时与布局契约校验。"""
    _write(
        tmp_path / "api" / "modeling.html",
        '<li class="toctree-l2 current"><a href="#">模型</a><ul>'
        '<li class="toctree-l3"><a href="classical_models.html">经典模型</a></li>'
        '<li class="toctree-l3"><a href="boosting.html">Boosting</a></li>'
        '<li class="toctree-l3"><a href="model_rules.html">规则器</a></li>'
        '<li class="toctree-l3"><a href="scorecard.html">评分卡</a></li>'
        '<li class="toctree-l3"><a href="losses.html">损失函数</a></li>'
        '<li class="toctree-l3"><a href="model_metrics.html">评估指标</a></li>'
        '<li class="toctree-l3"><a href="tuning.html">超参数调优</a></li>'
        "</ul></li>",
    )
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoost.fit">'
        "fit</a></li>"
        '<li class="toctree-l5"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoost.predict">'
        "predict</a></li>",
    )
    _write(
        tmp_path / "api" / "eda.html",
        '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>',
    )
    _write(
        tmp_path / "database.html",
        "数据库与 NoSQL 连接池、读写及表结构导出 大 JSON 字段按路径读取 json_fields",
    )
    _write(
        tmp_path / "api" / "database.html",
        '<dt id="hscredit.database.client.Database">Database</dt>'
        '<dt id="hscredit.database.client.Database.stream_query">stream_query</dt>'
        '<dt id="hscredit.database.stream.QueryStream.to_result">to_result</dt>'
        '<dt id="hscredit.database.shortcuts.query">query</dt>'
        '<dt id="hscredit.database.shortcuts.read_query">read_query</dt>'
        '<dt id="hscredit.database.shortcuts.stream_write">stream_write</dt>',
    )
    _write(
        tmp_path / "api" / "tooling.html",
        '<li class="toctree-l2"><a href="database.html">数据库</a></li>',
    )
    _write(
        tmp_path / "searchindex.js",
        'Search.setIndex({"docnames":["api/eda","api/database"],"objects":{'
        '"hscredit.core.eda":[[0,0,1,"","feature_summary"]],'
        '"hscredit.database.client":[[1,0,1,"","Database"]]}})',
    )
    _write(tmp_path / "_static" / "language_data.js", "var Stemmer = function () {};")
    _write(
        tmp_path / "_static" / "custom.css",
        ".wy-menu-vertical a { box-sizing: border-box; border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.current > a { border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.hs-scrollspy-current > a { border-left: 3px solid transparent; }",
    )

    assert collect_validation_errors(tmp_path) == []


def test_rejects_nested_or_reordered_model_menu(tmp_path):
    """模型菜单必须移除包级中间页，并按产品顺序生成七个直接子项。"""
    _write(
        tmp_path / "api" / "modeling.html",
        '<li class="toctree-l2 current"><a href="#">模型</a><ul>'
        '<li class="toctree-l3"><a href="models.html">模型 hscredit.core.models</a><ul>'
        '<li class="toctree-l4"><a href="models.html#classical">经典模型</a></li>'
        '<li class="toctree-l4"><a href="models.html#rules">规则器</a></li>'
        "</ul></li>"
        '<li class="toctree-l3"><a href="boosting.html">Boosting</a></li>'
        '<li class="toctree-l3"><a href="tuning.html">超参数调优</a></li>'
        "</ul></li>",
    )

    errors = collect_validation_errors(tmp_path)

    assert any(
        "模型菜单" in error and "经典模型、Boosting、规则器、评分卡、损失函数、评估指标、超参数调优" in error
        for error in errors
    )


def test_reports_broken_search_runtime_and_navigation(tmp_path):
    """坏产物必须同时报告方法导航、索引、词干器和稳定边框问题。"""
    _write(tmp_path / "api" / "boosting.html", "<html></html>")
    _write(tmp_path / "api" / "eda.html", "<html></html>")
    _write(tmp_path / "searchindex.js", "Search.setIndex({})")
    _write(
        tmp_path / "_static" / "language_data.js",
        "var EnglishStemmer = function () {}; window.Stemmer=ChineseStemmer;",
    )
    _write(tmp_path / "_static" / "custom.css", ".wy-menu-vertical a { color: white; }")

    errors = collect_validation_errors(tmp_path)

    assert any("第 5 层" in error for error in errors)
    assert any("feature_summary" in error for error in errors)
    assert any("ChineseStemmer" in error for error in errors)
    assert any("3px" in error for error in errors)


def test_reports_missing_database_guide_and_api_artifacts(tmp_path):
    """数据库用户指南与 API 页面必须是发布构建的一部分。"""

    errors = collect_validation_errors(tmp_path)

    assert any("database.html" in error for error in errors)
    assert any("api/database.html" in error for error in errors)


def test_rejects_database_docs_without_navigation_or_search_result(tmp_path):
    """页面存在但未进入 API 导航和搜索对象表时仍应阻止发布。"""

    _write(tmp_path / "database.html", "数据库与 NoSQL 连接池、读写及表结构导出")
    _write(
        tmp_path / "api" / "database.html",
        '<dt id="hscredit.database.client.Database">Database</dt>',
    )
    _write(tmp_path / "api" / "tooling.html", "<html></html>")
    _write(tmp_path / "searchindex.js", "Search.setIndex({})")

    errors = collect_validation_errors(tmp_path)

    assert any("数据库 API 导航" in error for error in errors)
    assert any("Database" in error and "搜索结果" in error for error in errors)


def test_rejects_database_docs_without_json_projection_and_public_method_anchors(tmp_path):
    """数据库指南与 API 页面必须发布 JSON 投影和流式结果方法。"""
    _write(tmp_path / "database.html", "数据库与 NoSQL 连接池、读写及表结构导出")
    _write(
        tmp_path / "api" / "database.html",
        '<dt id="hscredit.database.client.Database">Database</dt>',
    )

    errors = collect_validation_errors(tmp_path)

    assert any("JSON 字段投影" in error for error in errors)
    assert any("stream_query" in error and "to_result" in error for error in errors)


def test_rejects_database_api_without_shortcut_method_anchors(tmp_path):
    """数据库 API 页面必须发布类外快捷查询和写入入口。"""
    _write(
        tmp_path / "api" / "database.html",
        '<dt id="hscredit.database.client.Database">Database</dt>'
        '<dt id="hscredit.database.client.Database.stream_query">stream_query</dt>'
        '<dt id="hscredit.database.stream.QueryStream.to_result">to_result</dt>',
    )

    errors = collect_validation_errors(tmp_path)

    assert any("类外快捷操作" in error for error in errors)


def test_reports_current_nav_rule_that_drops_reserved_border(tmp_path):
    """当前项规则必须覆盖 RTD 的 border:none，保持点击前后文字位置一致。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#XGBoost.fit">fit</a></li>',
    )
    _write(tmp_path / "api" / "eda.html", '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>')
    _write(tmp_path / "searchindex.js", 'Search.setIndex({"terms":{"feature_summary":1}})')
    _write(tmp_path / "_static" / "language_data.js", "var Stemmer = function () {};")
    _write(
        tmp_path / "_static" / "custom.css",
        ".wy-menu-vertical a { box-sizing: border-box; border-left: 3px solid transparent; }",
    )

    errors = collect_validation_errors(tmp_path)

    assert any("当前侧边栏链接" in error and "3px" in error for error in errors)


def test_rejects_feature_summary_outside_searchable_api_objects(tmp_path):
    """无关元数据中的函数名不能冒充可跳转的 API 搜索结果。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#XGBoost.fit">fit</a></li>'
        '<li class="toctree-l5"><a href="#XGBoost.predict">predict</a></li>',
    )
    _write(tmp_path / "api" / "eda.html", '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>')
    _write(
        tmp_path / "searchindex.js",
        'Search.setIndex({"docnames":["api/eda"],"objects":{},"metadata":"feature_summary"})',
    )
    _write(tmp_path / "_static" / "language_data.js", "var Stemmer = function () {};")
    _write(
        tmp_path / "_static" / "custom.css",
        ".wy-menu-vertical a { box-sizing: border-box; border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.current > a { border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.hs-scrollspy-current > a { border-left: 3px solid transparent; }",
    )

    errors = collect_validation_errors(tmp_path)

    assert any("搜索结果" in error and "feature_summary" in error for error in errors)


def test_rejects_method_anchors_outside_level_five_navigation(tmp_path):
    """正文锚点与无关第五层条目不能冒充类方法菜单入口。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#unrelated">unrelated</a></li>'
        '<main><a href="#XGBoost.fit">fit</a><a href="#XGBoost.predict">predict</a></main>',
    )
    _write(tmp_path / "api" / "eda.html", '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>')
    _write(
        tmp_path / "searchindex.js",
        'Search.setIndex({"docnames":["api/eda"],"objects":{"hscredit.core.eda":' '[[0,0,1,"","feature_summary"]]}})',
    )
    _write(tmp_path / "_static" / "language_data.js", "var Stemmer = function () {};")
    _write(
        tmp_path / "_static" / "custom.css",
        ".wy-menu-vertical a { box-sizing: border-box; border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.current > a { border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.hs-scrollspy-current > a { border-left: 3px solid transparent; }",
    )

    errors = collect_validation_errors(tmp_path)

    assert any("fit()" in error and "predict()" in error for error in errors)


def test_rejects_later_border_shorthand_in_the_same_rule(tmp_path):
    """同一规则内后出现的 border 简写不得破坏稳定左边框。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#XGBoost.fit">fit</a></li>'
        '<li class="toctree-l5"><a href="#XGBoost.predict">predict</a></li>',
    )
    _write(tmp_path / "api" / "eda.html", '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>')
    _write(
        tmp_path / "searchindex.js",
        'Search.setIndex({"docnames":["api/eda"],"objects":{"hscredit.core.eda":' '[[0,0,1,"","feature_summary"]]}})',
    )
    _write(tmp_path / "_static" / "language_data.js", "var Stemmer = function () {};")
    _write(
        tmp_path / "_static" / "custom.css",
        ".wy-menu-vertical a { box-sizing: border-box; border-left: 3px solid transparent; }"
        ".wy-menu-vertical li.current > a { border-left: 3px solid transparent; border: none; }"
        ".wy-menu-vertical li.hs-scrollspy-current > a { border-left: 3px solid transparent; }",
    )

    errors = collect_validation_errors(tmp_path)

    assert any("当前侧边栏链接" in error and "3px" in error for error in errors)


def test_runtime_probe_rejects_throwing_language_data(tmp_path):
    """即使 Stemmer 已定义，加载脚本时抛错也必须阻止发布。"""
    _write(
        tmp_path / "_static" / "language_data.js",
        "var Stemmer = function () { this.stemWord = function (word) { return word; }; };"
        "throw new Error('search runtime failed');",
    )

    errors = collect_search_runtime_errors(tmp_path)

    assert any("搜索运行时执行失败" in error for error in errors)


def test_docs_dependency_excludes_broken_sphinx_9_search():
    """docs 依赖必须保留 Sphinx 8，并拒绝已知搜索失效的 Sphinx 9。"""
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    docs_requirements = [Requirement(value) for value in config["project"]["optional-dependencies"]["docs"]]
    sphinx = next(requirement for requirement in docs_requirements if requirement.name.lower() == "sphinx")

    assert Version("8.2.3") in sphinx.specifier
    assert Version("9.0.0") not in sphinx.specifier


def test_database_class_docstring_is_valid_rst():
    """数据库公共门面 docstring 不得向 autodoc 注入 RST 解析告警。"""

    document = publish_doctree(Database.__doc__ or "")
    messages = list(document.findall(nodes.system_message))

    assert messages == []
