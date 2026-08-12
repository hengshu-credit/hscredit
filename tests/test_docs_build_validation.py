"""文档导航与搜索构建产物的回归测试。"""

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

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
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoostRiskModel.fit">'
        "fit</a></li>"
        '<li class="toctree-l5"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoostRiskModel.predict">'
        "predict</a></li>",
    )
    _write(
        tmp_path / "api" / "eda.html",
        '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>',
    )
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

    assert collect_validation_errors(tmp_path) == []


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


def test_reports_current_nav_rule_that_drops_reserved_border(tmp_path):
    """当前项规则必须覆盖 RTD 的 border:none，保持点击前后文字位置一致。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l5"><a href="#XGBoostRiskModel.fit">fit</a></li>',
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
        '<li class="toctree-l5"><a href="#XGBoostRiskModel.fit">fit</a></li>'
        '<li class="toctree-l5"><a href="#XGBoostRiskModel.predict">predict</a></li>',
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
        '<main><a href="#XGBoostRiskModel.fit">fit</a><a href="#XGBoostRiskModel.predict">predict</a></main>',
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
        '<li class="toctree-l5"><a href="#XGBoostRiskModel.fit">fit</a></li>'
        '<li class="toctree-l5"><a href="#XGBoostRiskModel.predict">predict</a></li>',
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
