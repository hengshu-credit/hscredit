"""文档导航与搜索构建产物的回归测试。"""

from pathlib import Path

import pytest
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


def _write_package_menu(tmp_path, path, label, level, children):
    links = "".join(f'<li class="toctree-l{level + 1}"><a href="{child}.html">{child}</a></li>' for child in children)
    _write(tmp_path / "api" / path, f'<li class="toctree-l{level}"><a href="#">{label}</a><ul>{links}</ul></li>')


def test_accepts_complete_docs_artifacts(tmp_path):
    """完整产物应通过导航、搜索运行时与布局契约校验。"""
    _write_package_menu(
        tmp_path,
        "index.html",
        "API 参考",
        1,
        ["core", "report", "database", "excel", "utils", "skills_runtime", "exceptions"],
    )
    _write_package_menu(
        tmp_path,
        "core.html",
        "核心模块 hscredit.core",
        2,
        [
            "binning",
            "encoders",
            "selectors",
            "modeling",
            "metrics",
            "feature_engineering",
            "eda",
            "viz",
            "rules",
            "financial",
            "model_selection",
        ],
    )
    _write_package_menu(tmp_path, "report.html", "报告 hscredit.report", 2, ["mining"])
    _write_package_menu(tmp_path, "losses.html", "损失函数 hscredit.core.models.losses", 4, ["model_metrics"])
    _write(
        tmp_path / "api" / "model_metrics.html",
        '<li class="toctree-l7"><a href="#hscredit.core.models.BaseMetric.evaluate">evaluate</a></li>',
    )
    _write(
        tmp_path / "api" / "modeling.html",
        '<li class="toctree-l3 current"><a href="#">模型 hscredit.core.models</a><ul>'
        '<li class="toctree-l4"><a href="classical_models.html">经典模型 hscredit.core.models.classical</a></li>'
        '<li class="toctree-l4"><a href="boosting.html">提升模型 hscredit.core.models.boosting</a></li>'
        '<li class="toctree-l4"><a href="net.html">神经网络 hscredit.core.models.net</a></li>'
        '<li class="toctree-l4"><a href="model_rules.html">规则模型 hscredit.core.models.rules</a></li>'
        '<li class="toctree-l4"><a href="scorecard.html">评分卡 hscredit.core.models.scorecard</a></li>'
        '<li class="toctree-l4"><a href="calibration.html">概率校准 hscredit.core.models.calibration</a></li>'
        '<li class="toctree-l4"><a href="explainability.html">模型解释 hscredit.core.models.explainability</a></li>'
        '<li class="toctree-l4"><a href="losses.html">损失函数 hscredit.core.models.losses</a></li>'
        '<li class="toctree-l4"><a href="tuning.html">参数搜索 hscredit.core.models.tuning</a></li>'
        "</ul></li>",
    )
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l6"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoost.fit">'
        "fit</a></li>"
        '<li class="toctree-l6"><a href="#hscredit.core.models.boosting.xgboost_model.XGBoost.predict">'
        "predict</a></li>",
    )
    _write(
        tmp_path / "api" / "eda.html",
        '<dt id="hscredit.core.eda.feature_summary">feature_summary</dt>',
    )
    _write(
        tmp_path / "database.html",
        "数据库 大 JSON 字段按路径读取 json_fields",
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


@pytest.mark.parametrize(
    "link",
    [
        '<li class="toctree-l4"><a href="net.html">神经网络 hscredit.core.models.net</a></li>',
        '<li class="toctree-l4"><a href="tuning.html">参数搜索 hscredit.core.models.tuning</a></li>',
    ],
)
def test_rejects_missing_model_api_menu_entry(tmp_path, link):
    """模型 API 必须保留可直接访问的导航入口。"""
    test_accepts_complete_docs_artifacts(tmp_path)
    path = tmp_path / "api" / "modeling.html"
    path.write_text(path.read_text(encoding="utf-8").replace(link, ""), encoding="utf-8")

    assert any("模型菜单" in error for error in collect_validation_errors(tmp_path))


def test_rejects_nested_or_reordered_model_menu(tmp_path):
    """模型菜单必须按顺序直接生成九个带模块路径的 API 入口。"""
    _write(
        tmp_path / "api" / "modeling.html",
        '<li class="toctree-l3 current"><a href="#">模型 hscredit.core.models</a><ul>'
        '<li class="toctree-l4"><a href="models.html">模型 hscredit.core.models</a><ul>'
        '<li class="toctree-l4"><a href="models.html#classical">经典模型</a></li>'
        '<li class="toctree-l4"><a href="models.html#rules">规则器</a></li>'
        "</ul></li>"
        '<li class="toctree-l4"><a href="boosting.html">提升模型 hscredit.core.models.boosting</a></li>'
        '<li class="toctree-l4"><a href="tuning.html">参数搜索 hscredit.core.models.tuning</a></li>'
        "</ul></li>",
    )

    errors = collect_validation_errors(tmp_path)

    assert any(
        "模型菜单" in error
        and "经典模型 hscredit.core.models.classical、提升模型 hscredit.core.models.boosting、神经网络 hscredit.core.models.net、规则模型 hscredit.core.models.rules、评分卡 hscredit.core.models.scorecard、概率校准 hscredit.core.models.calibration、模型解释 hscredit.core.models.explainability、损失函数 hscredit.core.models.losses、参数搜索 hscredit.core.models.tuning"
        in error
        for error in errors
    )


@pytest.mark.parametrize(
    "page, child, label",
    [("core", "modeling", "核心模块"), ("report", "mining", "报告"), ("losses", "model_metrics", "损失函数")],
)
def test_rejects_package_outside_its_code_parent(tmp_path, page, child, label):
    """子包链接移到父包外，即使页面仍存在，也不能通过结构校验。"""
    import re

    test_accepts_complete_docs_artifacts(tmp_path)
    path = tmp_path / "api" / f"{page}.html"
    source = path.read_text(encoding="utf-8")
    pattern = rf'<li class="toctree-l\d"><a href="{child}\.html">[^<]*</a></li>'
    misplaced = re.search(pattern, source).group()
    path.write_text(re.sub(pattern, "", source) + misplaced, encoding="utf-8")
    assert any("API 包目录" in error and label in error for error in collect_validation_errors(tmp_path))


@pytest.mark.parametrize("extra", ["modeling", "tooling"])
def test_rejects_functional_group_as_top_level_package(tmp_path, extra):
    """API 顶层不能重新混入并列的模型或跨目录工具分组。"""
    test_accepts_complete_docs_artifacts(tmp_path)
    path = tmp_path / "api" / "index.html"
    source = path.read_text(encoding="utf-8")
    source = source.replace("</ul>", f'<li class="toctree-l2"><a href="{extra}.html">{extra}</a></li></ul>', 1)
    path.write_text(source, encoding="utf-8")
    assert any("API 包目录 API 参考" in error for error in collect_validation_errors(tmp_path))


@pytest.mark.parametrize("level", [5, 7])
def test_rejects_wrong_model_method_depth(tmp_path, level):
    """模型目录加深后，第六层仍必须可展开 fit/predict。"""
    test_accepts_complete_docs_artifacts(tmp_path)
    path = tmp_path / "api" / "boosting.html"
    path.write_text(path.read_text(encoding="utf-8").replace("toctree-l6", f"toctree-l{level}"), encoding="utf-8")
    assert any("第 6 层" in error for error in collect_validation_errors(tmp_path))


def test_rejects_truncated_loss_metric_methods(tmp_path):
    test_accepts_complete_docs_artifacts(tmp_path)
    path = tmp_path / "api" / "model_metrics.html"
    path.write_text('<main id="hscredit.core.models.BaseMetric.evaluate">evaluate</main>', encoding="utf-8")
    assert any("第 7 层" in error for error in collect_validation_errors(tmp_path))


def test_api_package_indexes_cover_source_packages():
    """API 包索引必须覆盖实际公开目录，目录新增或挂错父级都会失败。"""
    import re

    root = Path(__file__).resolve().parents[1]
    package_indexes = {
        "hscredit": (
            "index",
            {
                "core": "core",
                "report": "report",
                "database": "database",
                "excel": "excel",
                "utils": "utils",
                "skills_runtime": "skills_runtime",
            },
        ),
        "hscredit/core": (
            "core",
            {
                "binning": "binning",
                "encoders": "encoders",
                "selectors": "selectors",
                "models": "modeling",
                "metrics": "metrics",
                "feature_engineering": "feature_engineering",
                "eda": "eda",
                "viz": "viz",
                "rules": "rules",
                "financial": "financial",
            },
        ),
        "hscredit/core/models": (
            "modeling",
            {
                "classical": "classical_models",
                "boosting": "boosting",
                "net": "net",
                "rules": "model_rules",
                "scorecard": "scorecard",
                "calibration": "calibration",
                "explainability": "explainability",
                "losses": "losses",
                "tuning": "tuning",
            },
        ),
        "hscredit/report": ("report", {"mining": "mining"}),
    }
    for package, (index, pages) in package_indexes.items():
        actual = {p.parent.name for p in (root / package).glob("*/__init__.py") if not p.parent.name.startswith("_")}
        assert actual == set(pages), package
        source = (root / "docs/api" / f"{index}.rst").read_text(encoding="utf-8")
        blocks = re.findall(r"(?m)^\.\. toctree::\n((?:[ \t]+[^\n]*\n|\n)*)", source)
        entries = {
            line.strip()
            for block in blocks
            for line in block.splitlines()
            if line.strip() and not line.strip().startswith(":")
        }
        assert set(pages.values()) <= entries, package
        for page in pages.values():
            assert (root / "docs/api" / f"{page}.rst").is_file()


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

    assert any("第 6 层" in error for error in errors)
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

    _write(tmp_path / "database.html", "数据库")
    _write(
        tmp_path / "api" / "database.html",
        '<dt id="hscredit.database.client.Database">Database</dt>',
    )
    _write(tmp_path / "api" / "tooling.html", "<html></html>")
    _write(tmp_path / "searchindex.js", "Search.setIndex({})")

    errors = collect_validation_errors(tmp_path)

    assert any("API 包目录 API 参考" in error and "database.html" in error for error in errors)
    assert any("Database" in error and "搜索结果" in error for error in errors)


def test_rejects_database_docs_without_json_projection_and_public_method_anchors(tmp_path):
    """数据库指南与 API 页面必须发布 JSON 投影和流式结果方法。"""
    _write(tmp_path / "database.html", "数据库")
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
        '<li class="toctree-l6"><a href="#XGBoost.fit">fit</a></li>',
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
        '<li class="toctree-l6"><a href="#XGBoost.fit">fit</a></li>'
        '<li class="toctree-l6"><a href="#XGBoost.predict">predict</a></li>',
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


def test_rejects_method_anchors_outside_level_six_navigation(tmp_path):
    """正文锚点与无关第六层条目不能冒充类方法菜单入口。"""
    _write(
        tmp_path / "api" / "boosting.html",
        '<li class="toctree-l6"><a href="#unrelated">unrelated</a></li>'
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
        '<li class="toctree-l6"><a href="#XGBoost.fit">fit</a></li>'
        '<li class="toctree-l6"><a href="#XGBoost.predict">predict</a></li>',
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
