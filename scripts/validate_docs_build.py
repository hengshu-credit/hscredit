"""校验 Sphinx HTML 构建产物中的导航与搜索契约。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Sequence

_BorderCandidate = tuple[bool, tuple[int, int, int], int, int, str]
_EXPECTED_MODEL_MENU = (
    "经典模型",
    "Boosting",
    "规则器",
    "评分卡",
    "损失函数",
    "评估指标",
    "超参数调优",
)


class _LevelFiveLinkParser(HTMLParser):
    """收集真正位于 ``li.toctree-l5`` 内的导航链接。"""

    def __init__(self) -> None:
        super().__init__()
        self._li_stack: list[bool] = []
        self.hrefs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li":
            classes = (attributes.get("class") or "").split()
            self._li_stack.append("toctree-l5" in classes)
        elif tag == "a" and any(self._li_stack):
            href = attributes.get("href")
            if href:
                self.hrefs.add(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._li_stack:
            self._li_stack.pop()


class _ModelMenuParser(HTMLParser):
    """收集“模型”二级菜单下直接生成的三级菜单文字。"""

    def __init__(self) -> None:
        super().__init__()
        self._li_stack: list[dict[str, Any]] = []
        self._anchor: tuple[int, list[str]] | None = None
        self.items: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li":
            self._li_stack.append(
                {
                    "classes": set((attributes.get("class") or "").split()),
                    "is_model": False,
                }
            )
        elif tag == "a" and self._li_stack:
            classes = self._li_stack[-1]["classes"]
            if "toctree-l2" in classes or "toctree-l3" in classes:
                self._anchor = (len(self._li_stack) - 1, [])

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            stack_index, text_parts = self._anchor
            self._anchor = None
            if stack_index >= len(self._li_stack):
                return
            item = self._li_stack[stack_index]
            label = " ".join("".join(text_parts).split())
            if "toctree-l2" in item["classes"] and label == "模型":
                item["is_model"] = True
            elif "toctree-l3" in item["classes"] and any(parent["is_model"] for parent in self._li_stack[:stack_index]):
                self.items.append(label)
        elif tag == "li" and self._li_stack:
            self._li_stack.pop()


def _read(build_dir: Path, relative_path: str, errors: list[str]) -> str:
    """读取单个构建产物，并把缺失文件转换成可汇总的中文错误。"""
    path = build_dir / relative_path
    if not path.is_file():
        errors.append(f"缺少文档构建产物：{relative_path}")
        return ""
    return path.read_text(encoding="utf-8")


def _load_search_index(source: str, errors: list[str]) -> dict[str, Any]:
    """解析 ``Search.setIndex(...)`` 中的 JSON 数据。"""
    match = re.fullmatch(r"\s*Search\.setIndex\((?P<payload>.*)\)\s*;?\s*", source, re.DOTALL)
    if not match:
        errors.append("searchindex.js 不是有效的 Search.setIndex 构建产物")
        return {}
    try:
        index = json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        errors.append(f"searchindex.js 无法解析：{exc.msg}")
        return {}
    return index if isinstance(index, dict) else {}


def _has_feature_summary_result(index: dict[str, Any]) -> bool:
    """确认搜索对象表可构造出 feature_summary 的 API 跳转结果。"""
    docnames = index.get("docnames")
    objects = index.get("objects")
    if not isinstance(docnames, list) or not isinstance(objects, dict):
        return False

    prefix = "hscredit.core.eda"
    entries = objects.get(prefix)
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 5 or entry[4] != "feature_summary":
            continue
        doc_index, anchor = entry[0], entry[3]
        if not isinstance(doc_index, int) or not 0 <= doc_index < len(docnames):
            continue
        resolved_anchor = anchor or f"{prefix}.feature_summary"
        if docnames[doc_index] == "api/eda" and resolved_anchor == "hscredit.core.eda.feature_summary":
            return True
    return False


def _has_database_result(index: dict[str, Any]) -> bool:
    """确认搜索对象表可跳转到 ``Database`` 公共类。"""
    docnames = index.get("docnames")
    objects = index.get("objects")
    if not isinstance(docnames, list) or not isinstance(objects, dict):
        return False

    prefix = "hscredit.database.client"
    entries = objects.get(prefix)
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 5 or entry[4] != "Database":
            continue
        doc_index, anchor = entry[0], entry[3]
        if not isinstance(doc_index, int) or not 0 <= doc_index < len(docnames):
            continue
        resolved_anchor = anchor or f"{prefix}.Database"
        if docnames[doc_index] == "api/database" and resolved_anchor == "hscredit.database.client.Database":
            return True
    return False


def _normalize_selector(selector: str) -> str:
    """把选择器空白标准化，便于比较同一条导航规则。"""
    selector = re.sub(r"/\*.*?\*/", " ", selector, flags=re.DOTALL)
    selector = re.sub(r"\s*([>+~])\s*", r" \1 ", selector.strip())
    return re.sub(r"\s+", " ", selector)


def _selector_specificity(selector: str) -> tuple[int, int, int]:
    """计算本文件导航选择器所需的简化 CSS specificity。"""
    ids = len(re.findall(r"#[\w-]+", selector))
    classes = len(re.findall(r"\.[\w-]+|\[[^]]+\]|(?<!:):[\w-]+", selector))
    elements = len(re.findall(r"(?:^|[\s>+~])(?![.#:\[])([a-zA-Z][\w-]*)", selector))
    return ids, classes, elements


def _winning_css_declaration(
    css_rules: list[tuple[str, str]],
    matches: Callable[[str], bool],
    property_name: str,
) -> _BorderCandidate | None:
    """按 important、specificity 与源码顺序返回目标元素的最终声明。"""
    candidates: list[_BorderCandidate] = []
    declaration_pattern = re.compile(
        rf"(?<![\w-]){re.escape(property_name)}(?![\w-])\s*:\s*(?P<value>[^;}}]+)",
        re.IGNORECASE,
    )
    for rule_order, (selectors, body) in enumerate(css_rules):
        for selector in selectors.split(","):
            normalized = _normalize_selector(selector)
            if not matches(normalized):
                continue
            for declaration in declaration_pattern.finditer(body):
                raw_value = declaration.group("value").strip().lower()
                important = bool(re.search(r"\s*!important\s*$", raw_value))
                value = re.sub(r"\s*!important\s*$", "", raw_value).strip()
                candidates.append(
                    (important, _selector_specificity(normalized), rule_order, declaration.start(), value)
                )
    return max(candidates) if candidates else None


def _is_stable_border(candidate: _BorderCandidate | str | None) -> bool:
    """判断级联结果是否仍为占位用的透明 3px 左边框。"""
    value = candidate[-1] if isinstance(candidate, tuple) else candidate
    return bool(value and re.fullmatch(r"3px\s+solid\s+transparent", value))


def _left_shorthand_value(value: str) -> str:
    """从 1 至 4 项的 border-* 简写值中取出左边对应项。"""
    values = value.split()
    if len(values) == 1:
        return values[0]
    if len(values) in (2, 3):
        return values[1]
    return values[3] if len(values) >= 4 else ""


def _override_preserves_stable_border(property_name: str, value: str) -> bool:
    """判断覆盖 border-left 子属性的声明是否仍保持稳定布局。"""
    if property_name == "border":
        return all(re.search(pattern, value) for pattern in (r"\b3px\b", r"\bsolid\b", r"\btransparent\b"))
    expected = {
        "border-width": "3px",
        "border-left-width": "3px",
        "border-style": "solid",
        "border-left-style": "solid",
        "border-color": "transparent",
        "border-left-color": "transparent",
    }[property_name]
    component = (
        _left_shorthand_value(value) if property_name in {"border-width", "border-style", "border-color"} else value
    )
    return component == expected


def _has_effective_stable_border(
    css_rules: list[tuple[str, str]],
    matches: Callable[[str], bool],
) -> bool:
    """校验 border 及其长短属性完成级联后的左边框。"""
    border_left = _winning_css_declaration(css_rules, matches, "border-left")
    if not _is_stable_border(border_left):
        return False
    assert border_left is not None

    for property_name in (
        "border",
        "border-width",
        "border-left-width",
        "border-style",
        "border-left-style",
        "border-color",
        "border-left-color",
    ):
        override = _winning_css_declaration(css_rules, matches, property_name)
        if override and override[:-1] > border_left[:-1]:
            if not _override_preserves_stable_border(property_name, override[-1]):
                return False
    return True


def collect_search_runtime_errors(build_dir: Path) -> list[str]:
    """用 Node.js 沙箱实际加载搜索语言脚本并验证 Stemmer。"""
    language_data = build_dir / "_static" / "language_data.js"
    if not language_data.is_file():
        return ["缺少文档构建产物：_static/language_data.js"]

    probe = Path(__file__).with_name("validate_search_runtime.js")
    try:
        result = subprocess.run(
            ["node", str(probe), str(language_data)],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=5,
        )
    except FileNotFoundError:
        return ["无法执行 Node.js，未能验证搜索运行时"]
    except subprocess.TimeoutExpired:
        return ["搜索运行时验证超时"]

    if result.returncode == 0:
        return []
    detail = (result.stderr or result.stdout).strip().splitlines()
    message = detail[-1] if detail else "未知 JavaScript 错误"
    return [f"搜索运行时执行失败：{message}"]


def collect_validation_errors(build_dir: Path) -> list[str]:
    """返回文档构建产物违反导航与搜索契约的错误列表。"""
    errors: list[str] = []
    modeling_html = _read(build_dir, "api/modeling.html", errors)
    boosting_html = _read(build_dir, "api/boosting.html", errors)
    eda_html = _read(build_dir, "api/eda.html", errors)
    database_html = _read(build_dir, "database.html", errors)
    database_api_html = _read(build_dir, "api/database.html", errors)
    tooling_html = _read(build_dir, "api/tooling.html", errors)
    search_index = _read(build_dir, "searchindex.js", errors)
    language_data = _read(build_dir, "_static/language_data.js", errors)
    custom_css = _read(build_dir, "_static/custom.css", errors)

    model_menu_parser = _ModelMenuParser()
    model_menu_parser.feed(modeling_html)
    if tuple(model_menu_parser.items) != _EXPECTED_MODEL_MENU:
        expected = "、".join(_EXPECTED_MODEL_MENU)
        actual = "、".join(model_menu_parser.items) or "未生成"
        errors.append(f"模型菜单必须依次生成七个直接子项：{expected}；实际为：{actual}")

    level_five_parser = _LevelFiveLinkParser()
    level_five_parser.feed(boosting_html)
    required_method_links = {
        "#hscredit.core.models.boosting.xgboost_model.XGBoost.fit",
        "#hscredit.core.models.boosting.xgboost_model.XGBoost.predict",
    }
    if not required_method_links.issubset(level_five_parser.hrefs):
        errors.append("侧边栏第 5 层未同时生成 XGBoost.fit() 与 predict() 方法入口")

    if "hscredit.core.eda.feature_summary" not in eda_html:
        errors.append("feature_summary 的 API 页面锚点缺失")
    parsed_search_index = _load_search_index(search_index, errors)
    if not _has_feature_summary_result(parsed_search_index):
        errors.append("搜索结果未包含可跳转到 API 页面锚点的 feature_summary")

    if "数据库连接、流式读写与表结构导出" not in database_html:
        errors.append("数据库用户指南标题或正文缺失")
    if "大 JSON 字段按路径读取" not in database_html or "json_fields" not in database_html:
        errors.append("数据库用户指南缺少大 JSON 字段投影说明")
    if "hscredit.database.client.Database" not in database_api_html:
        errors.append("数据库 API 页面缺少 Database 公共类锚点")
    required_database_methods = (
        "hscredit.database.client.Database.stream_query",
        "hscredit.database.stream.QueryStream.to_result",
    )
    if not all(method in database_api_html for method in required_database_methods):
        errors.append("数据库 API 页面必须包含 stream_query 与 QueryStream.to_result 方法锚点")
    if 'href="database.html"' not in tooling_html or ">数据库" not in tooling_html:
        errors.append("数据库 API 导航未进入报告与工具侧边栏")
    if not _has_database_result(parsed_search_index):
        errors.append("Database 搜索结果未包含可跳转的 API 页面锚点")

    defines_chinese = re.search(r"(?:class|function|var|let|const)\s+ChineseStemmer\b", language_data)
    references_chinese = re.search(r"(?:window\s*\.\s*)?Stemmer\s*=\s*ChineseStemmer\b", language_data)
    if references_chinese and not defines_chinese:
        errors.append("搜索运行时引用了未定义的 ChineseStemmer")
    if not re.search(r"(?:var|let|const)\s+Stemmer\s*=|window\s*\.\s*Stemmer\s*=", language_data):
        errors.append("搜索运行时没有提供 Stemmer")

    css_rules = re.findall(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", custom_css)
    base_selector = ".wy-menu-vertical a"
    current_selector = ".wy-menu-vertical li.current > a"
    scrollspy_selector = ".wy-menu-vertical li.hs-scrollspy-current > a"
    base_bodies = [
        body
        for selectors, body in css_rules
        if base_selector in {_normalize_selector(selector) for selector in selectors.split(",")}
    ]
    has_border_box = any(re.search(r"box-sizing\s*:\s*border-box", body) for body in base_bodies)
    if not has_border_box or not _has_effective_stable_border(
        css_rules,
        lambda selector: selector == base_selector,
    ):
        errors.append("普通侧边栏链接未预留透明 3px 左边框")

    current_matching_selectors = {
        base_selector,
        current_selector,
        scrollspy_selector,
        ".wy-menu-vertical li.current a",
        ".wy-menu-vertical li.toctree-l3.current > a",
        ".wy-menu-vertical li.toctree-l2.current li.toctree-l3 > a",
        ".wy-menu-vertical li.current > ul li.hs-scrollspy-current > a",
    }
    if not all(
        _has_effective_stable_border(css_rules, matches)
        for matches in (
            lambda selector: selector == current_selector,
            lambda selector: selector == scrollspy_selector,
            lambda selector: selector in current_matching_selectors,
        )
    ):
        errors.append("当前侧边栏链接的最终样式未保留透明 3px 左边框")

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行校验并返回适合 CI 使用的退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", type=Path, default=Path("docs/_build/html"))
    args = parser.parse_args(argv)

    errors = collect_validation_errors(args.build_dir)
    errors.extend(collect_search_runtime_errors(args.build_dir))
    if errors:
        for error in errors:
            print(f"文档校验失败：{error}")
        return 1

    print("文档导航与搜索产物校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
