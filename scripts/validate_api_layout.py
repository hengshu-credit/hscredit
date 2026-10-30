"""验证生成 HTML 中 API 对象的目录父子关系与参数标记。"""

from html.parser import HTMLParser
from pathlib import Path


class ApiLayoutParser(HTMLParser):
    """对照正文对象的嵌套结构检查本页目录，不依赖特定类名。"""

    def __init__(self):
        super().__init__()
        self.lists = []
        self.descriptions = []
        self.links = {}
        self.objects = {}
        self.page_levels = set()
        self.field_depth = 0
        self.code_depth = 0
        self.unrendered = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if tag == "li":
            self.lists.append(
                (next((int(c[9:]) for c in classes if c.startswith("toctree-l")), None), "current" in classes)
            )
        elif tag == "a" and self.lists and self.lists[-1][0]:
            level, current = self.lists[-1]
            href = attrs.get("href", "")
            if href == "#" and current:
                self.page_levels.add(level)
            if href.startswith("#hscredit."):
                self.links[href[1:]] = level
        elif tag == "dl":
            self.descriptions.append(classes)
            if "field-list" in classes:
                self.field_depth += 1
        elif tag == "dt" and attrs.get("id", "").startswith("hscredit."):
            ancestors = sum("class" in cs or "exception" in cs for cs in self.descriptions[:-1])
            self.objects[attrs["id"]] = 1 + ancestors
        elif tag in {"pre", "code"}:
            self.code_depth += 1

    def handle_endtag(self, tag):
        if tag == "li" and self.lists:
            self.lists.pop()
        elif tag == "dl" and self.descriptions:
            if "field-list" in self.descriptions.pop():
                self.field_depth -= 1
        elif tag in {"pre", "code"}:
            self.code_depth -= 1

    def handle_data(self, data):
        if self.field_depth and not self.code_depth and "``" in data:
            self.unrendered.append(data.strip())


def collect_api_layout_errors(build_dir):
    """逐页核对 API 类、函数及成员层级，拒绝未解析的参数双反引号。"""
    errors = []
    for path in sorted((Path(build_dir) / "api").glob("*.html")):
        parser = ApiLayoutParser()
        parser.feed(path.read_text(encoding="utf-8"))
        if parser.objects:
            if len(parser.page_levels) != 1:
                errors.append(f"{path.name}：无法确定唯一的 API 页面目录入口层级：{sorted(parser.page_levels)}")
            else:
                page_level = next(iter(parser.page_levels))
                for anchor, offset in parser.objects.items():
                    expected = page_level + offset
                    actual = parser.links.get(anchor)
                    if actual != expected:
                        errors.append(f"{path.name}：{anchor} 应位于第 {expected} 层目录，实际为 {actual}")
        for text in parser.unrendered:
            errors.append(f"{path.name}：参数标记未解析：{text[:120]}")
    return errors
