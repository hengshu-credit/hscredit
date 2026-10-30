"""检查实际 API HTML 中参数说明是否为空或重复，不依赖浏览器和第三方 HTML 库。"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path


class _Element:
    def __init__(self, tag="", attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def elements(self, tag=None):
        return [child for child in self.children if isinstance(child, _Element) and (tag is None or child.tag == tag)]

    def walk(self):
        yield self
        for child in self.elements():
            yield from child.walk()

    def text(self):
        return "".join(child.text() if isinstance(child, _Element) else child for child in self.children)

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Element()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _Element(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Element(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parameter_docs(html):
    """返回各公开对象的参数名称与实际渲染说明，保留重复项供检查。"""
    parser = _Parser()
    parser.feed(html)
    result = {}
    for node in parser.root.walk():
        if node.tag != "dl" or not node.has_class("py"):
            continue
        signatures, bodies = node.elements("dt"), node.elements("dd")
        if not signatures or not bodies:
            continue
        name = signatures[0].attrs.get("id", "")
        if not name.startswith("hscredit."):
            continue
        fields = []
        for field_list in bodies[0].elements("dl"):
            if not field_list.has_class("field-list"):
                continue
            label = ""
            for field in field_list.elements():
                if field.tag == "dt":
                    label = field.text().strip().rstrip(":：")
                elif field.tag == "dd" and label in {"参数", "Parameters"}:
                    for paragraph in field.walk():
                        if paragraph.tag != "p":
                            continue
                        strong = paragraph.elements("strong")
                        if not strong or paragraph.children[0] is not strong[0]:
                            continue
                        parameter = strong[0].text().lstrip("*")
                        text = paragraph.text()
                        description = text.split(" -- ", 1)[1].strip() if " -- " in text else ""
                        fields.append((parameter, description))
        result[name] = fields
    return result


def collect_parameter_doc_errors(build_dir):
    errors = []
    pages = sorted((Path(build_dir) / "api").glob("*.html"))
    if not pages:
        return ["未找到 API HTML 页面"]
    for page in pages:
        for name, fields in parameter_docs(page.read_text(encoding="utf-8")).items():
            seen = set()
            for parameter, description in fields:
                if not description:
                    errors.append(f"{name}.{parameter}: 缺少参数说明")
                if parameter in seen:
                    errors.append(f"{name}.{parameter}: 参数重复展示")
                seen.add(parameter)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    errors = collect_parameter_doc_errors(args.build_dir)
    for error in errors:
        print(error)
    print(f"API 参数说明检查：{len(errors)} 个问题")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
