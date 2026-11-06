"""将数据集说明中提供下载的工作簿呈现为静态网页。"""

import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries


def _load_report(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book["特征摘要"]
        cells = iter(sheet.rows)
        next(cells)  # 工作簿标题
        notes = [cell.value for _ in range(2) for cell in next(cells) if cell.value]
        groups = []
        for cell in next(cells):
            if cell.value:
                groups.append([cell.value, 0])
            groups[-1][1] += 1
        headers = [c.value for c in next(cells)]
        first = next(cells)
        formats = [c.number_format or "General" for c in first]
        rows = [[c.value for c in first]]
        rows.extend([c.value for c in row] for row in cells)
        sheet_part = sheet._worksheet_path
    finally:
        book.close()

    # openpyxl 的只读单元格不含迷你图。按 Excel 的实际目标和引用取数，
    # 保留报告中排序后的行序，也适用于引用其他行的折线。
    with ZipFile(path) as archive, archive.open(sheet_part) as xml:
        for _, node in ElementTree.iterparse(xml, events=("end",)):
            if node.tag.endswith("}sparkline"):
                target_row, target_col = coordinate_to_tuple(node.findtext("{*}sqref"))
                region = node.findtext("{*}f").rsplit("!", 1)[-1]
                left, top, right, bottom = range_boundaries(region)
                if top != bottom:
                    raise ValueError(f"数据分布迷你图应引用同一行：{region}")
                rows[target_row - 6][target_col - 1] = rows[top - 6][left - 1:right]
                node.clear()
            elif node.tag.endswith("}row"):
                node.clear()
    return dict(notes=notes, groups=groups, headers=headers, formats=formats, rows=rows)


def _render_report(app, pagename, templatename, context, doctree):
    if pagename != "dataset-summary":
        return
    path = Path(app.srcdir) / "assets" / "datasets" / "hscredit-summary.xlsx"
    report = json.dumps(_load_report(path), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    context["body"] += '<script type="application/json" id="feature-report-data">' + report.replace("<", "\\u003c") + "</script>"
    app.add_css_file("dataset-summary.css")
    app.add_js_file("dataset-summary.js", loading_method="defer")


def setup(app):
    app.connect("html-page-context", _render_report)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
