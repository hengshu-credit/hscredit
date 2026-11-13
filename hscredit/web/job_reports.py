"""任务管理模块的 Excel 报告导出和分页预览。"""

import json
import mimetypes
import re
from bisect import bisect_right
from contextlib import closing
from datetime import datetime, timedelta
from functools import lru_cache
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.drawing.image import Image
from openpyxl.packaging.relationship import get_dependents, get_rels_path
from openpyxl.utils import get_column_letter, range_boundaries

from hscredit.excel import ExcelWriter, dataframe2excel


def scatter_rows(axes, points):
    rows = []
    for point in points:
        row = {}
        for index, (axis, value) in enumerate(zip(axes, point["values"]), 1):
            if "categories" in axis:
                value = axis["categories"][int(value)]
            elif axis["kind"] == "date":
                value = (datetime(1970, 1, 1) + timedelta(milliseconds=value)).isoformat() + "Z"
            row[f"坐标{index} · {axis['name']}"] = value
        rows.append({**row, "分组标签": point["label"]})
    return rows


def result_tables(result):
    """从任务已发布的结果取数，兼容早期只保存 JSON 的任务。"""
    yield from result.get("tables", {}).items()
    if "exploration" in result:
        eda = result["exploration"]
        yield "探索概况", [{"字段": eda.get("feature"), "样本数": eda.get("rows"), "有效数": eda.get("valid"), "缺失数": eda.get("missing"), "无效数": eda.get("invalid"), "分布说明": eda.get("distribution_note")}]
        yield "样本分布", eda.get("distribution", [])
        if "correlation" in eda:
            matrix = eda["correlation"]
            label = "字段名"
            while label in matrix.get("columns", []):
                label = "_" + label
            yield "相关性矩阵", [{label: name, **dict(zip(matrix["columns"], row))} for name, row in zip(matrix.get("columns", []), matrix.get("values", []))]
        if eda.get("scatter"):
            yield "散点样本", [dict(zip([eda["feature"], eda["other"]], point)) for point in eda["scatter"]]
    if "interaction" in result:
        data = result["interaction"]
        yield "交互概况", [{"样本数": data["total"], "共同有效样本数": data["valid"], "抽样行数": data["sampled"], "统计说明": data["note"]}]
        for dist in data["distributions"]:
            yield dist["feature"] + "分布", dist["distribution"]
        axes = data["axes"]
        for pair in data["pairs"]:
            pair_axes = [axes[i] for i in pair["indices"]]
            x, y = [axis["name"] for axis in pair_axes]
            if pair.get("means"):
                yield f"{x} × {y} 分组均值", pair["means"]
            yield f"{x} × {y} 密度", [{"横轴区间": row["x"], "纵轴区间": row["y"], "标签": row["标签"], "样本数": row["样本数"]} for row in pair["density"]]
            if len(axes) > 2:
                yield f"{x} × {y} 散点", scatter_rows(pair_axes, pair["points"])
        yield "散点样本", scatter_rows(axes, data["points"])
    if result.get("candidates"):
        yield "候选规则", [{"规则名称": r["name"], "规则表达式": r["expression"]} for r in result["candidates"]]


def _report_sheet(writer, title):
    base = re.sub(r"[\\/*?:\[\]]", " ", ILLEGAL_CHARACTERS_RE.sub("", title)).strip(" '") or "分析结果"
    titles = {name.casefold() for name in writer.workbook.sheetnames}
    name = base[:31]
    i = 1
    while name.casefold() in titles:
        i += 1
        suffix = f" ({i})"
        name = base[: 31 - len(suffix)] + suffix
    return writer.get_sheet_by_name(name)


def _report_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value


def _write_result_table(writer, ws, rows):
    frame = pd.DataFrame([{key: _report_value(value) for key, value in row.items()} for row in rows], dtype=object)
    left_cols, right_cols = [], []
    for position, (_, column) in enumerate(frame.items()):
        kind = pd.api.types.infer_dtype(column, skipna=True)
        if kind in {"integer", "floating", "mixed-integer-float", "empty", "decimal"}:
            right_cols.append(position)
        else:
            left_cols.append(position)
    frame.columns = [_report_value(key) for key in frame.columns]
    dataframe2excel(frame, writer, sheet_name=ws, start_row=1, start_col=1, index=False, decimal=None, left_cols=left_cols, right_cols=right_cols, auto_width=True)
    # 任务结果中的公式样文本和 Excel 错误码均是原始字段值。
    for row in ws.iter_rows():
        for cell in row:
            if cell.data_type in {"f", "e"}:
                cell.data_type = "s"
    ws.freeze_panes = "A2"


def write_result_workbook(directory, result, filename="report.xlsx"):
    """为缺少原生工作簿的分析保存表格和图像，以原子替换发布。"""
    destination = directory / filename
    with ExcelWriter().set_filename(str(destination)) as writer:
        for title, rows in result_tables(result):
            if rows:
                _write_result_table(writer, _report_sheet(writer, title), rows)
        for picture in result.get("images", []):
            ws = _report_sheet(writer, picture["title"])
            writer.insert_value2sheet(ws, "A1", _report_value(picture["title"]), style="header", auto_width=True)
            ws["A1"].data_type = "s"
            ws.add_image(Image(str(directory / picture["file"])), "A3")
        if not any(ws is not writer.style_sheet for ws in writer.workbook.worksheets):
            writer.insert_value2sheet(_report_sheet(writer, "分析结果"), "A1", "本次任务没有表格数据", auto_width=True)
    return destination


def sheet_images(book, ws):
    """按包内关系登记图片，保留原图下载序号和工作表锚点。"""
    archive = book._archive

    def targets(part, kind):
        relations = get_rels_path(part)
        if relations in archive.NameToInfo:
            for relation in get_dependents(archive, relations):
                if relation.Type.endswith("/" + kind) and relation.TargetMode != "External":
                    yield relation.Id, relation.target

    images = []
    for _, drawing in targets(ws._worksheet_path, "drawing"):
        anchors = {}
        for anchor in ElementTree.fromstring(archive.read(drawing)):
            for picture in anchor.findall(".//{*}pic"):
                blip = picture.find(".//{*}blip")
                if blip is not None:
                    embed = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                    row = int(anchor.findtext("{*}from/{*}row", "0")) + 1
                    col = int(anchor.findtext("{*}from/{*}col", "0")) + 1
                    anchors.setdefault(embed, []).append((row, col))
        for rid, image in targets(drawing, "image"):
            if image.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                images.append((image, anchors.get(rid, [(1, 1)])))
    return images


def workbook_image(path, sheet, index):
    with closing(load_workbook(path, read_only=True)) as book:
        if sheet not in book.sheetnames:
            raise ValueError("工作表不存在")
        images = sheet_images(book, book[sheet])
        if not 0 <= index < len(images):
            raise ValueError("工作表图片不存在")
        image, _ = images[index]
        return book._archive.read(image), mimetypes.guess_type(image)[0]


def cell_text(cell):
    value = cell.value
    if value is None:
        return ""
    if isinstance(value, datetime):
        fmt = (cell.number_format or "").lower()
        date_formats = {"yyyy-mm-dd": "%Y-%m-%d", "yyyy/mm/dd": "%Y/%m/%d", "yyyy-mm": "%Y-%m", "yyyy-mm-dd hh:mm": "%Y-%m-%d %H:%M", "yyyy-mm-dd hh:mm:ss": "%Y-%m-%d %H:%M:%S"}
        if fmt in date_formats:
            return value.strftime(date_formats[fmt])
    return ExcelWriter._format_value_for_width(value, cell.number_format or "General")


@lru_cache(maxsize=48)
def _sheet_layout(path, modified, size, part, bold_styles):
    """扫描一次不可变的报告工作表，记录章节位置、合并区域及内部链接。"""
    merges, links, headings, continuations = [], {}, [], {}
    data_starts, data_ends = [], []
    header = None
    previous_row = 0
    with ZipFile(path) as archive, archive.open(part) as source:
        rels = get_rels_path(part)
        destinations = {rel.Id: rel.target for rel in get_dependents(archive, rels)} if rels in archive.NameToInfo else {}
        sheet_data = None
        for event, node in ElementTree.iterparse(source, events=("start", "end")):
            if event == "start":
                if node.tag.endswith("}sheetData"):
                    sheet_data = node
                continue
            if node.tag.endswith("}row"):
                # 只记录独占一行的加粗文字，避免把多列表头或数据行误列为章节。
                populated = [cell for cell in node if len(cell)]
                row = int(node.get("r", previous_row + 1))
                bold_text = bool(populated) and all(cell.get("s") in bold_styles and cell.get("t") in {"s", "inlineStr"} for cell in populated)
                # 横向窗口里可能整行为空，仍须保留原表的数据行和真正的章节边界。
                if populated and not (bold_text and len(populated) == 1):
                    if data_ends and row == data_ends[-1] + 1:
                        data_ends[-1] = row
                    else:
                        data_starts.append(row)
                        data_ends.append(row)
                if not populated or row != previous_row + 1:
                    header = None
                # 跨越整页的加粗区域按普通行分页，避免单个窗口重复读取无界行数。
                if header and header[1] - header[0] < 49 and (row - 1) % 50 == 0 and (not bold_text or header[1] == row - 1):
                    continuations[row] = header
                if bold_text and len(populated) == 1:
                    cell = populated[0]
                    label = int(cell.findtext("{*}v")) if cell.get("t") == "s" else "".join(t.text or "" for t in cell.iterfind(".//{*}t"))
                    if cell.get("r"):
                        headings.append((cell.attrib["r"], label))
                if bold_text:
                    # 独占的标题行不作为跨页数据的列头。多行表头记录其连续行范围。
                    if header and header[1] == row - 1:
                        header = (header[0], row)
                    else:
                        header = (row, row) if len(populated) > 1 else None
                previous_row = row
                if sheet_data is not None:
                    sheet_data.clear()
            elif node.tag.endswith("}mergeCell"):
                merges.append(range_boundaries(node.attrib["ref"]))
            elif node.tag.endswith("}hyperlink"):
                rid = node.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                location = node.get("location") or destinations.get(rid, "")
                if node.get("location") or location.startswith("#"):
                    match = re.fullmatch(r"#?(?:'((?:[^']|'')+)'|([^!]+))!\$?([A-Za-z]+)\$?([1-9]\d*)", location)
                    if match:
                        name = (match[1] or match[2]).replace("''", "'")
                        links[node.attrib["ref"]] = (name, match[3].upper(), int(match[4]))
    return merges, links, headings, continuations, data_starts, data_ends


def _page_cells(ws, bounds, merges, links, sheets, heading_addresses):
    left, top, right, bottom = bounds
    visible = [r for r in merges if r[0] <= right and r[2] >= left and r[1] <= bottom and r[3] >= top]
    anchors = {(r[1], r[0]): r for r in visible}
    start_row = min([top, *(r[1] for r in visible)])
    start_col = min([left, *(r[0] for r in visible)])
    rows = [[""] * (right - left + 1) for _ in range(bottom - top + 1)]
    cells = []
    # 跨页合并单元格从其原始左上角读取一次，并裁切为当前窗口中的跨度。
    for row in ws.iter_rows(min_row=start_row, min_col=start_col, max_row=bottom, max_col=right):
        for cell in row:
            if cell.value is None:
                continue
            r, c = cell.row, cell.column
            extent = anchors.get((r, c))
            if (r < top or c < left) and extent is None:
                continue
            value = cell_text(cell)
            if r >= top and c >= left:
                rows[r - top][c - left] = value
            y, x = max(r, top), max(c, left)
            item = {"row": y, "col": x, "text": value, "numeric": cell.data_type == "n", "bold": bool(cell.font and cell.font.b), "heading": cell.coordinate in heading_addresses, "rowspan": min(bottom, extent[3]) - y + 1 if extent else 1, "colspan": min(right, extent[2]) - x + 1 if extent else 1}
            horizontal = cell.alignment.horizontal
            if horizontal in {"left", "center", "right"}:
                item["align"] = horizontal
            if isinstance(cell.value, float) and cell.number_format == "General" and len(value) > 14:
                item["display"] = format(cell.value, ".10g")
            if extent:
                item["merge"] = extent
            link = links.get(cell.coordinate)
            if link and link[0] in sheets:
                item["link"] = {"sheet": link[0], "col": link[1], "row": link[2]}
            cells.append(item)
    return rows, cells


def workbook_page(path, sheet, page, column_page=1, *, first_col=None, column_count=30, include_catalog=True):
    """读取选中工作表的单元格窗口，保留原始行列坐标。"""
    with closing(load_workbook(path, read_only=True, data_only=True)) as book:
        if sheet is not None and sheet not in book.sheetnames:
            raise ValueError("工作表不存在")
        ws = book[sheet or book.sheetnames[0]]
        if ws.max_row is None or ws.max_column is None:
            ws.calculate_dimension(force=True)
        images = sheet_images(book, ws)
        pictures = [{"index": index, "row": row, "col": col} for index, (_, anchors) in enumerate(images) for row, col in anchors]
        height = max([ws.max_row or 1, *(pic["row"] for pic in pictures)])
        width = max([ws.max_column or 1, *(pic["col"] for pic in pictures)])
        row_start = (page - 1) * 50 + 1
        col_start = first_col if first_col is not None else (column_page - 1) * 30 + 1
        if row_start > height or col_start > width:
            raise ValueError("页码超出工作表范围")
        col_end = min(width, col_start + column_count - 1)
        row_end = min(height, row_start + 49)
        stat = path.stat()
        bold_styles = frozenset(str(i) for i, style in enumerate(book._cell_styles) if book._fonts[style.fontId].b)
        merges, links, headings, continuations, data_starts, data_ends = _sheet_layout(path, stat.st_mtime_ns, stat.st_size, ws._worksheet_path, bold_styles)
        outline = []
        for address, label in headings if include_catalog else ():
            text = ws._shared_strings[label] if isinstance(label, int) else label
            col, row, _, _ = range_boundaries(address)
            if text:
                outline.append({"text": text, "row": row, "col": get_column_letter(col)})
        heading_addresses = {address for address, _ in headings}
        rows, cells = _page_cells(ws, (col_start, row_start, col_end, row_end), merges, links, book.sheetnames, heading_addresses)
        context = None
        if row_start in continuations:
            top, bottom = continuations[row_start]
            _, header_cells = _page_cells(ws, (col_start, top, col_end, bottom), merges, links, book.sheetnames, set())
            context = {"first_row": top, "last_row": bottom, "cells": header_cells}
        data_rows = []
        if col_start > 1 or col_end < width:
            for row in range(row_start, row_end + 1):
                i = bisect_right(data_starts, row) - 1
                if i >= 0 and row <= data_ends[i]:
                    data_rows.append(row)
        return {"sheets": book.sheetnames if include_catalog else [], "sheet": ws.title, "height": height, "width": width, "first_row": row_start, "first_col": col_start, "image_count": len(images), "columns": [get_column_letter(i) for i in range(col_start, col_end + 1)], "rows": rows, "cells": cells, "data_rows": data_rows, "header_context": context, "outline": outline, "pictures": [p for p in pictures if row_start <= p["row"] <= row_end and col_start <= p["col"] <= col_end]}
