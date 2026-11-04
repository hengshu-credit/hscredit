"""Excel 列按父子层级合并的边界、顺序和保存后结果。"""

from copy import copy
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import pytest
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from hscredit.excel import ExcelWriter, dataframe2excel


def export(tmp_path, data, entrypoint, speed, **kwargs):
    writer = ExcelWriter(theme_color=kwargs.pop("theme_color", "2639E9"))
    worksheet = writer.get_sheet_by_name("分组")
    merge = kwargs.pop("merge", True)
    if entrypoint == "direct":
        writer.insert_df2sheet(worksheet, data, "B2", speed=speed, merge=merge, **kwargs)
    else:
        dataframe2excel(data, writer, sheet_name=worksheet, speed=speed, merge=merge, **kwargs)
    path = tmp_path / "层级合并.xlsx"
    writer.save(path)
    writer.workbook.close()
    return load_workbook(path)


def vertical_merges(worksheet):
    return {str(area) for area in worksheet.merged_cells.ranges if area.max_row > area.min_row}


@pytest.mark.parametrize("entrypoint", ["direct", "wrapper"])
@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("multi_columns", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_same_month_at_dataset_boundary_respects_requested_order(tmp_path, entrypoint, speed, multi_columns, independent):
    # 字段在表中的顺序与传入层级相反，避免便捷入口悄悄重排层级。
    months = ["2025-11", "2025-12", "2026-01", "2026-02"]
    data = pd.DataFrame({"数据分组": months + ["2026-02"] + months, "数据集": ["训练集"] * 4 + ["验证集"] + ["测试集"] * 4})
    if multi_columns:
        data.columns = pd.MultiIndex.from_product([["统计详情"], data.columns])
    original = data.copy(deep=True)
    options = {"merge_hierarchy": False} if independent else {}
    book = export(tmp_path, data, entrypoint, speed, merge_column=["数据集", "数据分组"], **options)
    ws = book["分组"]
    first = 2 + data.columns.nlevels
    expected = {f"C{first}:C{first + 3}", f"C{first + 5}:C{first + 8}"}
    if independent:
        expected.add(f"B{first + 3}:B{first + 4}")
    assert vertical_merges(ws) == expected
    assert ws.cell(first + 3, 2).value == "2026-02"
    assert ws.cell(first + 4, 2).value == (None if independent else "2026-02")
    assert ws.cell(first + 4, 3).value == "验证集"
    pd.testing.assert_frame_equal(data, original)
    book.close()


@pytest.mark.parametrize("entrypoint", ["direct", "wrapper"])
@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("selected", [None, ["统计口径", "数据分组", "数据集"], ["统计口径"]])
def test_explicit_hierarchy_overrides_order_and_can_use_unmerged_parents(tmp_path, entrypoint, speed, selected):
    data = pd.DataFrame(
        {
            "数据集": ["训练集"] * 4 + ["验证集"] * 4,
            "数据分组": ["2025-11", "2025-11", "2026-02", "2026-02"] * 2,
            "统计口径": ["订单"] * 8,
        }
    )
    book = export(
        tmp_path,
        data,
        entrypoint,
        speed,
        merge_column=selected,
        merge_hierarchy=["数据集", "数据分组", "统计口径"],
    )
    ws = book["分组"]
    expected = {f"D{first}:D{first + 1}" for first in [3, 5, 7, 9]}
    if selected != ["统计口径"]:
        expected.update({"B3:B6", "B7:B10"})
        expected.update(f"C{first}:C{first + 1}" for first in [3, 5, 7, 9])
    else:
        assert [ws.cell(row, 2).value for row in range(3, 11)] == data["数据集"].tolist()
    assert vertical_merges(ws) == expected
    book.close()


@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("header", [False, True])
def test_all_ancestors_constrain_missing_values_with_index_offsets(tmp_path, speed, header):
    data = pd.DataFrame(
        {"数据集": ["训练", "训练", "验证", "验证", "训练", "训练"], "月份": ["2026-02"] * 6, "口径": [pd.NA] * 6},
        index=pd.MultiIndex.from_product([["订单"], range(6)]),
    )
    data.columns = pd.MultiIndex.from_product([["统计详情"], data.columns])
    book = export(
        tmp_path,
        data,
        "direct",
        speed,
        header=header,
        index=True,
        merge_index=False,
        merge_column=data.columns.tolist(),
    )
    first = 4 if header else 2
    assert vertical_merges(book["分组"]) == {f"{col}{row}:{col}{row + 1}" for col in "DEF" for row in [first, first + 2, first + 4]}
    book.close()


@pytest.mark.parametrize("columns", [[0, 1], ["父级", "子级"]])
def test_empty_and_single_row_tables_do_not_create_vertical_merges(tmp_path, columns):
    for records in [[], [["训练", "2026-02"]]]:
        data = pd.DataFrame(records, columns=["父级", "子级"])
        book = export(tmp_path, data, "wrapper", "fast", merge_column=columns)
        assert vertical_merges(book["分组"]) == set()
        book.close()


@pytest.mark.parametrize(
    "hierarchy,error",
    [(["父级"], ValueError), (["不存在", "子级"], ValueError), ("父级", TypeError)],
)
def test_invalid_hierarchy_is_rejected_before_writing(hierarchy, error):
    writer = ExcelWriter()
    ws = writer.get_sheet_by_name("分组")
    data = pd.DataFrame({"父级": ["训练", "验证"], "子级": ["2026-02", "2026-02"]})
    with pytest.raises(error):
        writer.insert_df2sheet(ws, data, "B2", merge_column=list(data), merge=True, merge_hierarchy=hierarchy)
    assert all(cell.value is None for row in ws for cell in row)
    writer.workbook.close()


@pytest.mark.parametrize("entrypoint", ["direct", "wrapper"])
@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("hierarchy", [True, False])
@pytest.mark.parametrize("header", [True, False])
def test_fill_merge_borders_follow_group_ranges_and_keep_outer_weight(tmp_path, entrypoint, speed, hierarchy, header):
    data = pd.DataFrame({"数据集": ["训练"] * 4 + ["验证"] * 2 + ["测试"], "口径": ["订单"] * 2 + ["金额"] * 5, "数值": range(7)})
    data.columns = pd.MultiIndex.from_product([["统计详情"], data.columns])
    options = dict(merge_column=["数据集", "口径"], merge_hierarchy=hierarchy, fill=True, header=header, theme_color="B42266")
    original = export(tmp_path, data, entrypoint, speed, **options)
    bordered = export(tmp_path, data, entrypoint, speed, merge_border=True, **options)
    before, after = original["分组"], bordered["分组"]
    first = 4 if header else 2
    groups = {
        2: [(0, 4), (4, 6), (6, 7)],
        3: [(0, 2), (2, 4), (4, 6), (6, 7)] if hierarchy else [(0, 2), (2, 7)],
    }
    assert vertical_merges(after) == vertical_merges(before)
    assert before.cell(first, 2).border.bottom.color.rgb[-6:] == "FFFFFF"
    for col, ranges in groups.items():
        for start, end in ranges:
            anchor = after.cell(first + start, col)
            bottom = after.cell(first + end - 1, col)
            assert anchor.border.top.style == ("medium" if start == 0 else "thin")
            assert anchor.border.bottom.style == ("medium" if end == len(data) else "thin")
            assert bottom.border.bottom.style == anchor.border.bottom.style
            assert anchor.border.top.color.rgb[-6:] == "B42266"
            assert anchor.border.bottom.color.rgb[-6:] == "B42266"
            assert bottom.border.bottom.color.rgb[-6:] == "B42266"
    for row in range(2, first + len(data)):
        for col in range(2, 5):
            old, new = before.cell(row, col), after.cell(row, col)
            assert new.value == old.value
            assert new.fill == copy(old.fill)
            assert new.font == copy(old.font)
            assert new.number_format == old.number_format
            assert new.border.left == old.border.left
            assert new.border.right == old.border.right
            if col == 4 or row < first:
                assert new.border == copy(old.border)
    original.close()
    bordered.close()


@pytest.mark.parametrize("entrypoint", ["direct", "wrapper"])
@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("fill,merge", [(False, True), (True, False)])
def test_merge_borders_do_not_change_non_fill_or_unmerged_tables(tmp_path, entrypoint, speed, fill, merge):
    data = pd.DataFrame({"数据集": ["训练", "训练", "验证", "验证"], "值": [1, 2, 3, 4]})
    options = dict(merge_column="数据集", fill=fill, merge=merge)
    original = export(tmp_path, data, entrypoint, speed, merge_border=False, **options)
    bordered = export(tmp_path, data, entrypoint, speed, merge_border=True, **options)
    for row in range(2, 7):
        for col in range(2, 4):
            old, new = original["分组"].cell(row, col), bordered["分组"].cell(row, col)
            assert new.value == old.value
            assert new.border == copy(old.border)
            assert new.fill == copy(old.fill)
    assert vertical_merges(original["分组"]) == vertical_merges(bordered["分组"])
    original.close()
    bordered.close()


def test_single_row_group_has_both_heavy_outer_borders(tmp_path):
    data = pd.DataFrame({"数据集": ["训练"], "值": [1]})
    book = export(tmp_path, data, "direct", "fast", merge_column="数据集", fill=True, header=False, merge_border=True)
    border = book["分组"]["B2"].border
    assert border.top.style == border.bottom.style == "medium"
    assert border.top.color.rgb[-6:] == border.bottom.color.rgb[-6:] == "2639E9"
    book.close()


@pytest.mark.parametrize("end_col", [2, 4])
@pytest.mark.parametrize("side_style", [None, "thin"])
def test_merge_last_corner_does_not_introduce_side_borders_or_reset_anchor_style(tmp_path, end_col, side_style):
    writer = ExcelWriter()
    ws = writer.get_sheet_by_name("边框")
    anchor = ws["B2"]
    anchor.value = 0.123456
    anchor.font = Font(name="宋体", size=14, bold=True)
    anchor.fill = PatternFill(fill_type="solid", fgColor="DEE1FB")
    anchor.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    anchor.number_format = "0.00%"
    side = Side(style=side_style, color="2639E9") if side_style is not None else None
    anchor.border = Border(top=Side(style="medium", color="2639E9"), left=side, right=side)
    if end_col != 2:
        ws.cell(2, end_col).border = Border(right=side)
    bottom = ws.cell(4, end_col)
    bottom.border = Border(
        left=Side(style="thick", color="FFFFFF"),
        right=Side(style="thick", color="FFFFFF"),
        bottom=Side(style="medium", color="2639E9"),
    )
    ws["B4"].border = Border(
        left=bottom.border.left if end_col == 2 else None,
        right=bottom.border.right if end_col == 2 else None,
        bottom=Side(style="medium", color="2639E9"),
    )
    expected_style = {name: copy(getattr(anchor, name)) for name in ["font", "fill", "alignment"]}
    writer.merge_cells(ws, "B2", (end_col, 4))
    for name, expected in expected_style.items():
        assert getattr(anchor, name) == expected
    assert anchor.number_format == "0.00%"
    path = tmp_path / "末行边框.xlsx"
    writer.save(path)
    loaded = load_workbook(path)
    saved = loaded["边框"]
    for row in range(2, 5):
        for edge in [saved.cell(row, 2).border.left, saved.cell(row, end_col).border.right]:
            assert (edge.style if edge is not None else None) == side_style
            if side_style is not None:
                assert edge.color.rgb[-6:] == "2639E9"
    assert saved.cell(4, 2).border.bottom.style == "medium"
    assert saved.cell(4, end_col).border.bottom.color.rgb[-6:] == "2639E9"
    for name, expected in expected_style.items():
        assert getattr(saved["B2"], name) == expected
    assert saved["B2"].number_format == "0.00%"
    assert saved["B2"].value == pytest.approx(0.123456)
    loaded.close()
    writer.workbook.close()


@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("merge_border", [False, True])
def test_shaded_merge_writes_matching_fill_and_side_borders_at_last_cell(tmp_path, speed, merge_border):
    writer = ExcelWriter()
    ws = writer.get_sheet_by_name("边框")
    data = pd.DataFrame({"组": ["训练"] * 3 + ["验证"] * 3, "值": range(6)})
    writer.insert_df2sheet(ws, data, "B2", merge_column="组", merge=True, fill=True, merge_border=merge_border, speed=speed)
    anchor, last = ws["B6"], ws["B8"]
    assert anchor.fill.fgColor.rgb[-6:] != "FFFFFF"
    assert last.fill == copy(anchor.fill)
    assert last.border.left == anchor.border.left
    assert last.border.right == anchor.border.right
    path = tmp_path / "合并填充.xlsx"
    writer.save(path)
    # 直接检查落盘样式：openpyxl 重新加载时会丢弃非首格的填充，不能用重载对象验证这一点。
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {cell.attrib["r"]: cell for cell in root.findall(".//x:sheetData/x:row/x:c", ns)}
        styles = ET.fromstring(archive.read("xl/styles.xml")).find("x:cellXfs", ns)
        for coordinate in ["B7", "B8"]:
            assert styles[int(cells[coordinate].attrib["s"])].attrib["fillId"] == styles[int(cells["B6"].attrib["s"])].attrib["fillId"]
    writer.workbook.close()
