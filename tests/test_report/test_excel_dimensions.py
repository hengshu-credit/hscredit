"""Excel 行高及合并单元格列宽的保存后布局契约。"""

from copy import copy

import pytest
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from hscredit.excel import ExcelWriter


@pytest.fixture
def sheet():
    writer = ExcelWriter()
    worksheet = writer.get_sheet_by_name("布局")
    yield writer, worksheet
    writer.workbook.close()


def put(worksheet, coordinate, value, *, size=10, wrap=False):
    cell = worksheet[coordinate]
    cell.value = value
    cell.font = Font(size=size)
    cell.alignment = Alignment(wrap_text=wrap)
    return cell


def test_row_height_preserves_styles_values_and_row_properties(sheet, tmp_path):
    writer, ws = sheet
    cell = put(ws, "B2", "第一行\n第二行", size=16)
    cell.fill = PatternFill(fill_type="solid", fgColor="FFCC00")
    cell.number_format = "@"
    ws.row_dimensions[2].hidden = True
    ws.row_dimensions[2].outlineLevel = 2
    ws.row_dimensions[2].fill = PatternFill(fill_type="solid", fgColor="FFFFFF")
    cell_style, row_style = copy(cell._style), copy(ws.row_dimensions[2]._style)
    columns = {name: copy(dimension) for name, dimension in ws.column_dimensions.items()}
    writer.set_row_height(ws, row=3, height=35)
    writer.adjust_rows_height(ws, rows=2)
    assert cell._style == cell_style
    path = tmp_path / "行高样式.xlsx"
    writer.save(path)
    loaded = load_workbook(path)
    saved = loaded["布局"]
    assert saved.row_dimensions[2].height == pytest.approx(40.4)
    assert saved.row_dimensions[3].height == 35
    assert saved.row_dimensions[2].hidden
    assert saved.row_dimensions[2].outlineLevel == 2
    assert saved.row_dimensions[2]._style == row_style
    assert saved["B2"].font == copy(cell.font)
    assert saved["B2"].fill == copy(cell.fill)
    assert saved["B2"].number_format == cell.number_format
    assert not saved["B2"].alignment.wrap_text
    assert saved["B2"].value == "第一行\n第二行"
    assert {name: dimension.width for name, dimension in saved.column_dimensions.items()} == {name: dimension.width for name, dimension in columns.items()}
    loaded.close()


@pytest.mark.parametrize(
    "options,selected",
    [({"rows": 3}, {3}), ({"rows": [2, 4]}, {2, 4}), ({"start_row": 2, "end_row": 3}, {2, 3}), ({}, {2, 3, 4})],
)
def test_row_selection_preserves_other_rows_and_automatic_blank_rows(sheet, options, selected):
    writer, ws = sheet
    for row in range(2, 6):
        ws.row_dimensions[row].height = 55
        if row < 5:
            put(ws, f"B{row}", "第一行\n第二行")
    writer.adjust_rows_height(ws, **options)
    for row in range(2, 6):
        assert ws.row_dimensions[row].height == (26 if row in selected else 55)


def test_row_height_uses_fonts_newlines_and_limits_without_enabling_wrap(sheet):
    writer, ws = sheet
    put(ws, "B2", "a\r\nb\r\n", size=10)
    put(ws, "C2", "短文本", size=30)
    put(ws, "B3", "a\nb", size=10)
    put(ws, "B4", "不换行的长文本" * 30)
    writer.adjust_rows_height(ws, start_row=2, end_row=5, min_height=10, max_height=30, extra_padding=5)
    assert [ws.row_dimensions[row].height for row in range(2, 6)] == [30, 29, 17, 10]
    assert not ws["B4"].alignment.wrap_text


def test_row_height_uses_template_column_ranges_and_horizontal_merge_width(sheet, tmp_path):
    writer, ws = sheet
    ws.column_dimensions["C"].width = 8  # 模板 C:XFD 范围的继承宽度。
    put(ws, "D2", "abcdefghijklmn", wrap=True)
    ws.merge_cells("C3:E3")
    put(ws, "C3", "abcdefghijklmn", wrap=True)
    ws.merge_cells("C4:C5")
    put(ws, "C4", "跨行合并\n内容\n不分摊", wrap=True)
    ws.row_dimensions[4].height = 45
    ws.row_dimensions[5].height = 45
    writer.set_column_width(ws, "E", 14)
    put(ws, "E6", "abcdefghijklmn", wrap=True)
    dimensions = {key: (dim.min, dim.max, dim.width) for key, dim in ws.column_dimensions.items()}
    writer.adjust_rows_height(ws)
    assert {key: (dim.min, dim.max, dim.width) for key, dim in ws.column_dimensions.items()} == dimensions
    path = tmp_path / "合并行高.xlsx"
    writer.save(path)
    loaded = load_workbook(path)
    heights = loaded["布局"].row_dimensions
    assert heights[2].height == 38
    assert heights[3].height == 14
    assert heights[4].height == heights[5].height == 45
    assert heights[6].height == 26
    loaded.close()


def test_wrapped_row_height_uses_displayed_numbers_without_changing_precision(sheet):
    writer, ws = sheet
    writer.set_column_width(ws, "B", 10)
    value = 0.12345678901234567
    cell = put(ws, "B2", value, wrap=True)
    cell.number_format = "0.00%"
    put(ws, "B3", str(value), wrap=True)
    writer.adjust_rows_height(ws)
    assert ws.row_dimensions[2].height == 14
    assert ws.row_dimensions[3].height == 38
    assert cell.value == value
    assert cell.number_format == "0.00%"


@pytest.mark.parametrize(
    "options",
    [
        {"rows": 0},
        {"rows": [2, -1]},
        {"rows": True},
        {"rows": 1.5},
        {"rows": "2"},
        {"start_row": 2},
        {"end_row": 2},
        {"start_row": 3, "end_row": 2},
        {"rows": [2], "start_row": 1, "end_row": 3},
        {"rows": 1_048_577},
        {"min_height": 0},
        {"min_height": 30, "max_height": 20},
        {"max_height": float("inf")},
        {"extra_padding": -1},
    ],
)
def test_invalid_row_options_fail_before_changing_heights(sheet, options):
    writer, ws = sheet
    put(ws, "B2", "内容")
    ws.row_dimensions[2].height = 45
    with pytest.raises(ValueError):
        writer.adjust_rows_height(ws, **options)
    assert ws.row_dimensions[2].height == 45


@pytest.mark.parametrize("speed", ["normal", "fast"])
def test_merged_column_deficit_is_shared_in_dataframe_writes(sheet, speed, tmp_path):
    import pandas as pd

    writer, ws = sheet
    group = "abcdefghijklmnopqrstuvwxyz1234"  # 30 个字符，需要的总宽度为 38。
    data = pd.DataFrame([["abcdefghij", "x"]], columns=pd.MultiIndex.from_tuples([(group, "甲"), (group, "乙")]))
    writer.insert_df2sheet(ws, data, "B2", auto_width=True, speed=speed)
    path = tmp_path / f"合并列宽-{speed}.xlsx"
    writer.save(path)
    loaded = load_workbook(path)
    saved = loaded["布局"]
    assert saved.column_dimensions["B"].width == pytest.approx(23.95)
    assert saved.column_dimensions["C"].width == pytest.approx(14.05)
    assert str(next(iter(saved.merged_cells.ranges))) == "B2:C2"
    assert saved["B2"].value == group
    loaded.close()


@pytest.mark.parametrize("selected_col,fixed_col", [("B", "C"), ("C", "B")])
def test_merged_column_width_respects_unselected_columns_and_caps(sheet, selected_col, fixed_col):
    writer, ws = sheet
    writer.set_column_width(ws, fixed_col, 10)
    ws.merge_cells("B1:C1")
    put(ws, "B1", "abcdefghijklmnopqrst")  # 总宽度 26。
    put(ws, "B2", "短")
    writer.adjust_columns_width(ws, columns=[selected_col], max_width=25)
    assert ws.column_dimensions[selected_col].width == pytest.approx(16)
    assert ws.column_dimensions[fixed_col].width == 10
    writer.adjust_columns_width(ws, start_col=selected_col, end_col=selected_col, max_width=12)
    assert ws.column_dimensions[selected_col].width == 12
    assert ws.column_dimensions[fixed_col].width == 10


def test_merged_column_width_redistributes_after_a_column_reaches_cap(sheet):
    writer, ws = sheet
    ws.merge_cells("B1:C1")
    put(ws, "B1", "a" * 35)  # 总宽度 44。
    ws.merge_cells("B2:B3")
    put(ws, "B2", "a" * 15)  # 单列基础宽度 20。
    put(ws, "C2", "短")
    writer.adjust_columns_width(ws, columns=[2, 3], max_width=25)
    assert ws.column_dimensions["B"].width == 25
    assert ws.column_dimensions["C"].width == pytest.approx(19)


@pytest.mark.parametrize(
    "width_options,height_options,expected_width,expected_height",
    [
        ({}, {}, 3.2, 14),
        ({"min_width": None}, {"min_height": None}, 3.2, 14),
        ({"min_width": 8}, {"min_height": 15}, 8, 15),
        ({"min_width": 2}, {"min_height": 10}, 3.2, 14),
    ],
)
def test_minimum_dimensions_only_apply_when_explicitly_requested(sheet, tmp_path, width_options, height_options, expected_width, expected_height):
    writer, ws = sheet
    put(ws, "B2", "x")
    writer.adjust_columns_width(ws, columns="B", **width_options)
    writer.adjust_rows_height(ws, rows=2, **height_options)
    path = tmp_path / "可选下限.xlsx"
    writer.save(path)
    loaded = load_workbook(path)
    assert loaded["布局"].column_dimensions["B"].width == pytest.approx(expected_width)
    assert loaded["布局"].row_dimensions[2].height == pytest.approx(expected_height)
    loaded.close()


def test_explicit_empty_dimensions_keep_sizes_without_minimum(sheet):
    writer, ws = sheet
    writer.set_column_width(ws, "C", 23)
    writer.set_row_height(ws, 3, 37)
    writer.adjust_columns_width(ws, columns="C")
    writer.adjust_rows_height(ws, rows=3)
    assert ws.column_dimensions["C"].width == 23
    assert ws.row_dimensions[3].height == 37
    writer.adjust_columns_width(ws, columns="C", min_width=8)
    writer.adjust_rows_height(ws, rows=3, min_height=15)
    assert ws.column_dimensions["C"].width == 8
    assert ws.row_dimensions[3].height == 15
