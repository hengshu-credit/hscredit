"""整表格式完成后统一调尺寸，并保持合并边缘及单元格样式。"""

from copy import copy

import pandas as pd
import pytest
from openpyxl import load_workbook
from openpyxl.styles import Border, PatternFill, Side

from hscredit.excel import ExcelWriter, dataframe2excel


@pytest.mark.parametrize("speed", ["normal", "fast"])
def test_dimensions_run_once_after_final_formats_and_before_save(tmp_path, monkeypatch, speed):
    events = []
    inserting = False
    insert = ExcelWriter.insert_df2sheet
    calculate_width = ExcelWriter._calculate_content_width
    adjust_width = ExcelWriter.adjust_columns_width
    adjust_height = ExcelWriter.adjust_rows_height
    save = ExcelWriter.save

    def capture_insert(writer, worksheet, *args, **kwargs):
        nonlocal inserting
        assert kwargs["auto_width"] is False
        inserting = True
        try:
            return insert(writer, worksheet, *args, **kwargs)
        finally:
            inserting = False

    def capture_calculation(writer, *args, **kwargs):
        assert not inserting, "写入过程中不得计算列宽"
        return calculate_width(writer, *args, **kwargs)

    def capture_width(writer, worksheet, **kwargs):
        assert kwargs == {"start_col": 2, "end_col": 4}
        assert worksheet["B5"].number_format == "0.00%"
        assert worksheet["C6"].number_format == "#,##0.00"
        assert worksheet["C6"].alignment.horizontal == "right"
        assert worksheet["D6"].alignment.horizontal == "left"
        assert worksheet["D6"].value == "c\nd\ne"
        assert worksheet.auto_filter.ref == "B4:D6"
        assert len(worksheet.conditional_formatting) == 1
        assert "B2:D2" in {str(area) for area in worksheet.merged_cells.ranges}
        events.append("width")
        return adjust_width(writer, worksheet, **kwargs)

    def capture_height(writer, worksheet, **kwargs):
        assert kwargs == {"start_row": 2, "end_row": 6}
        assert worksheet.column_dimensions["B"].width == pytest.approx(9.2)
        events.append("height")
        return adjust_height(writer, worksheet, **kwargs)

    def capture_save(writer, *args, **kwargs):
        assert events == ["width", "height"]
        events.append("save")
        return save(writer, *args, **kwargs)

    monkeypatch.setattr(ExcelWriter, "insert_df2sheet", capture_insert)
    monkeypatch.setattr(ExcelWriter, "_calculate_content_width", capture_calculation)
    monkeypatch.setattr(ExcelWriter, "adjust_columns_width", capture_width)
    monkeypatch.setattr(ExcelWriter, "adjust_rows_height", capture_height)
    monkeypatch.setattr(ExcelWriter, "save", capture_save)
    data = pd.DataFrame({"比率": [0.123456789, 0.87654321], "金额": [1234.56789, 2345.67891], "说明": ["a\nb", "c\nd\ne"]})
    path = tmp_path / f"最终尺寸-{speed}.xlsx"
    dataframe2excel(
        data,
        path,
        title="汇总",
        sheet_name="报告",
        percent_cols=["比率"],
        custom_cols=["金额"],
        custom_format="#,##0.00",
        left_cols=["说明"],
        right_cols=["金额"],
        condition_cols=["比率"],
        auto_filter=True,
        auto_width=True,
        auto_height=True,
        decimal=None,
        speed=speed,
    )
    assert events == ["width", "height", "save"]
    book = load_workbook(path)
    ws = book["报告"]
    assert ws.column_dimensions["B"].width == pytest.approx(9.2)
    assert ws.column_dimensions["C"].width == pytest.approx(11.6)
    assert ws.row_dimensions[5].height == 26
    assert ws.row_dimensions[6].height == 38
    assert ws["B5"].value == 0.123456789
    book.close()


@pytest.mark.parametrize("auto_width,auto_height", [(False, False), (True, False), (False, True)])
def test_dimension_switches_are_independent_and_empty_exports_are_valid(monkeypatch, auto_width, auto_height):
    calls = []
    monkeypatch.setattr(ExcelWriter, "adjust_columns_width", lambda *args, **kwargs: calls.append("width"))
    monkeypatch.setattr(ExcelWriter, "adjust_rows_height", lambda *args, **kwargs: calls.append("height"))
    writer = ExcelWriter()
    dataframe2excel(pd.DataFrame({"值": [1]}), writer, auto_width=auto_width, auto_height=auto_height)
    assert calls == (["width"] if auto_width else []) + (["height"] if auto_height else [])
    calls.clear()
    dataframe2excel(pd.DataFrame(), writer, sheet_name="空表", header=False, auto_width=True, auto_height=True)
    assert calls == []
    writer.workbook.close()


@pytest.mark.parametrize("speed", ["normal", "fast"])
@pytest.mark.parametrize("header", [True, False])
def test_row_formats_are_applied_before_width_with_or_without_header(tmp_path, speed, header):
    data = pd.DataFrame({"A": [0.123456789], "B": [0.87654321]}, index=["比例"])
    path = tmp_path / "按行格式.xlsx"
    dataframe2excel(data, path, header=header, percent_rows=["比例"], auto_width=True, decimal=None, speed=speed)
    book = load_workbook(path)
    ws = book.active
    row = 3 if header else 2
    assert ws[f"B{row}"].number_format == ws[f"C{row}"].number_format == "0.00%"
    assert ws.column_dimensions["B"].width == ws.column_dimensions["C"].width == pytest.approx(9.2)
    book.close()


@pytest.mark.parametrize("auto_width,auto_height", [(True, False), (False, True), (True, True)])
def test_adjustments_preserve_custom_borders_merged_corners_and_template_styles(tmp_path, auto_width, auto_height):
    writer = ExcelWriter(theme_color="B42266")
    ws = writer.get_sheet_by_name("边框")
    writer.insert_value2sheet(ws, "B2", "合并表头", style="header", end_space="D2")
    writer.insert_value2sheet(ws, "B4", "父级分组", background_color="DEE1FB")
    ws["B4"].border = Border(
        left=Side(style="medium", color="B42266"),
        right=Side(style="dashed", color="00AA00"),
        top=Side(style="double", color="B42266"),
    )
    ws["B6"].border = Border(bottom=Side(style="medium", color="B42266"))
    writer.merge_cells(ws, "B4", "B6")
    writer.insert_value2sheet(ws, "C4", 0.123456789, decimal=None, number_format="0.00%")
    writer.insert_value2sheet(ws, "D4", "第一行\n第二行", wrap_text=True, horizontal="left")
    ws["C4"].border = Border(
        left=Side(style="dashDot", color="FF0000"),
        right=Side(style="double", color="123456"),
        top=Side(style="medium", color="B42266"),
        bottom=Side(style="dotted", color="654321"),
    )
    ws.row_dimensions[4].fill = PatternFill(fill_type="solid", fgColor="EEFFEE")
    ws.row_dimensions[4].border = Border(bottom=Side(style="thin", color="AABBCC"))
    ws.row_dimensions[4].outlineLevel = 1
    writer.set_column_width(ws, "B", 5)
    writer.set_row_height(ws, 4, 10)
    original_path = tmp_path / "调整前.xlsx"
    writer.save(original_path, close=False)
    original = load_workbook(original_path)
    cell_styles = {cell.coordinate: copy(cell._style) for row in ws for cell in row}
    row_style = copy(ws.row_dimensions[4]._style)
    if auto_width:
        writer.adjust_columns_width(ws, start_col="B", end_col="D")
    if auto_height:
        writer.adjust_rows_height(ws, start_row=2, end_row=6)
    assert all(cell._style == cell_styles[cell.coordinate] for row in ws for cell in row)
    assert ws.row_dimensions[4]._style == row_style
    adjusted_path = tmp_path / "调整后.xlsx"
    writer.save(adjusted_path)
    adjusted = load_workbook(adjusted_path)
    before, after = original["边框"], adjusted["边框"]
    assert set(map(str, before.merged_cells.ranges)) == set(map(str, after.merged_cells.ranges))
    for row in range(1, 7):
        for col in range(1, 5):
            old, new = before.cell(row, col), after.cell(row, col)
            assert new.value == old.value
            assert new.border == copy(old.border), f"边框发生变化：{new.coordinate}"
            assert new.fill == copy(old.fill)
            assert new.font == copy(old.font)
            assert new.alignment == copy(old.alignment)
            assert new.number_format == old.number_format
    assert after.row_dimensions[4]._style == before.row_dimensions[4]._style
    assert after.row_dimensions[4].outlineLevel == 1
    if auto_width:
        assert after.column_dimensions["B"].width > 5
        for col in "BCD":
            assert after.column_dimensions[col].style == before.column_dimensions["B"].style
    if auto_height:
        assert after.row_dimensions[4].height > 10
    original.close()
    adjusted.close()
    writer.workbook.close()
