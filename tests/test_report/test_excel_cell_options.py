"""单元格写入的样式覆盖、尺寸、数字样文本及批注保存契约。"""

from copy import copy
from datetime import datetime
import xml.etree.ElementTree as ET
import zipfile

import pytest
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, NamedStyle, PatternFill

from hscredit.excel import ExcelWriter


@pytest.fixture
def sheet():
    writer = ExcelWriter()
    ws = writer.get_sheet_by_name("内容")
    yield writer, ws
    writer.workbook.close()


def save_and_load(writer, tmp_path):
    path = tmp_path / "内容.xlsx"
    writer.save(path)
    return path, load_workbook(path)


def test_merged_cell_keeps_all_requested_options_and_numeric_text_warning_suppression(sheet, tmp_path):
    writer, ws = sheet
    result = writer.insert_value2sheet(
        ws,
        "B2",
        "00123",
        style="header",
        end_space="D3",
        align={"horizontal": "right", "vertical": "bottom", "text_rotation": 15},
        horizontal="left",
        vertical="top",
        wrap_text=True,
        indent=2,
        bold=True,
        italic=True,
        strike=True,
        font_name="宋体",
        font_size=18,
        font_color="#123456",
        background_color="FFEEDDCC",
        number_format="0.00",
        column_width=18,
        row_height=24,
        auto_width=True,
        auto_height=True,
        max_col_width=5,
        max_row_height=5,
        comment="保留前导零，按文本展示。",
    )
    assert result == (4, 5)
    assert ws["D3"].fill == copy(ws["B2"].fill)
    writer.insert_value2sheet(ws, "B6", "默认样式")
    path, book = save_and_load(writer, tmp_path)
    saved = book["内容"]
    cell = saved["B2"]
    assert cell.value == "00123" and cell.data_type == "s" and cell.number_format == "@"
    assert cell.alignment.horizontal == "left" and cell.alignment.vertical == "top"
    assert cell.alignment.wrap_text and cell.alignment.indent == 2 and cell.alignment.textRotation == 15
    assert cell.font.bold and cell.font.italic and cell.font.strike
    assert cell.font.name == "宋体" and cell.font.sz == 18 and cell.font.color.rgb == "FF123456"
    assert cell.fill.fill_type == "solid" and cell.fill.fgColor.rgb == "FFEEDDCC"
    assert cell.comment.text == "保留前导零，按文本展示。"
    assert cell.comment.author == "logo hscredit"
    assert all(saved.column_dimensions[col].width == 18 for col in "BCD")
    assert all(saved.row_dimensions[row].height == 24 for row in [2, 3])
    assert not saved["B6"].font.bold and not saved["B6"].font.italic and not saved["B6"].font.strike
    book.close()
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ignored = root.find("x:ignoredErrors/x:ignoredError", ns)
        assert ignored.attrib["numberStoredAsText"] == "1" and ignored.attrib["sqref"] == "B2"
        tags = [child.tag.rsplit("}", 1)[-1] for child in root]
        assert tags.index("ignoredErrors") < tags.index("legacyDrawing")


@pytest.mark.parametrize("horizontal", ["left", "center", "right"])
@pytest.mark.parametrize("vertical", ["top", "center", "bottom"])
def test_alignment_directions_round_trip(sheet, tmp_path, horizontal, vertical):
    writer, ws = sheet
    writer.insert_value2sheet(ws, (2, 2), "对齐", horizontal=horizontal, vertical=vertical)
    _, book = save_and_load(writer, tmp_path)
    assert book["内容"]["B2"].alignment.horizontal == horizontal
    assert book["内容"]["B2"].alignment.vertical == vertical
    book.close()


def test_partial_alignment_and_style_none_preserve_unspecified_properties(sheet):
    writer, ws = sheet
    cell = ws["B2"]
    cell.font = Font(size=16, bold=True, color="123456")
    cell.alignment = Alignment(horizontal="right", vertical="bottom", wrap_text=True, textRotation=30, indent=2)
    writer.insert_value2sheet(ws, "B2", "保留", style=None, align={"horizontal": "left"})
    assert cell.font.sz == 16 and cell.font.bold
    assert cell.alignment.horizontal == "left" and cell.alignment.vertical == "bottom"
    assert cell.alignment.wrap_text and cell.alignment.textRotation == 30 and cell.alignment.indent == 2
    alignment = Alignment(horizontal="center", vertical="top", wrap_text=False)
    writer.insert_value2sheet(ws, "C2", "对象", align=alignment, horizontal="right")
    assert ws["C2"].alignment.horizontal == "right" and ws["C2"].alignment.vertical == "top"
    assert alignment.horizontal == "center"
    writer.insert_value2sheet(ws, "D2", "标题", style="header", align={"horizontal": "right"})
    assert ws["D2"].alignment.wrap_text  # 不再因只改水平对齐而清除标题换行。


def test_false_options_clear_flags_without_modifying_named_style(sheet, tmp_path):
    writer, ws = sheet
    style = NamedStyle(name="强调", font=Font(bold=True, italic=True, strike=True), alignment=Alignment(wrap_text=True, indent=2))
    writer.insert_value2sheet(ws, "B2", "取消强调", style=style, bold=False, italic=False, strike=False, wrap_text=False, indent=0)
    writer.insert_value2sheet(ws, "C2", "保留强调", style=style)
    _, book = save_and_load(writer, tmp_path)
    cleared, original = book["内容"]["B2"], book["内容"]["C2"]
    assert not cleared.font.bold and not cleared.font.italic and not cleared.font.strike
    assert not cleared.alignment.wrap_text and cleared.alignment.indent == 0
    assert original.font.bold and original.font.italic and original.font.strike and original.alignment.wrap_text
    assert original.alignment.indent == 2
    book.close()


@pytest.mark.parametrize("value,number_format", [(0.123456789, "0.00%"), (12345.678901, "#,##0.00"), (datetime(2026, 11, 4), "yyyy-mm-dd")])
def test_number_format_preserves_native_value_with_decimal_none(sheet, tmp_path, value, number_format):
    writer, ws = sheet
    writer.insert_value2sheet(ws, "B2", value, decimal=None, number_format=number_format)
    _, book = save_and_load(writer, tmp_path)
    assert book["内容"]["B2"].value == value
    assert book["内容"]["B2"].number_format == number_format
    book.close()


@pytest.mark.parametrize("wrap_text,maximum,expected", [(True, 120, 38), (False, 120, 14), (True, 30, 30)])
def test_auto_height_uses_wrap_and_explicit_column_width(sheet, wrap_text, maximum, expected):
    writer, ws = sheet
    writer.insert_value2sheet(ws, "B2", "abcdefghijklmn", column_width=10, wrap_text=wrap_text, auto_height=True, max_row_height=maximum)
    assert ws.row_dimensions[2].height == expected
    assert ws.column_dimensions["B"].width == 10


def test_auto_height_uses_merged_width_and_larger_font(sheet):
    writer, ws = sheet
    writer.insert_value2sheet(ws, "B2", "a" * 20, end_space="D2", column_width=10, wrap_text=True, font_size=20, auto_height=True)
    assert ws.row_dimensions[2].height == 50


def test_auto_height_accounts_for_indented_wrapped_text(sheet):
    writer, ws = sheet
    writer.insert_value2sheet(ws, "B2", "abcdefghijklmn", column_width=10, horizontal="left", indent=1, wrap_text=True, auto_height=True)
    assert ws.row_dimensions[2].height == 50


def test_auto_width_uses_font_size_without_scanning_or_resetting_column_styles(sheet, monkeypatch):
    writer, ws = sheet
    ws["B5"].fill = PatternFill(fill_type="solid", fgColor="00FF00")
    original_style = copy(ws["B5"]._style)

    original_cell = ws.cell

    def only_target_cell(row, column, value=None):
        assert (row, column) == (2, 2), "单值调整列宽无需扫描其他单元格"
        return original_cell(row, column, value)

    monkeypatch.setattr(ws, "cell", only_target_cell)
    writer.insert_value2sheet(ws, "B2", "a" * 40, font_size=20, bold=True, auto_width=True, max_col_width=100)
    assert ws.column_dimensions["B"].width == pytest.approx(96)
    assert ws["B2"].font.bold and ws["B2"].font.sz == 20
    assert ws["B5"]._style == original_style


def test_comment_object_author_and_removal_and_numeric_text_overwrite(sheet, tmp_path):
    writer, ws = sheet
    comment = Comment("对象批注", "hscredit", width=300, height=120)
    writer.insert_value2sheet(ws, "B2", "00123", comment=comment)
    assert comment.parent is None
    assert ws["B2"].comment.width == 300 and ws["B2"].comment.height == 120
    writer.insert_value2sheet(ws, "B2", 123, style=None, number_format="General")
    assert ws["B2"].comment.author == "hscredit"
    writer.insert_value2sheet(ws, "C2", "25%", comment="待删除", comment_author="自定义作者")
    writer.insert_value2sheet(ws, "C2", "25%", style=None, comment="")
    path, book = save_and_load(writer, tmp_path)
    assert book["内容"]["B2"].data_type == "n" and book["内容"]["B2"].comment.author == "hscredit"
    assert book["内容"]["C2"].comment is None
    book.close()
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ignored = root.find("x:ignoredErrors/x:ignoredError", ns)
        assert ignored.attrib["sqref"] == "C2"


@pytest.mark.parametrize(
    "options,error",
    [
        ({"bold": "yes"}, TypeError),
        ({"wrap_text": "false"}, TypeError),
        ({"font_color": "red"}, ValueError),
        ({"background_color": "#xyzxyz"}, ValueError),
        ({"row_height": -1}, ValueError),
        ({"column_width": 0}, ValueError),
        ({"font_size": float("inf")}, ValueError),
        ({"comment": 123}, TypeError),
        ({"align": {"unknown_option": True}}, ValueError),
        ({"indent": 256}, ValueError),
    ],
)
def test_invalid_cell_options_do_not_write_value(sheet, options, error):
    writer, ws = sheet
    with pytest.raises(error):
        writer.insert_value2sheet(ws, "B2", "不应写入", **options)
    assert ws["B2"].value is None
