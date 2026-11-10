"""自动报告统一的 save/sheet 解析及工作簿生命周期管理。"""

import os
from copy import copy
from io import BytesIO

from openpyxl.worksheet.worksheet import Worksheet

from ..excel import ExcelWriter


def reject_legacy_report_keywords(kwargs):
    """拒绝旧输出参数，避免经由 kwargs 静默丢失。"""
    for old, new in {"excel_path": "save", "excel_writer": "save", "sheet_name": "sheet"}.items():
        if old in kwargs:
            raise TypeError(f"参数 {old} 已移除，请使用 {new}")


def resolve_report_save(save=None):
    """save 只接受文件路径、hscredit ExcelWriter 或 None。"""
    if save is None or isinstance(save, ExcelWriter):
        return save
    try:
        path = os.fspath(save)
    except TypeError as exc:
        raise TypeError("save 必须是文件路径或 hscredit.excel.ExcelWriter") from exc
    if not isinstance(path, str):
        raise TypeError("save 必须是文件路径或 hscredit.excel.ExcelWriter")
    if not path.strip():
        raise ValueError("save 不能为空路径")
    return path


def validate_report_output(save=None, sheet=None, *, default="分析报告", max_length=31):
    """校验输出参数，不创建工作簿、工作表或文件。"""
    save = resolve_report_save(save)
    name = sheet.title if isinstance(sheet, Worksheet) else default if sheet is None else sheet
    if not isinstance(name, str) or not name or len(name) > max_length or any(char in name for char in "[]:*?/\\"):
        raise ValueError(f"sheet 必须是 1 至 {max_length} 字符的有效工作表名称或 openpyxl Worksheet")
    if isinstance(save, ExcelWriter) and isinstance(sheet, Worksheet) and sheet.parent is not save.workbook:
        raise ValueError("sheet 必须属于 save 指定的 writer 工作簿")
    return save, name


def resolve_report_output(save=None, sheet=None, *, default="分析报告", max_length=31, writer_params=None):
    """返回写入器、目标工作表、落盘路径及工作簿是否由本次调用创建。"""
    save, name = validate_report_output(save, sheet, default=default, max_length=max_length)
    owned = not isinstance(save, ExcelWriter) and not isinstance(sheet, Worksheet)
    writer = save if isinstance(save, ExcelWriter) else ExcelWriter(**(writer_params or {}))
    if isinstance(sheet, Worksheet) and sheet.parent is not writer.workbook:
        template = writer.workbook
        writer.workbook = sheet.parent
        writer.style_sheet = None
        for style in writer.name_styles:
            if style.name not in writer.workbook.style_names:
                writer.workbook.add_named_style(copy(style))
        writer._style_cache = {style.name: copy(style.as_tuple()) for style in writer.workbook._named_styles}
        template.close()
    worksheet = sheet if isinstance(sheet, Worksheet) else writer.get_sheet_by_name(name)
    return writer, worksheet, save if isinstance(save, str) else None, owned


def preserve_report_images(writer):
    """把图片保存在内存中，让外部工作簿脱离临时图片目录继续保存。"""
    for sheet in writer.workbook:
        for image in sheet._images:
            image.ref = BytesIO(image._data())


def save_report_workbook(writer, save, *, close=True):
    """保存报告，复用调用方工作簿时保留图片流以支持继续追加和保存。"""
    images = [] if close else [(image, image._data()) for sheet in writer.workbook for image in sheet._images]
    for image, payload in images:
        image.ref = BytesIO(payload)
    try:
        writer.save(save, close=close)
    finally:
        for image, payload in images:
            image.ref = BytesIO(payload)
