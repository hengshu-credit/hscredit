"""读写 Excel 2010 原生数据条扩展（实心填充和固定中轴）。"""

import os
import posixpath
import re
import shutil
import tempfile
from uuid import uuid4
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from openpyxl.formatting.rule import DataBarRule
from openpyxl.utils import absolute_coordinate

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
X14 = "http://schemas.microsoft.com/office/spreadsheetml/2009/9/main"
XM = "http://schemas.microsoft.com/office/excel/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RULE_EXT = "{B025F937-C7B1-47D3-B67F-A62EFF666E3E}"
SHEET_EXT = "{78C0D931-6437-407d-A8EE-F0AAD7539E65}"


def solid_bar(cell_range, color, negative_color=None, *, start_zero=False):
    """生成兼容数据条，并为 Excel 登记实心颜色及可选的居中零轴。"""

    def opaque(value):
        rgb = value.lstrip("#")
        if not re.fullmatch(r"(?:[0-9a-fA-F]{2})?[0-9a-fA-F]{6}", rgb):
            raise ValueError("数据条颜色必须是十六进制 RGB 或 ARGB 颜色")
        return "FF" + rgb[-6:].upper()

    attrs = {"minLength": "0", "maxLength": "100", "gradient": "0", "direction": "leftToRight"}
    if negative_color is None:
        attrs.update(axisPosition="none", negativeBarColorSameAsPositive="1")
        bounds = [("num", "0") if start_zero else ("min", None), ("max", None)]
    else:
        attrs["axisPosition"] = "middle"
        ref = absolute_coordinate(cell_range)
        magnitude = f"MAX(MAX({ref}),-MIN({ref}))"
        bound = f"IF({magnitude}=0,1,{magnitude})"
        bounds = [("formula", "-" + bound), ("formula", bound)]
    (low, low_value), (high, high_value) = bounds
    rule = DataBarRule(start_type=low, start_value=low_value, end_type=high, end_value=high_value, color=opaque(color), minLength=0, maxLength=100)
    bar = ET.Element(f"{{{X14}}}dataBar", attrs)
    for kind, value in bounds:
        node = ET.SubElement(bar, f"{{{X14}}}cfvo", type=kind)
        if value is not None:
            ET.SubElement(node, f"{{{XM}}}f").text = value
    if negative_color is not None:
        ET.SubElement(bar, f"{{{X14}}}negativeFillColor", rgb=opaque(negative_color))
        ET.SubElement(bar, f"{{{X14}}}axisColor", rgb="FF85829D")
    rule._hscredit_bar = bar
    return rule


def restore(workbook, filename):
    """openpyxl 读取文件时会跳过 x14，把原数据条的扩展重新挂到对应规则上。"""
    sheets = {ws.title: ws for ws in workbook.worksheets if any(rule.type == "dataBar" for cf in ws.conditional_formatting for rule in cf.rules)}
    if not sheets:
        return
    with ZipFile(filename) as package:
        rels = ET.fromstring(package.read("xl/_rels/workbook.xml.rels"))
        paths = {r.get("Id"): posixpath.normpath(posixpath.join("xl", r.get("Target"))).lstrip("/") for r in rels}
        for sheet in ET.fromstring(package.read("xl/workbook.xml")).findall(f"{{{MAIN}}}sheets/{{{MAIN}}}sheet"):
            ws = sheets.get(sheet.get("name"))
            if ws is None:
                continue
            raw = package.read(paths[sheet.get(f"{{{REL}}}id")])
            if RULE_EXT.encode() not in raw.upper():
                continue
            root = ET.fromstring(raw)
            bars = {r.get("id"): r.find(f"{{{X14}}}dataBar") for r in root.findall(f"{{{MAIN}}}extLst/{{{MAIN}}}ext/{{{X14}}}conditionalFormattings/{{{X14}}}conditionalFormatting/{{{X14}}}cfRule")}
            links = {}
            for cf in root.findall(f"{{{MAIN}}}conditionalFormatting"):
                for r in cf.findall(f"{{{MAIN}}}cfRule"):
                    guid = r.findtext(f"{{{MAIN}}}extLst/{{{MAIN}}}ext/{{{X14}}}id")
                    if guid in bars and bars[guid] is not None:
                        links[(cf.get("sqref"), int(r.get("priority")))] = bars[guid]
            for cf in ws.conditional_formatting:
                for rule in cf.rules:
                    bar = links.get((str(cf.sqref), rule.priority))
                    if bar is not None:
                        rule._hscredit_bar = bar


def _sheet_xml(xml, bars):
    """只插入规则关联和工作表级扩展，保留绘图、透视表及迷你图的 XML。"""
    extensions = []

    def block(match):
        sqref = match[1].decode()

        def link(rule_match):
            raw = rule_match[0]
            priority = re.search(rb'\bpriority="(\d+)"', rule_match[1])
            bar = bars.get((sqref, int(priority[1]))) if priority else None
            if bar is None:
                return raw
            guid = "{" + str(uuid4()).upper() + "}"
            cf = ET.Element(f"{{{X14}}}conditionalFormatting")
            ET.SubElement(cf, f"{{{X14}}}cfRule", type="dataBar", id=guid).append(bar)
            ET.SubElement(cf, f"{{{XM}}}sqref").text = sqref
            extensions.append(ET.tostring(cf, encoding="utf-8"))
            ext = f'<extLst><ext uri="{RULE_EXT}" xmlns:x14="{X14}"><x14:id>{guid}</x14:id></ext></extLst>'
            return raw.replace(b"</cfRule>", ext.encode() + b"</cfRule>")

        return re.sub(rb"<cfRule\b([^>]*?)(?<!/)>.*?</cfRule>", link, match[0], flags=re.S)

    xml = re.sub(rb'<conditionalFormatting\b[^>]*\bsqref="([^"]+)"[^>]*>.*?</conditionalFormatting>', block, xml, flags=re.S)
    if len(extensions) != len(bars):
        raise ValueError("数据条扩展与工作表条件格式未能一一对应")
    ext = f'<ext uri="{SHEET_EXT}" xmlns:x14="{X14}"><x14:conditionalFormattings>'.encode() + b"".join(extensions) + b"</x14:conditionalFormattings></ext>"
    tail = re.search(rb"</extLst>(?=\s*</worksheet>\s*$)", xml)
    if tail:
        return xml[: tail.start()] + ext + xml[tail.start() :]
    return xml.replace(b"</worksheet>", b"<extLst>" + ext + b"</extLst></worksheet>")


def write(workbook, filename):
    """将当前实际存在的数据条写入暂存工作簿。"""
    parts = {}
    for ws in workbook.worksheets:
        bars = {(str(cf.sqref), r.priority): r._hscredit_bar for cf in ws.conditional_formatting for r in cf.rules if hasattr(r, "_hscredit_bar")}
        if bars:
            parts[ws.path.lstrip("/")] = bars
    if not parts:
        return
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(filename), suffix=".xlsx")
    os.close(descriptor)
    try:
        with ZipFile(filename) as src, ZipFile(temporary, "w") as dst:
            for item in src.infolist():
                if item.filename in parts:
                    dst.writestr(item, _sheet_xml(src.read(item), parts[item.filename]))
                else:
                    with src.open(item) as source, dst.open(item, "w") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
