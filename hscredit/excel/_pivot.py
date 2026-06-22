# -*- coding: utf-8 -*-
"""数据透视表 OOXML 部件构建工具.

openpyxl 仅支持读取数据透视表而不支持创建，故本模块负责根据源数据与透视配置，
构建符合 ECMA-376 规范的透视表相关 XML 部件（pivotCacheDefinition / pivotCacheRecords /
pivotTable），由 :mod:`hscredit.excel.writer` 在 openpyxl 保存后注入到 xlsx（zip）中。

构建出的透视表会写入 ``refreshOnLoad="1"``，Excel 打开时会基于缓存记录自动刷新，
即便布局信息存在细微偏差也会被重新计算，从而提高兼容性。

设计要点:
- 行/列标签字段在缓存中以共享项（sharedItems）索引引用，值字段以字面量存储
- 行/列字段默认关闭分类汇总（defaultSubtotal=0），布局为「明细 + 总计」，便于精确枚举
- 多值字段时，值字段占位（fld=-2）作为列轴最内层
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

# OOXML 命名空间 / 关系类型 / 内容类型
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

REL_CACHE_DEF = NS_REL + "/pivotCacheDefinition"
REL_CACHE_REC = NS_REL + "/pivotCacheRecords"
REL_PIVOT_TABLE = NS_REL + "/pivotTable"

CT_CACHE_DEF = "application/vnd.openxmlformats-officedocument.spreadsheetml.pivotCacheDefinition+xml"
CT_CACHE_REC = "application/vnd.openxmlformats-officedocument.spreadsheetml.pivotCacheRecords+xml"
CT_PIVOT_TABLE = "application/vnd.openxmlformats-officedocument.spreadsheetml.pivotTable+xml"

# 聚合方式 -> (OOXML subtotal 取值, 中文前缀)
AGG_MAP: Dict[str, Tuple[str, str]] = {
    "sum": ("sum", "求和项"),
    "count": ("count", "计数项"),
    "counta": ("count", "计数项"),
    "average": ("average", "平均值项"),
    "mean": ("average", "平均值项"),
    "max": ("max", "最大值项"),
    "min": ("min", "最小值项"),
    "product": ("product", "乘积项"),
    "count_nums": ("countNums", "数值计数项"),
    "countnums": ("countNums", "数值计数项"),
    "std": ("stdDev", "标准偏差项"),
    "stddev": ("stdDev", "标准偏差项"),
    "stdp": ("stdDevp", "总体标准偏差项"),
    "var": ("var", "方差项"),
    "varp": ("varp", "总体方差项"),
}


def _esc(text: Any) -> str:
    """XML 文本/属性转义。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _as_list(value: Any) -> List[Any]:
    """将单值/None 规范化为列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not (
        isinstance(value, float) and (np.isnan(value) or np.isinf(value))
    )


def normalize_values(values: Any, data: pd.DataFrame) -> List[Dict[str, str]]:
    """将 ``values`` 参数规范化为值字段配置列表。

    支持的输入形式:
    - ``'col'`` 或 ``['c1', 'c2']``：默认聚合（数值列用 sum，非数值列用 count）
    - ``[('col', 'sum'), ('col2', 'mean')]``：显式指定聚合
    - ``{'col': 'sum', 'col2': 'count'}``：字典形式

    :return: 形如 ``[{'col': 列名, 'agg': 聚合键, 'subtotal': OOXML值, 'name': 中文标题}]``
    """
    items: List[Tuple[Any, Optional[str]]] = []
    if isinstance(values, dict):
        items = [(k, v) for k, v in values.items()]
    else:
        for v in _as_list(values):
            if isinstance(v, (list, tuple)):
                items.append((v[0], v[1] if len(v) > 1 else None))
            else:
                items.append((v, None))

    result: List[Dict[str, str]] = []
    for col, agg in items:
        if col not in data.columns:
            raise ValueError("值字段 '{}' 不存在于源数据列中".format(col))
        if agg is None:
            agg = "sum" if pd.api.types.is_numeric_dtype(data[col]) else "count"
        key = str(agg).lower()
        if key not in AGG_MAP:
            raise ValueError(
                "不支持的聚合方式 '{}'，可选: {}".format(agg, ", ".join(sorted(AGG_MAP)))
            )
        subtotal, prefix = AGG_MAP[key]
        result.append({
            "col": col,
            "agg": key,
            "subtotal": subtotal,
            "name": "{}:{}".format(prefix, col),
        })
    return result


def _shared_items(series: pd.Series) -> Tuple[List[Any], bool]:
    """计算某列的共享项（去重 + 排序）及是否含缺失值。

    :return: (共享项列表, 是否含缺失值)
    """
    has_missing = bool(series.isna().any())
    uniques = series.dropna().unique().tolist()
    try:
        uniques = sorted(uniques)
    except TypeError:
        uniques = sorted(uniques, key=lambda x: str(x))
    return uniques, has_missing


def build_pivot_spec(
    data: pd.DataFrame,
    source_sheet: str,
    source_ref: str,
    pivot_sheet: str,
    pivot_anchor: Tuple[int, int],
    rows: Sequence[Any],
    columns: Sequence[Any],
    values: List[Dict[str, str]],
    filters: Sequence[Any],
    name: str,
    cache_id: int,
    show_row_totals: bool = True,
    show_col_totals: bool = True,
) -> Dict[str, Any]:
    """根据源数据与透视配置，计算构建 XML 所需的全部中间信息。

    :return: 透视表规格字典，供 :func:`render_*` 系列函数渲染 XML
    """
    rows = list(rows)
    columns = list(columns)
    filters = list(filters)
    all_columns = data.columns.tolist()

    label_cols = list(dict.fromkeys(rows + columns + filters))  # 去重保序
    for c in label_cols:
        if c not in all_columns:
            raise ValueError("字段 '{}' 不存在于源数据列中".format(c))

    # 每个字段的共享项（仅标签字段需要）
    field_infos: List[Dict[str, Any]] = []
    shared_lookup: Dict[Any, Dict[Any, int]] = {}
    for col in all_columns:
        info: Dict[str, Any] = {"name": col, "is_label": col in label_cols}
        if col in label_cols:
            shared, has_missing = _shared_items(data[col])
            info["shared"] = shared
            info["has_missing"] = has_missing
            info["all_number"] = all(_is_number(v) for v in shared) if shared else False
            shared_lookup[col] = {v: i for i, v in enumerate(shared)}
        else:
            ser = data[col]
            info["is_number"] = bool(pd.api.types.is_numeric_dtype(ser))
            if info["is_number"]:
                non_null = ser.dropna()
                info["min"] = float(non_null.min()) if len(non_null) else 0.0
                info["max"] = float(non_null.max()) if len(non_null) else 0.0
                info["all_integer"] = bool(
                    len(non_null) and np.all(np.equal(np.mod(non_null.to_numpy(dtype=float), 1), 0))
                )
            info["has_missing"] = bool(ser.isna().any())
        field_infos.append(info)

    col_index = {c: i for i, c in enumerate(all_columns)}

    # 缓存记录：标签字段用共享项索引，值字段用字面量
    records: List[List[Tuple[str, Any]]] = []
    for _, row in data.iterrows():
        rec: List[Tuple[str, Any]] = []
        for col in all_columns:
            val = row[col]
            if pd.isna(val):
                rec.append(("m", None))
            elif col in label_cols:
                rec.append(("x", shared_lookup[col][val]))
            elif _is_number(val):
                rec.append(("n", float(val)))
            else:
                rec.append(("s", val))
        records.append(rec)

    row_field_idx = [col_index[c] for c in rows]
    col_field_idx = [col_index[c] for c in columns]
    page_field_idx = [col_index[c] for c in filters]

    # 行轴明细组合（按共享项索引排序）
    def _combos(fields: List[Any]) -> List[Tuple[int, ...]]:
        if not fields:
            return []
        sub = data[fields].dropna(how="any").drop_duplicates()
        combos = []
        for _, r in sub.iterrows():
            combos.append(tuple(shared_lookup[f][r[f]] for f in fields))
        return sorted(set(combos))

    row_combos = _combos(rows)
    col_combos = _combos(columns)

    n_data = len(values)

    return {
        "name": name,
        "cache_id": cache_id,
        "source_sheet": source_sheet,
        "source_ref": source_ref,
        "pivot_sheet": pivot_sheet,
        "pivot_anchor": pivot_anchor,
        "field_infos": field_infos,
        "records": records,
        "row_field_idx": row_field_idx,
        "col_field_idx": col_field_idx,
        "page_field_idx": page_field_idx,
        "values": values,
        "row_combos": row_combos,
        "col_combos": col_combos,
        "n_data": n_data,
        "show_row_totals": show_row_totals and bool(rows),
        "show_col_totals": show_col_totals and bool(columns),
    }


def render_cache_definition_xml(spec: Dict[str, Any], rid_records: str) -> str:
    """渲染 pivotCacheDefinition XML。"""
    fields_xml = []
    for info in spec["field_infos"]:
        name = _esc(info["name"])
        if info["is_label"]:
            shared = info["shared"]
            items = []
            for v in shared:
                if _is_number(v) and not isinstance(v, bool):
                    items.append('<n v="{}"/>'.format(_fmt_num(v)))
                else:
                    items.append('<s v="{}"/>'.format(_esc(v)))
            if info.get("has_missing"):
                items.append("<m/>")
            attrs = ' count="{}"'.format(len(items))
            if info.get("all_number"):
                attrs += ' containsNumber="1"'
            shared_xml = "<sharedItems{}>{}</sharedItems>".format(attrs, "".join(items))
        else:
            if info.get("is_number"):
                attrs = (
                    ' containsSemiMixedTypes="0" containsString="0" containsNumber="1"'
                    ' minValue="{}" maxValue="{}"'.format(_fmt_num(info["min"]), _fmt_num(info["max"]))
                )
                if info.get("all_integer"):
                    attrs += ' containsInteger="1"'
            else:
                attrs = ""
            if info.get("has_missing"):
                attrs = ' containsBlank="1"' + attrs
            shared_xml = "<sharedItems{}/>".format(attrs)
        fields_xml.append(
            '<cacheField name="{}" numFmtId="0">{}</cacheField>'.format(name, shared_xml)
        )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<pivotCacheDefinition xmlns="{ns}" xmlns:r="{nsr}" r:id="{rid}" '
        'refreshOnLoad="1" refreshedBy="hscredit" createdVersion="6" refreshedVersion="6" '
        'minRefreshableVersion="3" recordCount="{rc}">'
        '<cacheSource type="worksheet">'
        '<worksheetSource ref="{ref}" sheet="{sheet}"/>'
        '</cacheSource>'
        '<cacheFields count="{fc}">{fields}</cacheFields>'
        '</pivotCacheDefinition>'
    ).format(
        ns=NS_MAIN, nsr=NS_REL, rid=rid_records, rc=len(spec["records"]),
        ref=_esc(spec["source_ref"]), sheet=_esc(spec["source_sheet"]),
        fc=len(spec["field_infos"]), fields="".join(fields_xml),
    )


def render_cache_records_xml(spec: Dict[str, Any]) -> str:
    """渲染 pivotCacheRecords XML。"""
    rows_xml = []
    for rec in spec["records"]:
        cells = []
        for tag, val in rec:
            if tag == "m":
                cells.append("<m/>")
            elif tag == "x":
                cells.append('<x v="{}"/>'.format(val))
            elif tag == "n":
                cells.append('<n v="{}"/>'.format(_fmt_num(val)))
            else:
                cells.append('<s v="{}"/>'.format(_esc(val)))
        rows_xml.append("<r>{}</r>".format("".join(cells)))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<pivotCacheRecords xmlns="{ns}" xmlns:r="{nsr}" count="{rc}">{rows}</pivotCacheRecords>'
    ).format(ns=NS_MAIN, nsr=NS_REL, rc=len(rows_xml), rows="".join(rows_xml))


def _fmt_num(value: float) -> str:
    """数字格式化：整数去掉 .0，其余保留原始浮点。"""
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _axis_items_xml(combos: List[Tuple[int, ...]], grand: bool) -> Tuple[str, int]:
    """根据明细组合枚举行/列轴 ``<i>`` 项（含游程压缩 ``r`` 属性）。

    :param combos: 已排序的索引组合列表，每个元素为各层级共享项索引构成的元组
    :param grand: 是否追加总计项
    :return: (拼接后的 XML, 项数)
    """
    out: List[str] = []
    prev: Tuple[int, ...] = ()
    for combo in combos:
        # 与上一行相同的前缀长度（游程）
        r = 0
        for a, b in zip(prev, combo):
            if a == b:
                r += 1
            else:
                break
        xs = "".join(
            "<x/>" if combo[k] == 0 else '<x v="{}"/>'.format(combo[k])
            for k in range(r, len(combo))
        )
        attr = ' r="{}"'.format(r) if r else ""
        out.append("<i{}>{}</i>".format(attr, xs))
        prev = combo
    if grand:
        out.append('<i t="grand"><x/></i>')
    return "".join(out), len(out)


def compute_pivot_layout(spec: Dict[str, Any]) -> Dict[str, Any]:
    """估算透视表在工作表中的落位（Excel 刷新时会重算，此处用于 location 与透视图引用）。

    :return: 含 ``first_header_row`` / ``first_data_row`` / ``first_data_col`` /
        ``n_row_leaf`` / ``n_col_leaf`` / ``n_row_label_cols`` / ``width`` / ``height`` / ``ref``
        （行列偏移均相对透视表左上角锚点，从 0 开始）
    """
    from openpyxl.utils import get_column_letter

    row_idx = spec["row_field_idx"]
    col_idx = spec["col_field_idx"]
    page_idx = spec["page_field_idx"]
    n_data = spec["n_data"]

    n_row_label_cols = max(1, len(row_idx))
    n_page_rows = len(page_idx) + (1 if page_idx else 0)
    col_header_rows = len(col_idx) + (1 if n_data > 1 else 0)
    first_header_row = n_page_rows
    first_data_row = first_header_row + max(1, col_header_rows)
    first_data_col = n_row_label_cols

    n_row_leaf = len(spec["row_combos"])
    base_combos = spec["col_combos"] if col_idx else [()]
    n_col_leaf = (len(base_combos) * n_data) if (col_idx or n_data > 1) else 1
    n_col_leaf = max(1, n_col_leaf)

    width = n_row_label_cols + n_col_leaf
    height = first_data_row + max(1, n_row_leaf) + (1 if spec["show_row_totals"] else 0)

    anchor_row, anchor_col = spec["pivot_anchor"]
    c0 = get_column_letter(anchor_col)
    c1 = get_column_letter(anchor_col + width - 1)
    ref = "{}{}:{}{}".format(c0, anchor_row, c1, anchor_row + height - 1)

    return {
        "first_header_row": first_header_row,
        "first_data_row": first_data_row,
        "first_data_col": first_data_col,
        "n_row_label_cols": n_row_label_cols,
        "n_row_leaf": n_row_leaf,
        "n_col_leaf": n_col_leaf,
        "width": width,
        "height": height,
        "ref": ref,
    }


def render_pivot_table_xml(spec: Dict[str, Any]) -> str:
    """渲染 pivotTableDefinition XML。"""
    field_infos = spec["field_infos"]
    n_fields = len(field_infos)
    row_idx = spec["row_field_idx"]
    col_idx = spec["col_field_idx"]
    page_idx = spec["page_field_idx"]
    values = spec["values"]
    n_data = spec["n_data"]

    axis_of = {}
    for i in row_idx:
        axis_of[i] = "axisRow"
    for i in col_idx:
        axis_of[i] = "axisCol"
    for i in page_idx:
        axis_of[i] = "axisPage"
    name_to_idx = {info["name"]: i for i, info in enumerate(field_infos)}
    data_field_idx = {name_to_idx[v["col"]] for v in values}

    # pivotFields
    pivot_fields_xml = []
    for i, info in enumerate(field_infos):
        attrs = ""
        is_axis = i in axis_of
        if is_axis:
            attrs += ' axis="{}"'.format(axis_of[i])
        if i in data_field_idx:
            attrs += ' dataField="1"'
        attrs += ' showAll="0"'
        if is_axis:
            attrs += ' defaultSubtotal="0"'
            shared = info.get("shared", [])
            items = "".join(
                '<item x="{}"/>'.format(k) for k in range(len(shared))
            )
            if info.get("has_missing"):
                items += '<item x="{}"/>'.format(len(shared))
            n_items = len(shared) + (1 if info.get("has_missing") else 0)
            pivot_fields_xml.append(
                '<pivotField{}><items count="{}">{}</items></pivotField>'.format(attrs, n_items, items)
            )
        else:
            pivot_fields_xml.append("<pivotField{}/>".format(attrs))

    # rowFields / rowItems
    if row_idx:
        row_fields_xml = '<rowFields count="{}">{}</rowFields>'.format(
            len(row_idx), "".join('<field x="{}"/>'.format(i) for i in row_idx)
        )
        items_xml, n_row_items = _axis_items_xml(spec["row_combos"], spec["show_row_totals"])
        row_items_xml = '<rowItems count="{}">{}</rowItems>'.format(n_row_items, items_xml)
    else:
        row_fields_xml = ""
        # 无行字段：仅一个总计行
        row_items_xml = '<rowItems count="1"><i><x/></i></rowItems>'

    # colFields / colItems（含多值字段占位 fld=-2）
    col_levels = list(col_idx)
    if n_data > 1:
        col_levels = col_levels + [-2]

    if not col_levels:
        col_fields_xml = ""
        col_items_xml = '<colItems count="1"><i/></colItems>'
    else:
        col_fields_xml = '<colFields count="{}">{}</colFields>'.format(
            len(col_levels), "".join('<field x="{}"/>'.format(i) for i in col_levels)
        )
        # 构造列轴组合：列标签组合 × 值字段索引
        base_combos = spec["col_combos"] if col_idx else [()]
        if n_data > 1:
            combos = [bc + (d,) for bc in base_combos for d in range(n_data)]
        else:
            combos = [bc for bc in base_combos]
        combos = sorted(combos)
        items_xml, n_col_items = _axis_items_xml(combos, spec["show_col_totals"])
        col_items_xml = '<colItems count="{}">{}</colItems>'.format(n_col_items, items_xml)

    # pageFields
    if page_idx:
        page_fields_xml = '<pageFields count="{}">{}</pageFields>'.format(
            len(page_idx), "".join('<pageField fld="{}" hier="-1"/>'.format(i) for i in page_idx)
        )
    else:
        page_fields_xml = ""

    # dataFields
    data_fields_xml = '<dataFields count="{}">{}</dataFields>'.format(
        n_data,
        "".join(
            '<dataField name="{}" fld="{}" subtotal="{}" baseField="0" baseItem="0"/>'.format(
                _esc(v["name"]), name_to_idx[v["col"]],
                v["subtotal"],
            ) if v["subtotal"] != "sum" else
            '<dataField name="{}" fld="{}" baseField="0" baseItem="0"/>'.format(
                _esc(v["name"]), name_to_idx[v["col"]]
            )
            for v in values
        ),
    )

    # location 估算（Excel 刷新时会重算）
    layout = compute_pivot_layout(spec)
    location_xml = (
        '<location ref="{ref}" firstHeaderRow="{fhr}" firstDataRow="{fdr}" firstDataCol="{fdc}"{page}/>'
    ).format(
        ref=layout["ref"], fhr=layout["first_header_row"], fdr=layout["first_data_row"],
        fdc=layout["first_data_col"],
        page=' rowPageCount="{}" colPageCount="1"'.format(len(page_idx)) if page_idx else "",
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<pivotTableDefinition xmlns="{ns}" name="{name}" cacheId="{cid}" '
        'applyNumberFormats="0" applyBorderFormats="0" applyFontFormats="0" '
        'applyPatternFormats="0" applyAlignmentFormats="0" applyWidthHeightFormats="1" '
        'dataCaption="值" updatedVersion="6" minRefreshableVersion="3" useAutoFormatting="1" '
        'itemPrintTitles="1" createdVersion="6" indent="0" compact="0" compactData="0" '
        'outline="1" outlineData="1" gridDropZones="0" multipleFieldFilters="0"{rgt}{cgt}>'
        '{location}'
        '<pivotFields count="{nf}">{pivot_fields}</pivotFields>'
        '{row_fields}{row_items}{col_fields}{col_items}{page_fields}{data_fields}'
        '<pivotTableStyleInfo name="{style}" showRowHeaders="1" showColHeaders="1" '
        'showRowStripes="0" showColStripes="0" showLastColumn="1"/>'
        '</pivotTableDefinition>'
    ).format(
        ns=NS_MAIN, name=_esc(spec["name"]), cid=spec["cache_id"],
        rgt="" if spec["show_row_totals"] else ' rowGrandTotals="0"',
        cgt="" if spec["show_col_totals"] else ' colGrandTotals="0"',
        location=location_xml,
        nf=n_fields, pivot_fields="".join(pivot_fields_xml),
        row_fields=row_fields_xml, row_items=row_items_xml,
        col_fields=col_fields_xml, col_items=col_items_xml,
        page_fields=page_fields_xml, data_fields=data_fields_xml,
        style=spec.get("style", "PivotStyleLight16"),
    )
