"""筛选报告的安全、完整 JSON 和流式 Excel 发布适配。

保存结果位于不可变版本目录；消费者使用返回路径或 manifest 获取同一版本，
而不是猜测根目录文件名。JSON 使用有类型的字段/单元格协议，禁止 pickle。
"""

from datetime import date, datetime
import copy
import json
import math
import logging
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import pandas as pd

from ..exceptions import ValidationError
from ..core.selectors.reporting import (
    SelectionReport,
    SCHEMA_VERSION,
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    METRIC_COLUMNS,
    HISTORY_COLUMNS,
    _INT_COLUMNS,
    _FLOAT_COLUMNS,
    _BOOL_COLUMNS,
)

_TABLES = {"summary": "筛选汇总", "details": "特征决策", "metrics": "指标明细", "history": "迭代记录"}
logger = logging.getLogger(__name__)


def to_report_result(report):
    """将不可变协议快照适配为共享的 ReportResult 章节结果。"""
    from .result import ReportResult

    result = ReportResult(
        mode="strict" if report._metadata.get("完整", True) else "best_effort", metadata=report.metadata
    )
    for attribute, name in _TABLES.items():
        frame = getattr(report, attribute)
        result.plan(name, input_rows=len(frame))
        result.success(name, frame)
    if not report._metadata.get("完整", True):
        result.failure("来源完整性", "；".join(map(str, report._metadata.get("诊断", []))) or "报告来源不完整")
    return result


def _pack(value):
    if value is pd.NA or value is pd.NaT:
        return {"类型": "空值"}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"类型": "浮点特殊值", "值": "NaN"}
        if math.isinf(value):
            return {"类型": "浮点特殊值", "值": "+Inf" if value > 0 else "-Inf"}
        return value
    if isinstance(value, pd.Timestamp):
        return {"类型": "时间戳", "值": value.isoformat()}
    if isinstance(value, datetime):
        return {"类型": "日期时间", "值": value.isoformat()}
    if isinstance(value, date):
        return {"类型": "日期", "值": value.isoformat()}
    if isinstance(value, tuple):
        return {"类型": "元组", "值": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        # Preserve integer/string keys, and avoid ambiguity with type-tag-like user dictionaries.
        return {"类型": "映射", "值": [[_pack(key), _pack(item)] for key, item in value.items()]}
    raise ValidationError(f"筛选报告含不支持的 JSON 类型：{type(value).__name__}")


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("类型")
    if kind == "空值":
        return pd.NA
    if kind == "浮点特殊值":
        return {"NaN": float("nan"), "+Inf": float("inf"), "-Inf": -float("inf")}[value["值"]]
    if kind == "日期时间":
        return datetime.fromisoformat(value["值"])
    if kind == "日期":
        return date.fromisoformat(value["值"])
    if kind == "时间戳":
        return pd.Timestamp(value["值"])
    if kind == "元组":
        return tuple(_unpack(item) for item in value["值"])
    if kind == "映射":
        return {_unpack(key): _unpack(item) for key, item in value["值"]}
    raise ValidationError("无法识别筛选报告 JSON 类型标记")


def _write_json(report, path, run_id, sheet_mapping):
    # Stream rows; do not materialize every table as a second full list of dictionaries.
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write('{"格式":"hscredit.selection-report","协议版本":')
        handle.write(str(SCHEMA_VERSION))
        handle.write(',"run_id":')
        json.dump(run_id, handle)
        handle.write(',"元数据":')
        json.dump(_pack(report.metadata), handle, ensure_ascii=False, allow_nan=False)
        handle.write(',"工作表映射":')
        json.dump(sheet_mapping, handle, ensure_ascii=False, allow_nan=False)
        handle.write(',"表":{')
        for index, (attribute, name) in enumerate(_TABLES.items()):
            if index:
                handle.write(",")
            json.dump(attribute, handle)
            handle.write(':{"名称":')
            json.dump(name, handle, ensure_ascii=False)
            frame = getattr(report, "_" + attribute)
            handle.write(',"列":')
            json.dump(list(frame.columns), handle, ensure_ascii=False)
            handle.write(',"行":[')
            for position, row in enumerate(frame.itertuples(index=False, name=None)):
                if position:
                    handle.write(",")
                json.dump([_pack(value) for value in row], handle, ensure_ascii=False, allow_nan=False)
            handle.write("]}")
        handle.write("}}")


def load_selection_report(path):
    """读取本模块保存的完整、非可执行 JSON 报告。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("格式") != "hscredit.selection-report"
        or type(payload.get("协议版本")) is not int
        or payload.get("协议版本") != SCHEMA_VERSION
    ):
        raise ValidationError("不是支持的筛选报告 JSON 协议")
    metadata = _unpack(payload.get("元数据"))
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != SCHEMA_VERSION
        or type(metadata.get("完整")) is not bool
    ):
        raise ValidationError("筛选报告元数据的协议版本或完整性状态无效")
    if not isinstance(metadata.get("诊断"), list):
        raise ValidationError("筛选报告诊断必须为列表")
    schemas = {
        "summary": SUMMARY_COLUMNS,
        "details": DETAIL_COLUMNS,
        "metrics": METRIC_COLUMNS,
        "history": HISTORY_COLUMNS,
    }
    if not isinstance(payload.get("表"), dict) or set(payload["表"]) != set(schemas):
        raise ValidationError("筛选报告必须包含全部四个固定结构表")
    values = {}
    for attribute in _TABLES:
        table = payload["表"][attribute]
        if (
            not isinstance(table, dict)
            or table.get("列") != schemas[attribute]
            or not isinstance(table.get("行"), list)
        ):
            raise ValidationError(f"筛选报告 {attribute} 的列结构与协议不一致")
        if any(not isinstance(row, list) or len(row) != len(table["列"]) for row in table["行"]):
            raise ValidationError(f"筛选报告 {attribute} 的行长度与列结构不一致")
        rows = [[_unpack(value) for value in row] for row in table["行"]]
        for row in rows:
            for column, value in zip(table["列"], row):
                if value is None or value is pd.NA:
                    continue
                valid = True
                if column in _INT_COLUMNS:
                    valid = type(value) is int
                elif column in _FLOAT_COLUMNS:
                    valid = type(value) in {int, float}
                elif column in _BOOL_COLUMNS:
                    valid = type(value) is bool
                if not valid:
                    raise ValidationError(f"筛选报告 {attribute} 的 {column} 列值类型无效，不能静默转换为缺失")
        values[attribute] = pd.DataFrame(rows, columns=table["列"], dtype=object)
    return SelectionReport(metadata=metadata, **values)


def _sheet_name(name, used):
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name)).strip("'") or "报告"
    base = cleaned[:31]
    candidate, suffix = base, 1
    while candidate.lower() in used:
        suffix += 1
        candidate = f"{base[:31-len(str(suffix))-1]}_{suffix}"
    used.add(candidate.lower())
    return candidate


def _excel_value(value):
    if value is pd.NA or value is pd.NaT or value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "正无穷" if value == float("inf") else "负无穷" if value == -float("inf") else None
    if isinstance(value, (dict, tuple, list)):
        return json.dumps(_pack(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10**15:
        return str(value)
    return value


def _plan_excel(report, max_rows_per_sheet, max_excel_cells, max_sheets, by_selector):
    if (
        not isinstance(max_rows_per_sheet, int)
        or isinstance(max_rows_per_sheet, bool)
        or not 1 <= max_rows_per_sheet <= 1048575
    ):
        raise ValidationError("max_rows_per_sheet 必须为 1 至 1048575 的整数（不含表头）")
    if not isinstance(max_excel_cells, int) or max_excel_cells < 1 or not isinstance(max_sheets, int) or max_sheets < 1:
        raise ValidationError("max_excel_cells 和 max_sheets 必须为正整数")
    frames = [(name, getattr(report, "_" + attribute)) for attribute, name in _TABLES.items()]
    status = pd.DataFrame(
        [
            {"项目": "报告完整", "内容": report._metadata.get("完整", True)},
            {"项目": "诊断", "内容": report._metadata.get("诊断", [])},
        ]
    )
    snapshots = report._metadata.get("阶段快照", {})
    parameters = pd.DataFrame(
        [
            {"阶段路径": stage, "参数": snapshot.get("拟合参数", {}), "fit_id": snapshot.get("fit_id")}
            for stage, snapshot in snapshots.items()
        ],
        columns=["阶段路径", "参数", "fit_id"],
    )
    frames = [("执行状态", status)] + frames + [("参数说明", parameters)]
    if by_selector:
        for path, part in report._details.groupby("阶段路径", sort=False):
            frames.append((str(path), part))
    total_cells = sum(
        (len(frame) + max(1, math.ceil(len(frame) / max_rows_per_sheet))) * len(frame.columns) for _, frame in frames
    )
    if total_cells > max_excel_cells:
        raise ValidationError(
            f"Excel 预计 {total_cells} 个单元格，超出 max_excel_cells={max_excel_cells}；请仅保存完整 JSON 或提高预算"
        )
    plans, used = [], set()
    for label, frame in frames:
        chunks = max(1, math.ceil(len(frame) / max_rows_per_sheet))
        for chunk in range(chunks):
            start, end = chunk * max_rows_per_sheet, min(len(frame), (chunk + 1) * max_rows_per_sheet)
            sheet = _sheet_name(label if chunks == 1 else f"{label}_{chunk+1}", used)
            plans.append((sheet, label, frame, start, end))
    if len(plans) > max_sheets:
        raise ValidationError(f"Excel 需要 {len(plans)} 个工作表，超出 max_sheets={max_sheets}")
    return plans


def _write_excel(plans, path, run_id=None):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    workbook = Workbook(write_only=True)
    workbook.properties.identifier = run_id
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="244A73")
    text_alignment = Alignment(vertical="top", wrap_text=True)
    number_alignment = Alignment(vertical="top")
    widths = {
        "阶段路径": 42,
        "父路径": 36,
        "特征": 30,
        "阶段名称": 26,
        "筛选器": 28,
        "筛选方法": 24,
        "指标名称": 32,
        "关联特征": 30,
        "筛选原因": 54,
        "停止原因": 54,
        "原因": 54,
        "统计口径": 46,
        "文本值": 38,
        "参数": 64,
        "补充信息": 64,
        "候选子集": 48,
        "fit_id": 36,
        "内容": 64,
    }
    try:
        for sheet_name, _, frame, start, end in plans:
            sheet = workbook.create_sheet(sheet_name)
            sheet.freeze_panes = "A2"
            sheet.row_dimensions[1].height = 30
            for position, column in enumerate(frame.columns, start=1):
                sheet.column_dimensions[get_column_letter(position)].width = widths.get(column, 18)
            if len(frame.columns):
                sheet.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{end-start+1}"

            def cells(values, header=False):
                row = []
                for value in values:
                    value = _excel_value(value)
                    if isinstance(value, str) and len(value) > 32767:
                        raise ValidationError("Excel 单元格文本超出 32767 字符；请仅保存 JSON，不会静默截断")
                    cell = WriteOnlyCell(sheet, value=value)
                    if isinstance(value, str):
                        cell.data_type = "s"  # Includes =, +, -, @: never interpret user text as a formula.
                    cell.alignment = text_alignment if isinstance(value, str) else number_alignment
                    if header:
                        cell.font = header_font
                        cell.fill = header_fill
                    row.append(cell)
                return row

            sheet.append(cells(frame.columns, header=True))
            for row in frame.iloc[start:end].itertuples(index=False, name=None):
                sheet.append(cells(row))
        workbook.save(path)
    except BaseException:
        # Flush/close write-only XML generators before removing their private temporary files.
        # Cleanup must not replace the original serialization or IO error.
        for sheet in workbook.worksheets:
            try:
                if not sheet.closed:
                    sheet.close()
                writer = getattr(sheet, "_writer", None)
                if writer is not None and Path(writer.out).exists():
                    writer.cleanup()
            except Exception as error:
                logger.warning("筛选报告 Excel 临时资源清理失败：%s", error)
        raise
    finally:
        workbook.close()


def save_selection_report(
    report,
    path,
    *,
    formats=("xlsx", "json"),
    overwrite=False,
    max_rows_per_sheet=1048575,
    max_excel_cells=5000000,
    max_sheets=200,
    by_selector=False,
    allow_incomplete=False,
):
    """将完整报告保存为同一版本制品，返回实际路径、校验摘要和映射。

    **参数**

    :param report: SelectionReport 快照。
    :param path: 输出名称或带 .xlsx/.json 后缀的期望名称；实际文件位于版本目录。
    :param formats: 支持 xlsx/json，默认同时生成。
    :param overwrite: 是否允许更新同名发布清单，不删除旧版本。

    **参考样例**

    >>> paths = report.save('输出/筛选报告', overwrite=True)
    >>> paths['路径']['xlsx']
    """
    if not isinstance(report, SelectionReport):
        raise ValidationError("保存对象必须为 SelectionReport")
    if not report._metadata.get("完整", True) and not allow_incomplete:
        raise ValidationError("报告不完整，默认拒绝保存；如需诊断文档请显式设置 allow_incomplete=True")
    if isinstance(formats, str):
        formats = (formats,)
    if not formats or any(item not in {"xlsx", "json"} for item in formats) or len(set(formats)) != len(formats):
        raise ValidationError("formats 只能包含不重复的 xlsx 和 json")
    output = Path(path).expanduser().resolve()
    if output.suffix and output.suffix.lower() not in {".xlsx", ".json"}:
        raise ValidationError("筛选报告文件后缀只支持 .xlsx 或 .json，也可不指定后缀")
    name = output.stem if output.suffix else output.name
    plans = (
        _plan_excel(report, max_rows_per_sheet, max_excel_cells, max_sheets, by_selector) if "xlsx" in formats else []
    )
    mapping = [
        {"工作表": sheet, "来源": source, "起始行": start, "结束行": end, "行数": end - start}
        for sheet, source, _, start, end in plans
    ]
    # Core never imports this module; reuse the existing publication primitive only at save time.
    from ..skills_runtime.artifacts import ArtifactTransaction

    transaction = ArtifactTransaction({"directory": str(output.parent), "name": name, "overwrite": overwrite})
    run_id = uuid4().hex
    paths = {}
    with transaction:
        if "json" in formats:
            staged = transaction.stage_path(f"{name}.json")
            _write_json(report, staged, run_id, mapping)
            paths["json"] = transaction.publish(staged, artifact_type="selection_report_json")["path"]
        if "xlsx" in formats:
            staged = transaction.stage_path(f"{name}.xlsx")
            _write_excel(plans, staged, run_id=run_id)
            paths["xlsx"] = transaction.publish(staged, artifact_type="selection_report_excel")["path"]
        metadata = transaction.stage_path(f"{name}.index.json")
        metadata.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "协议版本": SCHEMA_VERSION,
                    "工作表映射": mapping,
                    "表行数": {key: len(getattr(report, "_" + key)) for key in _TABLES},
                    "完整": report._metadata.get("完整", True),
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        transaction.publish(metadata, artifact_type="selection_report_index")
    manifest = json.loads(transaction.manifest_path.read_text(encoding="utf-8"))
    return {
        "路径": paths,
        "清单": str(transaction.manifest_path),
        "run_id": run_id,
        "协议版本": SCHEMA_VERSION,
        "制品": manifest["制品"],
        "工作表映射": mapping,
        "表行数": {key: len(getattr(report, "_" + key)) for key in _TABLES},
        "完整": report._metadata.get("完整", True),
        "诊断": copy.deepcopy(report._metadata.get("诊断", []) + transaction.cleanup_errors),
    }


__all__ = ["save_selection_report", "load_selection_report", "to_report_result"]
