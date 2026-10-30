"""模型报告与规则报告共用的 Excel 标题布局。"""

import re
from typing import Any, Optional

title_pattern = re.compile(r"^(?:[一二三四五六七八九十]+|\d+(?:\.\d+)*)、")


def _report_title_level(row_idx: int, value: Any) -> Optional[int]:
    if not isinstance(value, str) or not value:
        return None
    if row_idx == 2:
        return 0
    if not title_pattern.match(value):
        return None
    prefix = value.split("、", 1)[0]
    if re.match(r"^\d+(?:\.\d+)*$", prefix):
        return prefix.count(".") + 1
    return 1


def _content_max_col(worksheet, start_row: int, end_row: int, start_col: int = 2) -> int:
    if end_row < start_row:
        return start_col
    max_content_col = start_col
    for row in worksheet.iter_rows(
        min_row=start_row,
        max_row=end_row,
        min_col=start_col,
        max_col=worksheet.max_column,
    ):
        for cell in row:
            if cell.value is not None:
                max_content_col = max(max_content_col, cell.column)
    for cell_range in worksheet.merged_cells.ranges:
        if cell_range.max_row >= start_row and cell_range.min_row <= end_row and cell_range.max_col >= start_col:
            max_content_col = max(max_content_col, cell_range.max_col)
    return max_content_col


def _merge_report_title(worksheet, row_idx: int, end_col: int, start_col: int = 2) -> None:
    for cell_range in list(worksheet.merged_cells.ranges):
        if cell_range.min_row == row_idx and cell_range.max_row == row_idx and cell_range.min_col == start_col:
            worksheet.unmerge_cells(str(cell_range))
    if end_col > start_col:
        worksheet.merge_cells(
            start_row=row_idx,
            start_column=start_col,
            end_row=row_idx,
            end_column=end_col,
        )


def adjust_report_title_merges(worksheet) -> None:
    title_rows = [
        (row_cell.row, level)
        for row_cells in worksheet.iter_rows(min_col=2, max_col=2)
        for row_cell in row_cells
        for level in [_report_title_level(row_cell.row, row_cell.value)]
        if level is not None
    ]
    for idx, (row_idx, level) in enumerate(title_rows):
        if level == 0:
            end_row_for_title = worksheet.max_row
        else:
            next_boundary = next(
                (next_row for next_row, next_level in title_rows[idx + 1 :] if next_level <= level),
                None,
            )
            end_row_for_title = next_boundary - 1 if next_boundary is not None else worksheet.max_row
        max_content_col = _content_max_col(worksheet, row_idx + 1, end_row_for_title)
        _merge_report_title(worksheet, row_idx, max_content_col)
