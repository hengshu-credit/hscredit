import types

import pandas as pd

from hscredit.report.excel import ExcelWriter
from hscredit.report.feature_report import auto_feature_analysis_report
import hscredit.report.feature_report as feature_report_module


def _mock_insert_pic2sheet(self, worksheet, fig, insert_space, figsize=(600, 250)):
    if isinstance(insert_space, str):
        row = int(''.join(ch for ch in insert_space if ch.isdigit()))
        col = 2
    else:
        row, col = insert_space
    # 固定返回占用高度，便于断言不同系统下间隔差异
    return row + 20, col + 8


def _get_feature_title_and_table_header_rows(ws):
    feature_title_row = None
    table_header_row = None

    for r in range(1, ws.max_row + 1):
        val = ws.cell(row=r, column=2).value
        if isinstance(val, str) and val.startswith("数据字段:") and feature_title_row is None:
            feature_title_row = r
        if val == "指标名称" and feature_title_row is not None and r > feature_title_row:
            table_header_row = r
            break

    return feature_title_row, table_header_row


def test_auto_feature_analysis_report_system_gap(monkeypatch):
    # 屏蔽绘图函数，避免真实生成图片
    monkeypatch.setattr(feature_report_module, "bin_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 1, 0, 1, 1, 1],
    })

    def run_for_system(system_name, sheet_name):
        writer = ExcelWriter(system=system_name)
        writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

        auto_feature_analysis_report(
            data,
            features=["x"],
            target="target",
            excel_writer=writer,
            sheet=sheet_name,
            pictures=["bin"],
            output_dir="model_report",
        )

        ws = writer.get_sheet_by_name(sheet_name)
        feature_title_row, table_header_row = _get_feature_title_and_table_header_rows(ws)

        assert feature_title_row is not None
        assert table_header_row is not None
        return table_header_row - feature_title_row

    windows_distance = run_for_system("windows", "win_gap")
    mac_distance = run_for_system("mac", "mac_gap")

    # windows 默认比 mac 多 1 行间隔，避免表头被图片覆盖
    assert windows_distance == mac_distance + 1


def test_feature_title_end_space_with_return_cols(monkeypatch):
    monkeypatch.setattr(feature_report_module, "bin_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 1, 0, 1, 1, 1],
    })

    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

    auto_feature_analysis_report(
        data,
        features=["x"],
        target="target",
        excel_writer=writer,
        sheet="return_cols_span",
        pictures=["bin"],
        bin_params={"return_cols": ["坏样本率"]},
        output_dir="model_report",
    )

    ws = writer.get_sheet_by_name("return_cols_span")
    feature_title_row, _ = _get_feature_title_and_table_header_rows(ws)
    assert feature_title_row is not None

    # 默认 merge_columns=5 列，return_cols=1 列 => 标题应覆盖 6 列（从 B 到 G）
    expected_span = 6
    actual_span = None

    for merged_range in ws.merged_cells.ranges:
        if merged_range.min_row == feature_title_row and merged_range.max_row == feature_title_row and merged_range.min_col == 2:
            actual_span = merged_range.max_col - merged_range.min_col + 1
            break

    assert actual_span == expected_span
