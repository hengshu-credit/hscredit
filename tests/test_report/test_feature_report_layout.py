import types

import pandas as pd

from hscredit.excel import ExcelWriter
from hscredit.report.feature_analyzer import auto_feature_analysis
import hscredit.report.feature_analyzer as feature_analyzer_module


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


def _has_merged_range(ws, min_row, min_col, max_row, max_col):
    return any(
        cell_range.min_row == min_row
        and cell_range.min_col == min_col
        and cell_range.max_row == max_row
        and cell_range.max_col == max_col
        for cell_range in ws.merged_cells.ranges
    )


def test_auto_feature_analysis_system_gap(monkeypatch):
    # 屏蔽绘图函数，避免真实生成图片
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 1, 0, 1, 1, 1],
    })

    def run_for_system(system_name, sheet_name):
        writer = ExcelWriter(system=system_name)
        writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

        auto_feature_analysis(
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
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 1, 0, 1, 1, 1],
    })

    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

    auto_feature_analysis(
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


def test_auto_feature_analysis_keeps_overdue_binning_mode(monkeypatch):
    calls = []

    def fake_feature_bin_stats(data, feature, **kwargs):
        calls.append({"columns": list(data.columns), **kwargs})
        return pd.DataFrame({
            "指标名称": [feature],
            "指标含义": [feature],
            "分箱标签": ["(0, 1]"],
            "样本总数": [len(data)],
            "样本占比": [1.0],
            "坏样本率": [0.5],
        })

    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", fake_feature_bin_stats)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "mob": [0, 1, 4, 6],
    })
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        overdue="mob",
        dpds=3,
        excel_writer=writer,
        sheet="overdue_mode",
        pictures=[],
        output_dir="model_report",
    )

    assert calls
    assert calls[0]["overdue"] == ["mob"]
    assert calls[0]["dpds"] == [3]
    assert calls[0]["target"] == "mob 3+"
    assert calls[0]["columns"] == ["x", "mob 3+", "mob"]


def test_auto_feature_analysis_missing_rate_and_summary_links(monkeypatch):
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0],
        "target": [0, 0, 1, 1],
    })
    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        excel_writer=writer,
        sheet="missing_and_links",
        pictures=["bin"],
        output_dir="model_report",
    )

    ws = writer.get_sheet_by_name("missing_and_links")
    feature_title_row, _ = _get_feature_title_and_table_header_rows(ws)
    assert feature_title_row is not None

    feature_title_cell = ws.cell(row=feature_title_row, column=2)
    assert "缺失率: 25.0%" in feature_title_cell.value

    summary_feature_cell = next(
        cell
        for row in ws.iter_rows()
        for cell in row
        if cell.value == "x" and cell.hyperlink is not None
    )
    assert summary_feature_cell.hyperlink.location == f"#'{ws.title}'!{feature_title_cell.coordinate}"
    assert feature_title_cell.hyperlink.location == f"#'{ws.title}'!{summary_feature_cell.coordinate}"


def test_auto_feature_analysis_sample_distribution_uses_model_report_layout(monkeypatch):
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(feature_analyzer_module, "distribution_plot", lambda *args, **kwargs: pd.DataFrame())

    def fake_feature_bin_stats(data, feature, **kwargs):
        return pd.DataFrame({
            "指标名称": [feature],
            "指标含义": [feature],
            "分箱标签": ["(0, 1]"],
            "样本总数": [len(data)],
            "样本占比": [1.0],
            "坏样本率": [0.5],
        })

    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", fake_feature_bin_stats)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "mob": [0, 2, 5, 9],
        "apply_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"]),
    })
    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

    auto_feature_analysis(
        data,
        features=["x"],
        overdue="mob",
        dpds=[3, 7],
        date="apply_date",
        excel_writer=writer,
        sheet="sample_layout",
        pictures=[],
        output_dir="model_report",
    )

    ws = writer.get_sheet_by_name("sample_layout")
    values = [cell.value for row in ws.iter_rows() for cell in row]

    assert "样本总体分布情况" in values
    assert "样本时间分布情况" in values
    assert values.count("整体样本") >= 2
    assert "mob@3" in values
    assert "mob@7" in values

    sample_total_header = next(cell for row in ws.iter_rows() for cell in row if cell.value == "样本总数")
    assert ws.cell(sample_total_header.row - 1, sample_total_header.column).value == "统计详情"
    assert ws.cell(sample_total_header.row - 1, sample_total_header.column - 1).value is None

    data_group_header = next(cell for row in ws.iter_rows() for cell in row if cell.value == "数据分组")
    time_header_row = data_group_header.row
    data_set_header = next(
        ws.cell(time_header_row, col)
        for col in range(1, ws.max_column + 1)
        if ws.cell(time_header_row, col).value == "数据集"
    )
    time_total_header = next(
        ws.cell(time_header_row, col)
        for col in range(1, ws.max_column + 1)
        if ws.cell(time_header_row, col).value == "样本总数"
    )
    assert ws.cell(time_header_row - 1, data_set_header.column).value == "统计详情"
    assert _has_merged_range(
        ws,
        time_header_row - 1,
        data_set_header.column,
        time_header_row - 1,
        time_total_header.column,
    )

    bad_rate_header = next(
        cell
        for row in ws.iter_rows()
        for cell in row
        if cell.value == "mob@3" and ws.cell(cell.row - 1, cell.column).value == "坏样本率"
    )
    assert ws.cell(bad_rate_header.row + 1, bad_rate_header.column).number_format == "0.00%"
