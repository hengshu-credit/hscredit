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
        if isinstance(val, str) and "、数据字段:" in val and feature_title_row is None:
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


def _merged_range_for_row(ws, row, start_col=2):
    for cell_range in ws.merged_cells.ranges:
        if cell_range.min_row == row and cell_range.max_row == row and cell_range.min_col == start_col:
            return cell_range
    return None


def _row_for_value(ws, value, col=2):
    for row in range(1, ws.max_row + 1):
        if ws.cell(row, col).value == value:
            return row
    raise AssertionError(f"未找到单元格值: {value}")


def _fake_feature_bin_stats(data, feature, **kwargs):
    return pd.DataFrame(
        {
            "指标名称": [feature],
            "指标含义": [feature],
            "分箱标签": ["(0, 1]"],
            "样本总数": [len(data)],
            "样本占比": [1.0],
            "坏样本率": [0.5],
            "LIFT值": [1.0],
            "分档KS值": [0.25],
        }
    )


def _conditional_format_colors(ws):
    colors = set()
    for rules in ws.conditional_formatting._cf_rules.values():
        for rule in rules:
            if rule.type == "dataBar":
                colors.add(rule.dataBar.color.rgb)
            elif rule.type == "colorScale":
                colors.update(color.rgb for color in rule.colorScale.color)
    return colors


def test_auto_feature_analysis_defaults_to_non_role_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "apply_date": pd.date_range("2024-01-01", periods=4),
            "target": [0, 1, 0, 1],
            "mob": [0, 4, 1, 8],
            "amount": [10, 20, 30, 40],
        }
    )
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=None,
        target="target",
        overdue="mob",
        dpds=3,
        date="apply_date",
        amount="amount",
        excel_writer=writer,
        sheet="default_features",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("default_features")
    feature_titles = [cell.value for row in ws.iter_rows() for cell in row if isinstance(cell.value, str) and "、数据字段:" in cell.value]
    assert feature_titles == ["4.1、数据字段: x (缺失率: 0.0%)", "4.2、数据字段: amount (缺失率: 0.0%)"]


def test_auto_feature_analysis_can_analyze_amount_as_default_feature(tmp_path):
    data = pd.DataFrame({
        "amount": list(range(1, 41)),
        "target": [0, 1] * 20,
    })
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=None,
        target="target",
        amount="amount",
        excel_writer=writer,
        sheet="amount_as_feature",
        pictures=[],
        output_dir=str(tmp_path),
        bin_params={"method": "quantile", "max_n_bins": 4},
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("amount_as_feature")
    assert _row_for_value(ws, "3.1、数据字段: amount (缺失率: 0.0%)") > 0
    assert _row_for_value(ws, "订单口径") > 0
    assert any(cell.value == "金额口径" for row in ws.iter_rows() for cell in row)


def test_auto_feature_analysis_labels_order_and_amount_tables_in_existing_gap(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "amount": [10, 20, 30, 40],
        }
    )
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        amount="amount",
        excel_writer=writer,
        sheet="metric_titles",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("metric_titles")
    order_title = next(cell for row in ws.iter_rows() for cell in row if cell.value == "订单口径")
    amount_title = next(cell for row in ws.iter_rows() for cell in row if cell.value == "金额口径")
    table_headers = [cell for row in ws.iter_rows() for cell in row if cell.value == "指标名称"]
    order_header, amount_header = table_headers
    order_title_range = _merged_range_for_row(ws, order_title.row, order_title.column)
    amount_title_range = _merged_range_for_row(ws, amount_title.row, amount_title.column)

    assert order_title.row == order_header.row - 1
    assert amount_title.row == amount_header.row - 1
    assert order_title.column == order_header.column
    assert amount_title.column == amount_header.column
    assert order_title.alignment.horizontal == "left"
    assert amount_title.alignment.horizontal == "left"
    assert order_title_range.max_col - order_title_range.min_col + 1 == 8
    assert amount_title_range.max_col - amount_title_range.min_col + 1 == 8


def test_auto_feature_analysis_reserves_title_row_when_amount_gap_is_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
        "amount": [10, 20, 30, 40],
    })
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        amount="amount",
        image_table_gap_rows=0,
        excel_writer=writer,
        sheet="zero_metric_title_gap",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("zero_metric_title_gap")
    feature_title = next(cell for row in ws.iter_rows() for cell in row if cell.value == "3.1、数据字段: x (缺失率: 0.0%)")
    order_title = next(cell for row in ws.iter_rows() for cell in row if cell.value == "订单口径")
    amount_title = next(cell for row in ws.iter_rows() for cell in row if cell.value == "金额口径")
    table_headers = [cell for row in ws.iter_rows() for cell in row if cell.value == "指标名称"]
    assert order_title.row == table_headers[0].row - 1
    assert amount_title.row == table_headers[1].row - 1
    assert feature_title.row < order_title.row


def test_auto_feature_analysis_left_aligns_feature_effect_section(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    data = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        excel_writer=writer,
        sheet="section_alignment",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("section_alignment")
    section_row = _row_for_value(ws, "3、数值类特征 OR 评分效果评估")
    assert ws.cell(section_row, 2).alignment.horizontal == "left"


def test_auto_feature_analysis_numbers_titles_and_formats_feature_summary(monkeypatch, tmp_path):
    """章节应连续编号，summary 比率列显示百分数且仅 KS/IV 使用色阶。"""
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    def fake_summary(self, **kwargs):
        return pd.DataFrame(
            {
                "特征名": ["x"],
                "缺失率": [0.25],
                "众数占比": [0.50],
                "覆盖占比": [0.60],
                "零值率": [0.10],
                "负值率": [0.20],
                "重复率": [0.40],
                "通过率": [0.70],
                "IV": [0.15],
                "KS": [0.30],
                "PSI": [0.08],
            }
        )

    monkeypatch.setattr(pd.DataFrame, "summary", fake_summary, raising=False)
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        corr=False,
        excel_writer=writer,
        sheet="numbered_summary",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("numbered_summary")
    title_rows = [_row_for_value(ws, title) for title in [
        "1、样本总体分布情况",
        "2、变量综合统计",
        "3、数值类特征 OR 评分效果评估",
        "3.1、数据字段: x (缺失率: 0.0%)",
    ]]
    assert title_rows == sorted(title_rows)

    summary_title_row = _row_for_value(ws, "2、变量综合统计")
    header_row = summary_title_row + 2
    data_row = header_row + 1
    header_columns = {
        ws.cell(header_row, column).value: column
        for column in range(1, ws.max_column + 1)
        if ws.cell(header_row, column).value is not None
    }
    percent_columns = {"缺失率", "众数占比", "覆盖占比", "零值率", "负值率", "重复率", "通过率", "KS", "PSI"}
    assert all(ws.cell(data_row, header_columns[column]).number_format == "0.00%" for column in percent_columns)
    assert ws.cell(data_row, header_columns["IV"]).number_format != "0.00%"

    color_scale_headers = set()
    for conditional_range, rules in ws.conditional_formatting._cf_rules.items():
        if not any(rule.type == "colorScale" for rule in rules):
            continue
        for cell_range in conditional_range.sqref.ranges:
            color_scale_headers.add(ws.cell(header_row, cell_range.min_col).value)
    assert color_scale_headers == {"KS", "IV"}


def test_auto_feature_analysis_defaults_condition_color_to_secondary_theme(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)

    data = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x"],
        target="target",
        excel_writer=writer,
        sheet="default_condition_color",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("default_condition_color")
    colors = _conditional_format_colors(ws)
    assert "00F76E6C" in colors
    assert "002639E9" not in colors


def test_auto_feature_analysis_applies_custom_condition_color_everywhere(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", _fake_feature_bin_stats)
    monkeypatch.setattr(feature_analyzer_module, "distribution_plot", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(feature_analyzer_module, "corr_plot", lambda *args, **kwargs: None)

    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [4, 3, 2, 1],
            "target": [0, 1, 0, 1],
            "apply_date": pd.date_range("2024-01-01", periods=4),
        }
    )
    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = types.MethodType(_mock_insert_pic2sheet, writer)

    auto_feature_analysis(
        data,
        features=["x", "y"],
        target="target",
        date="apply_date",
        corr=True,
        condition_color="12AB34",
        excel_writer=writer,
        sheet="custom_condition_color",
        pictures=[],
        output_dir=str(tmp_path),
        n_jobs=1,
    )

    ws = writer.get_sheet_by_name("custom_condition_color")
    colors = _conditional_format_colors(ws)
    assert "0012AB34" in colors
    assert "002639E9" not in colors


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


def test_auto_feature_analysis_title_merges_follow_actual_content_width(monkeypatch):
    def fake_feature_bin_stats(data, feature, **kwargs):
        return pd.DataFrame({
            "指标名称": [feature],
            "指标含义": [feature],
            "分箱标签": ["(0, 1]"],
            "样本总数": [len(data)],
            "样本占比": [1.0],
            "坏样本率": [0.5],
            "LIFT值": [1.0],
            "分档KS值": [0.25],
        })

    monkeypatch.setattr(feature_analyzer_module, "feature_bin_stats", fake_feature_bin_stats)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [4, 3, 2, 1],
        "target": [0, 1, 0, 1],
        "amount": [10, 20, 30, 40],
    })
    writer = ExcelWriter(system="windows")

    auto_feature_analysis(
        data,
        features=["x", "y"],
        target="target",
        amount="amount",
        excel_writer=writer,
        sheet="dynamic_titles",
        pictures=[],
        output_dir="model_report",
    )

    ws = writer.get_sheet_by_name("dynamic_titles")
    main_title = _merged_range_for_row(ws, 2)
    feature_section_row = _row_for_value(ws, "3、数值类特征 OR 评分效果评估")
    first_feature_row = _row_for_value(ws, "3.1、数据字段: x (缺失率: 0.0%)")
    second_feature_row = _row_for_value(ws, "3.2、数据字段: y (缺失率: 0.0%)")
    sample_title_row = _row_for_value(ws, "1、样本总体分布情况")
    feature_module_max_col = max(
        _merged_range_for_row(ws, first_feature_row).max_col,
        _merged_range_for_row(ws, second_feature_row).max_col,
    )

    assert main_title.max_col == ws.max_column
    assert main_title.max_col != 35
    assert _merged_range_for_row(ws, feature_section_row).max_col == feature_module_max_col
    assert _merged_range_for_row(ws, first_feature_row).max_col == feature_module_max_col
    assert _merged_range_for_row(ws, sample_title_row).max_col < ws.max_column


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

    assert "1、样本总体分布情况" in values
    assert "2、样本时间分布情况" in values
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
