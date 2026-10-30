"""树挖掘报告的统一 API、原始行统计与 Excel 落盘契约。"""

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from hscredit import MultiwayTreeExtractor
from hscredit.excel import ExcelWriter
from hscredit.report.mining import DecisionTreeAnalyzer, ManualTreeExtractor


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "评分": np.arange(48),
            "多头": np.tile(np.arange(12), 4),
            "FPD": ([0] * 8 + [1] * 4) * 4,
            "MOB1": [0, 2, 3, 4, 7, 8] * 8,
            "金额": np.arange(48) + 100,
        }
    )


@pytest.fixture(params=[MultiwayTreeExtractor, ManualTreeExtractor, DecisionTreeAnalyzer])
def tree(request, data):
    if request.param is MultiwayTreeExtractor:
        model = request.param(target="FPD", features=["评分", "多头"], max_depth=2, min_samples_leaf=2, n_jobs=1)
        model.fit(data).manual_split(feature="评分", thresholds=[15, 30])
    else:
        model = request.param(target="FPD", n_jobs=1)
        model.fit(data, features=["评分", "多头"], max_depth=2, min_samples_leaf=2)
        if request.param is ManualTreeExtractor:
            model.manual_split(data, feature="评分", threshold=22, node=0)
    return model


@pytest.mark.parametrize("long_format", [True, False])
def test_shared_rule_apis_and_excel(tree, data, tmp_path, long_format):
    datasets = {"训练集": data, "验证集": data.iloc[::2], "测试集": data.iloc[1::2]}
    expected = tree.report(
        datasets, overdue=["MOB1"], dpds=[7, 3, 0], overdue_operator=">=", amount="金额", long_format=long_format
    )
    path = tmp_path / "reports" / "挖掘.xlsx"
    actual = tree.report(
        datasets,
        overdue=["MOB1"],
        dpds=[7, 3, 0],
        overdue_operator=">=",
        amount="金额",
        save=path,
        long_format=long_format,
    )
    assert list(actual) == list(datasets)
    for key in datasets:
        pd.testing.assert_frame_equal(actual[key], expected[key])
    pd.testing.assert_frame_equal(tree.get_rule_table(datasets=data), tree.report(data))
    pd.testing.assert_frame_equal(tree.get_rule_table(data, "FPD"), tree.report(data, target="FPD"))
    all_rules = tree.get_rules(leaf_only=False)
    assert len(all_rules) == len(tree.get_rule_table())
    assert all_rules[0].expr == "True"
    assert len(tree.get_rules()) == len(tree.get_rule_table(leaf_only=True))
    book = load_workbook(path)
    assert book.sheetnames == ["组合策略汇总", "组合策略汇总_特征分箱"]
    assert len(book["组合策略汇总"]._images) == 8
    detail = book["组合策略汇总"]
    assert not any(c.value == "完整节点信息" for row in detail for c in row)
    headings = ["节点编号", "指标含义", "组合策略", "是否叶子", "是否调整"]
    headers = [
        (r, c)
        for r in range(1, detail.max_row + 1)
        for c in range(2, detail.max_column - 3)
        if [detail.cell(r, c + j).value for j in range(5)] == headings
    ]
    assert len(headers) == 8
    node_ids = tree.get_rule_table()["节点编号"].tolist()
    expressions = tree.get_rule_table()["组合策略"].tolist()
    for ordinal, ((header, column), image) in enumerate(zip(headers, detail._images)):
        image_start = image.anchor._from.row + 1
        assert image_start < header
        assert image.anchor._from.col + 1 == column
        reserved_pixels = sum(
            (detail.row_dimensions[r].height or detail.sheet_format.defaultRowHeight or 15) / 0.75
            for r in range(image_start, header)
        )
        assert reserved_pixels >= image.anchor.ext.cy / 9525
        repetitions = 3 if long_format and ordinal >= 2 else 1
        ids = np.repeat(node_ids, repetitions).tolist()
        exprs = np.repeat(expressions, repetitions).tolist()
        assert [detail.cell(header + 1 + i, column).value for i in range(len(ids))] == ids
        assert [detail.cell(header + 1 + i, column + 2).value for i in range(len(ids))] == exprs
        if ordinal % 2:
            assert header == headers[ordinal - 1][0]
            gap = column - 1
            assert all(detail.cell(r, gap).value is None for r in range(image_start, header + len(ids) + 1))
    sheet = book["组合策略汇总"]
    values = [cell.value for row in sheet for cell in row]
    assert sum(isinstance(v, str) and "最终树应用：" in v for v in values) == 3
    assert "MOB1>=7" in values
    assert "组合策略" in values and "命中数" in values and "命中率" in values
    assert "样本整体坏率" in values
    assert "节点编号" in values
    assert "是否调整" in values
    assert "规则数（含根节点）" in values and "树深度（根为0）" in values
    assert "AUC" in values and "KS" in values and "订单维度" in values and "金额维度" in values
    assert "[0, 0.25)" in values and "[5, inf)" in values
    # 根节点金额按原始行直接核算，而非与另一个报告入口互相比较。
    first = actual["训练集"].iloc[0]
    count_key = "样本总数" if long_format else ("分箱详情", "样本总数")
    bad_key = "坏样本数" if long_format else ("MOB1>=7", "坏样本数")
    assert first[count_key] == data["金额"].sum()
    assert first[bad_key] == data.loc[data.MOB1 >= 7, "金额"].sum()
    assert not any(c.data_type == "e" for ws in book for row in ws for c in row)
    assert all(not c.alignment.wrap_text for ws in book for row in ws for c in row if c.value is not None)
    assert all(dimension.width <= 50 for ws in book for dimension in ws.column_dimensions.values())
    for ws in book:
        dimensions = sorted(ws.column_dimensions.values(), key=lambda dim: dim.min)
        assert all(left.max < right.min for left, right in zip(dimensions, dimensions[1:]))
    assert sheet.column_dimensions["B"].width == ExcelWriter().style_sheet.column_dimensions["B"].width
    book.close()


def test_report_containers_and_external_writer(tree, data, tmp_path):
    for frames in ([data, data.iloc[:24]], (data, data.iloc[:24])):
        assert len(tree.report(frames)) == 2
    path = tmp_path / "合并.xlsx"
    with ExcelWriter() as writer:
        writer.set_filename(str(path))
        writer.get_sheet_by_name("业务原表")["A1"] = "保留"
        tree.report(data, sheet_name="FPD", excel_writer=writer, save=path)
        tree.report(data, overdue=["MOB1"], dpds=[7, 3, 0], sheet_name="MOB", excel_writer=writer)
    book = load_workbook(path)
    assert book["业务原表"]["A1"].value == "保留"
    assert len(book["FPD"]._images) == len(book["MOB"]._images) == 2
    book.close()


def test_export_imported_binary_tree_without_training_data(data, tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=2).fit(data[["评分", "多头"]], data.FPD)
    tree = ManualTreeExtractor.from_sklearn(clf, features=["评分", "多头"], target="FPD")
    path = tmp_path / "导入树.xlsx"
    tree.report(data, save=path)
    with path.open("rb") as stream:
        book = load_workbook(stream)
        assert len(book["组合策略汇总"]._images) == 2
        book.close()


def test_failed_export_preserves_existing_report(tree, data, tmp_path, monkeypatch):
    path = tmp_path / "原报告.xlsx"
    path.write_bytes(b"existing report")

    def fail(*args, **kwargs):
        raise ValueError("绘图失败")

    monkeypatch.setattr("hscredit.core.viz.plot_tree", fail)
    with pytest.raises(ValueError, match="绘图失败"):
        tree.report(data, save=path)
    assert path.read_bytes() == b"existing report"


def test_node_details_after_local_edits(data):
    tree = MultiwayTreeExtractor(target="FPD", features=["评分", "多头"], max_depth=3, n_jobs=1).fit(data)
    tree.manual_split(feature="评分", thresholds=[15, 30])
    child = tree.nodes_[0].children[1]
    tree.manual_split(node=child, feature="多头", thresholds=[4, 8])
    table = tree.get_node_table().set_index("节点编号")
    assert set(table.index) == set(tree.nodes_)
    assert str(table["父节点"].dtype) == "Int64"
    for node, rule in zip(tree.nodes_, tree.get_rules(leaf_only=False)):
        rows = data.loc[rule.predict(data)]
        assert table.loc[node, "样本总数"] == len(rows)
        assert table.loc[node, "坏样本数"] == rows.FPD.sum()
        assert table.loc[node, "坏样本率"] == pytest.approx(rows.FPD.mean())
        assert table.loc[node, "LIFT值"] == pytest.approx(rows.FPD.mean() / data.FPD.mean())
        assert table.loc[node, "指标含义"]
    tree.prune(node=child)
    after = tree.get_node_table()
    assert len(after) == len(tree.nodes_) < len(table)
    pd.testing.assert_frame_equal(tree.get_rule_table(data), tree.get_rule_table(datasets=data))
    with pytest.raises(TypeError, match="datasets"):
        tree.get_rule_table(data=data)
    with pytest.raises(TypeError):
        tree.get_rule_table(data, True)


def test_no_export_without_save(tree, data, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("普通 report 不应触发绘图")

    monkeypatch.setattr("hscredit.core.viz.plot_tree", fail)
    assert not tree.report(data).empty


def test_dataset_overview_and_grey_labels_match_model_report(tree, data, tmp_path):
    from hscredit.report._sample_stats import build_sample_stats_table
    from hscredit.utils.targets import bind_targets

    data = data.assign(放款时间=pd.date_range("2026-01-01", periods=len(data), freq="D"))
    path = tmp_path / "多标签样本.xlsx"
    expected = tree.report(data, overdue=["MOB1"], dpds=[7, 3, 0], del_grey=True, leaf_only=True)
    result = tree.report(
        data,
        overdue=["MOB1"],
        dpds=[7, 3, 0],
        del_grey=True,
        leaf_only=True,
        date="放款时间",
        save=path,
        project_desc="按放款样本复核",
        data_source="样本表",
    )
    pd.testing.assert_frame_equal(result, expected)
    prepared, bound = bind_targets(data, overdue=["MOB1"], dpds=[7, 3, 0], del_grey=True)
    stats, _ = build_sample_stats_table(
        ["数据集1"], [bound.y_map(prepared)], bound.names(), target_specific_totals=True, n_jobs=1
    )
    book = load_workbook(path)
    sheet = book["组合策略汇总"]
    title = next(c.row for row in sheet for c in row if c.value == "3、数据样本统计")
    header = title + 5
    assert sheet.cell(header + 1, 2).value == "数据集"
    values = [sheet.cell(header + 2, col).value for col in range(3, 3 + len(stats.columns))]
    np.testing.assert_allclose(values, stats.iloc[0].to_numpy(dtype=float))
    assert "目标标签" not in [sheet.cell(header + 1, col).value for col in range(2, 3 + len(stats.columns))]
    assert any(c.value == "4、样本分布情况" for row in sheet for c in row)
    assert any(c.value == "样本区间" for row in sheet for c in row)
    assert any(c.value == "样本表" for row in sheet for c in row)
    book.close()


def test_export_target_callable_evaluated_once_per_dataset(tree, data, tmp_path):
    calls = []

    def target(frame):
        calls.append(len(frame))
        return frame.FPD

    result = tree.report({"训练集": data, "测试集": data.iloc[:20]}, target=target, save=tmp_path / "动态标签.xlsx")
    assert calls == [48, 20]
    assert len(result) == 2


@pytest.mark.parametrize("cls", [ManualTreeExtractor, DecisionTreeAnalyzer, MultiwayTreeExtractor])
def test_root_only_tree_rules_and_reports(cls, data):
    if cls is MultiwayTreeExtractor:
        tree = cls(target="FPD", features=["评分"], min_samples_split=100, n_jobs=1).fit(data)
    else:
        tree = cls(target="FPD", n_jobs=1).fit(data, features=["评分"], min_samples_split=100)
    assert tree.get_rules()[0].expr == "True"
    table = tree.get_rule_table(leaf_only=True)
    assert table["节点编号"].tolist() == [0]
    assert table["样本总数"].tolist() == [len(data)]
    if hasattr(tree, "get_leaf_node_ids"):
        assert tree.get_leaf_node_ids() == [0]


@pytest.mark.parametrize("cls", [ManualTreeExtractor, MultiwayTreeExtractor])
def test_rule_table_marks_adjusted_subtree_after_split_and_prune(cls, data):
    if cls is MultiwayTreeExtractor:
        tree = cls(target="FPD", features=["评分", "多头"], max_depth=3, min_samples_leaf=2, n_jobs=1).fit(data)
        assert set(tree.get_rule_table()["是否调整"]) == {"否"}
        tree.manual_split(node=0, feature="评分", thresholds=[15, 30])
        tree.prune(node=tree.nodes_[0].children[0])
    else:
        tree = cls(target="FPD", n_jobs=1).fit(data, features=["评分", "多头"], max_depth=3, min_samples_leaf=2)
        assert set(tree.get_rule_table()["是否调整"]) == {"否"}
        tree.manual_split(data, feature="评分", threshold=22, node=0)
        tree.delete_node(node=1)
    table = tree.get_rule_table()
    assert list(table.columns[:5]) == ["节点编号", "指标含义", "组合策略", "是否叶子", "是否调整"]
    assert set(table["是否调整"]) == {"是"}
    multi = tree.report(data, overdue=["MOB1"], dpds=[7, 3, 0])
    assert set(multi["是否调整"]) == {"是"}


def test_tree_views_reuse_final_routes_and_distinguish_count_and_amount(data):
    from hscredit.report.mining._tree_report import _tree_view
    from hscredit.core.viz.tree_plots import _extract_tree_data

    tree = MultiwayTreeExtractor(target="FPD", features=["评分"], n_jobs=1).fit(data)
    tree.manual_split(feature="评分", thresholds=[15, 30])
    counts_before = {node: current.counts.copy() for node, current in tree.nodes_.items()}
    evaluation = data.iloc[:20]
    original_edges = _extract_tree_data(tree)["edges"]
    for amount in [None, "金额"]:
        table = tree.report(evaluation, amount=amount, round_digits=False)
        view, _ = _tree_view(tree, table, amount is not None)
        for node, rule in zip(view["nodes"], tree.get_rules(leaf_only=False)):
            mask = rule.predict(evaluation)
            expected = evaluation.loc[mask, amount].sum() if amount else mask.sum()
            assert node["n_samples"] == expected
        assert view["nodes"][-1]["n_samples"] == 0
        assert len(view["edges"]) == 3
        assert view["edges"] == original_edges  # 连线只保留阈值/分组条件，数量在节点卡片显示。
    for node, before in counts_before.items():
        np.testing.assert_array_equal(tree.nodes_[node].counts, before)
    assert tree.edit_history_[-1]["操作"] == "指定阈值分裂"
    assert tree.edit_history_[-1]["切分点"] == [15, 30]


def test_model_metrics_match_raw_leaf_predictions_and_weights(tree, data):
    from sklearn.metrics import roc_auc_score
    from hscredit.report.mining._tree_report import _prediction_metrics

    sample = data.iloc[::2]
    scores = pd.Series(index=sample.index, dtype=float)
    for rule in tree.get_rules():
        training_rows = data.loc[rule.predict(data)]
        scores.loc[rule.predict(sample)] = training_rows.FPD.mean()
    result = _prediction_metrics(tree, {"验证集": sample}, [{"FPD": sample.FPD}], "金额")
    for basis, weights in [("订单", np.ones(len(sample))), ("金额", sample["金额"].to_numpy())]:
        expected_auc = roc_auc_score(sample.FPD, scores, sample_weight=weights)
        grouped = (
            pd.DataFrame(
                {
                    "分数": scores.to_numpy(),
                    "坏": weights * sample.FPD.to_numpy(),
                    "好": weights * (1 - sample.FPD.to_numpy()),
                }
            )
            .groupby("分数")[["坏", "好"]]
            .sum()
        )
        expected_ks = (
            (grouped["坏"].cumsum() / grouped["坏"].sum() - grouped["好"].cumsum() / grouped["好"].sum()).abs().max()
        )
        assert result[(basis, "FPD", "AUC")][0] == pytest.approx(expected_auc)
        assert result[(basis, "FPD", "KS")][0] == pytest.approx(expected_ks)


def test_zero_amount_keeps_order_report_and_blank_weighted_auc(data, tmp_path):
    from hscredit.report.mining._tree_report import _prediction_metrics

    data = data.assign(金额=0)
    tree = MultiwayTreeExtractor(target="FPD", features=["评分"], min_samples_leaf=2, n_jobs=1).fit(data)
    metrics = _prediction_metrics(tree, {"样本": data}, [{"FPD": data.FPD}], "金额")
    assert np.isnan(metrics[("金额", "FPD", "AUC")][0])
    assert np.isfinite(metrics[("订单", "FPD", "AUC")][0])
    path = tmp_path / "零金额.xlsx"
    result = tree.report({"样本": data}, amount="金额", save=path)
    assert result["样本"].iloc[0]["样本总数"] == 0
    book = load_workbook(path)
    assert len(book.active._images) == 4
    assert not any(c.data_type == "e" for row in book.active for c in row)
    book.close()


@pytest.mark.parametrize("long_format", [True, False])
@pytest.mark.parametrize("grey", [True, False])
def test_group_statistics_share_overall_schema_and_raw_denominators(data, long_format, grey):
    from sklearn.metrics import roc_auc_score
    from hscredit.utils.targets import bind_targets
    from hscredit.utils.dates import date_groups
    from hscredit.report.mining._tree_report import _sample_tables, _distribution_tables

    data = data.assign(日期=pd.date_range("2026-01-01", periods=len(data)))
    tree = MultiwayTreeExtractor(target="FPD", features=["评分"], min_samples_leaf=2, n_jobs=1).fit(data)
    tree.manual_split(feature="评分", thresholds=[15, 30])
    frames = {"训练": data, "验证": data.iloc[::2]}
    contexts, targets, days, reports = {}, [], [], {}
    for name, frame in frames.items():
        prepared, bound = bind_targets(frame, overdue="MOB1", dpds=[7, 3, 0], del_grey=grey)
        contexts[name] = prepared, bound
        targets.append(bound.y_map(prepared, style="at"))
        days.append(date_groups(frame.日期, "M"))
        reports[name] = tree.report(prepared, target=bound, amount="金额", long_format=False, round_digits=False)
    overall = _sample_tables(tree, frames, targets, reports, "金额", long_format)
    grouped = _distribution_tables(
        tree, frames, contexts, days, "金额", long_format, None, ">", {"amount": "金额", "round_digits": False}
    )
    for basis, (full, table) in enumerate(zip(overall, grouped)):
        group_col = "数据分组" if long_format else ("统计详情", "数据分组")
        assert table.drop(columns=[group_col]).columns.tolist() == full.columns.tolist()
        for name, frame in frames.items():
            for month in date_groups(frame.日期, "M").unique():
                subset = frame.loc[date_groups(frame.日期, "M").eq(month)]
                for dpd in [7, 3, 0]:
                    valid = subset.MOB1.eq(0) | subset.MOB1.gt(dpd) if grey else pd.Series(True, index=subset.index)
                    selected = subset.loc[valid]
                    total = selected["金额"].sum() if basis else len(selected)
                    bad = selected.loc[selected.MOB1.gt(dpd), "金额"].sum() if basis else selected.MOB1.gt(dpd).sum()
                    dataset_col = "数据集" if long_format else ("统计详情", "数据集")
                    rows = table.loc[table[dataset_col].eq(name) & table[group_col].eq(month)]
                    label = f"MOB1@{dpd}"
                    if long_format:
                        row = rows.loc[rows["目标标签"].eq(label)].iloc[0]
                        total_col = "样本总数"
                        bad_col = "坏样本数"
                        auc_col = "AUC"
                    else:
                        row = rows.iloc[0]
                        count = "样本总数"
                        total_col = (count, label) if grey else ("统计详情", count)
                        bad_col = ("坏样本数", label)
                        auc_col = ("AUC", label)
                    assert row[total_col] == total
                    assert row[bad_col] == bad
                    if selected.MOB1.gt(dpd).nunique() == 2:
                        expected = roc_auc_score(
                            selected.MOB1.gt(dpd),
                            tree.predict_proba(selected)[:, 1],
                            sample_weight=selected["金额"] if basis else None,
                        )
                        assert row[auc_col] == pytest.approx(expected)


def test_feature_sheet_reuses_strategy_binning_and_caps_width(data, tmp_path, monkeypatch):
    from openpyxl.utils.cell import range_boundaries
    import hscredit.report.feature_analyzer as module

    tree = MultiwayTreeExtractor(target="FPD", features=["评分", "多头"], n_jobs=1).fit(data)
    tree.manual_split(feature="评分", thresholds=[15, 30])
    original, calls = module.feature_binning_summary, []

    def capture(*args, **kwargs):
        calls.append((kwargs["features"], kwargs["methods"], kwargs["long_format"]))
        return original(*args, **kwargs)

    adjust_width = ExcelWriter.adjust_columns_width

    def feature_width_only(writer, worksheet, **kwargs):
        assert worksheet.title.endswith("_特征分箱")
        assert kwargs["max_width"] == 25
        return adjust_width(writer, worksheet, **kwargs)

    monkeypatch.setattr(module, "feature_binning_summary", capture)
    monkeypatch.setattr(ExcelWriter, "adjust_columns_width", feature_width_only)
    path = tmp_path / "特征页.xlsx"
    tree.report(
        {"训练": data, "验证": data.iloc[::2]},
        overdue=["MOB1"],
        dpds=[7, 3, 0],
        amount="金额",
        methods=["quantile", "uniform", "tree"],
        del_grey=True,
        save=path,
    )
    assert len(calls) == 4
    assert all(features == ["评分"] and long for features, _, long in calls)
    book = load_workbook(path)
    sheet = book["组合策略汇总_特征分箱"]
    assert all(d.width <= 25 for d in sheet.column_dimensions.values())
    tables = list(sheet.tables.values())
    assert len(tables) == 2
    summary_range = range_boundaries(sheet.auto_filter.ref)
    ranges = [range_boundaries(table.ref) for table in tables]
    assert summary_range[3] < ranges[0][1] == ranges[1][1]
    assert ranges[1][0] == ranges[0][2] + 2
    for table, bounds in zip(tables, ranges):
        assert table.autoFilter.ref == table.ref
        assert table.tableStyleInfo.name is None
        left, top, right, bottom = bounds
        values = list(sheet.iter_rows(min_row=top, max_row=bottom, min_col=left, max_col=right, values_only=True))
        frame = pd.DataFrame(values[1:], columns=values[0])
        assert set(frame["指标名称"]) == {"评分"}
        assert set(frame["分箱方法"]) == {"quantile", "uniform", "tree"}
        assert set(frame["统计口径"]) == {"订单", "金额"}
        assert "目标标签" in frame or "逾期标签" in frame
    book.close()


def test_report_formats_metric_levels_and_preserves_full_precision(tmp_path):
    from hscredit.report.mining._tree_report import _write_report_table

    rate = 0.123456789012345
    wide = pd.DataFrame(
        {
            ("统计详情", "数据集"): ["训练"],
            ("坏样本率", "FPD"): [rate],
            ("AUC", "FPD"): [0.876543219876543],
            ("KS", "FPD"): [rate],
            ("PSI", "FPD"): [rate],
            ("样本总数", "FPD"): [1234],
            ("样本金额", "FPD"): [1234.56789],
        }
    )
    mixed = pd.DataFrame(
        {
            "统计口径": ["订单", "金额"],
            "坏样本数": [12, 1234.56789],
            "LIFT值": [1, 2],
            "指标IV值": [rate, rate],
            "分档KS值": [rate, rate],
        }
    )
    rule_wide = pd.DataFrame({("分箱详情", "节点编号"): [1], ("AUC", "坏样本率"): [rate]})
    path = tmp_path / "格式.xlsx"
    with ExcelWriter() as writer:
        writer.set_filename(str(path))
        _write_report_table(wide, writer, sheet_name="宽表", percent_cols=["坏样本率", "KS"], decimal=None)
        _write_report_table(mixed, writer, sheet_name="混合口径", decimal=None)
        _write_report_table(rule_wide, writer, sheet_name="规则宽表", decimal=None)
    book = load_workbook(path)
    assert [book["宽表"].cell(4, c).number_format for c in range(3, 9)] == [
        "0.00%",
        "0.0000",
        "0.00%",
        "0.0000",
        "#,##0",
        "#,##0.00",
    ]
    assert book["宽表"]["C4"].value == rate
    assert book["宽表"]["H4"].value == 1234.56789
    assert book["混合口径"]["C3"].number_format == "#,##0"
    assert book["混合口径"]["C4"].number_format == "#,##0.00"
    assert book["混合口径"]["D3"].number_format == "0.0000"
    assert book["混合口径"]["F3"].number_format == "0.00%"
    assert book["规则宽表"]["C4"].number_format == "0.00%"
    book.close()


@pytest.mark.parametrize("amount", [None, "金额"])
def test_feature_ks_auc_use_raw_scores_and_independent_weighted_pairs(amount):
    from hscredit.utils.targets import bind_targets
    from hscredit.report.mining._tree_report import _feature_ranking_metrics

    data = pd.DataFrame(
        {
            "评分": [0, 0, 1, 2, np.nan, np.inf],
            "类别": list("abcdef"),
            "目标": [0, 1, 0, 1, 1, 0],
            "金额": [1, 4, 3, 2, 5, 6],
        }
    )
    prepared, bound = bind_targets(data, target="目标")
    result = _feature_ranking_metrics(prepared, bound, ["评分", "类别"], amount)
    eligible = data.iloc[:4]
    weights = eligible["金额"].to_numpy() if amount else np.ones(4)
    bad = eligible.目标.eq(1).to_numpy()
    pairs = eligible.评分.to_numpy()[bad, None] - eligible.评分.to_numpy()[~bad]
    pair_weights = weights[bad, None] * weights[~bad]
    raw_auc = ((pairs > 0) + 0.5 * (pairs == 0)) * pair_weights
    raw_auc = raw_auc.sum() / pair_weights.sum()
    grouped = pd.DataFrame({"评分": eligible.评分, "坏": weights * bad, "好": weights * ~bad}).groupby("评分").sum()
    expected_ks = (grouped.坏.cumsum() / grouped.坏.sum() - grouped.好.cumsum() / grouped.好.sum()).abs().max()
    assert result["KS", "目标"]["评分"] == pytest.approx(expected_ks)
    assert result["AUC", "目标"]["评分"] == pytest.approx(max(raw_auc, 1 - raw_auc))
    assert np.isnan(result["KS", "目标"]["类别"])
    assert np.isnan(result["AUC", "目标"]["类别"])
    single, target = bind_targets(data.assign(目标=0), target="目标")
    one_class = _feature_ranking_metrics(single, target, ["评分"], amount)
    assert one_class["KS", "目标"]["评分"] == 0
    assert np.isnan(one_class["AUC", "目标"]["评分"])


def test_feature_summary_is_wide_and_ks_is_not_binning_maximum(tmp_path):
    data = pd.DataFrame({"评分": [0, 1, 2, 3] * 4, "FPD": [0, 1, 0, 1] * 4})
    tree = MultiwayTreeExtractor(target="FPD", features=["评分"], max_depth=1, n_jobs=1).fit(data)
    tree.manual_split(feature="评分", thresholds=[1.5])
    path = tmp_path / "特征宽表.xlsx"
    tree.report(
        data,
        target={"标签甲": data.FPD, "标签乙": data.FPD},
        methods=["uniform"],
        bin_params={"max_n_bins": 2},
        save=path,
    )
    book = load_workbook(path)
    sheet = book["组合策略汇总_特征分箱"]
    assert sheet.auto_filter.ref.startswith("B5:")
    assert len(sheet.tables) == 1  # 一份明细筛选表；宽表汇总使用工作表筛选。
    assert "目标标签" not in [c.value for c in sheet[5]]
    assert sheet["F4"].value == "KS" and sheet["H4"].value == "AUC"
    assert sheet["F6"].value == pytest.approx(0.5)
    assert sheet["H6"].value == pytest.approx(0.75)
    assert sheet["F6"].number_format == "0.00%"
    assert sheet["H6"].number_format == "0.0000"
    # 两个等宽箱的坏率相同，分档 KS 为 0；汇总 KS 必须来自原始特征。
    table = next(iter(sheet.tables.values()))
    from openpyxl.utils.cell import range_boundaries

    left, top, right, bottom = range_boundaries(table.ref)
    rows = list(sheet.iter_rows(min_row=top, max_row=bottom, min_col=left, max_col=right, values_only=True))
    detail = pd.DataFrame(rows[1:], columns=rows[0])
    assert detail["分档KS值"].max() == 0
    book.close()
