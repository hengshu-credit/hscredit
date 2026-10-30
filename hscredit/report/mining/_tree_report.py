"""复用模型报告样本统计与 scorecardpipeline 策略报告布局的树挖掘导出。"""

from copy import copy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ...utils.targets import bind_targets
from ...utils.dates import date_groups
from .._sample_stats import build_sample_stats_table, build_group_distribution_table


def _strategy_table(table):
    """仅转换 Excel 展示字段；指标直接保留 Rule.report 的数值。"""
    multi = isinstance(table.columns, pd.MultiIndex)

    def key(name):
        return ("分箱详情", name) if multi else name

    root = table.loc[table[key("节点编号")].eq(0)]
    result = table.drop(columns=[key(c) for c in ["规则分类", "分箱"] if key(c) in table]).copy()
    rename = {"指标名称": "组合策略", "样本总数": "命中数", "样本占比": "命中率"}
    result = result.rename(columns=rename, level=1) if multi else result.rename(columns=rename)
    for column in table.columns:
        if (column[-1] if multi else column) == "坏样本率":
            output = (column[0], "样本整体坏率") if multi else "样本整体坏率"
            result[output] = root[column].iloc[0] if len(root) else np.nan
    first = [key(c) for c in ["组合策略", "命中数", "命中率"] if key(c) in result]
    last = [key(c) for c in ["节点编号", "是否叶子", "指标含义", "入参字段", "字段含义"] if key(c) in result]
    middle = [c for c in result if c not in first + last]
    if multi:
        groups = dict.fromkeys(column[0] for column in middle)
        middle = [column for group in groups for column in middle if column[0] == group]
    return result[first + middle + last]


def tree_report_excel(
    tree,
    datasets,
    *,
    target=None,
    overdue=None,
    dpds=None,
    del_grey=False,
    leaf_only=False,
    feature_map=None,
    overdue_operator=">",
    save=None,
    sheet="组合策略汇总",
    excel_writer=None,
    date=None,
    freq="M",
    project_desc=None,
    data_source=None,
    **kwargs,
):
    """标签只求值一次；先写数据集信息，再写组合策略汇总与策略详情。

    :param tree: 已拟合的树规则挖掘器。
    :param datasets: DataFrame、数据集列表/元组或名称到 DataFrame 的字典。
    :param target: 默认 None；支持标签列名、Series、ndarray、list、callable 或标签映射；
        与 overdue + dpds 互斥，未提供标签时使用树的训练目标。
    :param overdue: 默认 None；逾期字段或字段列表，与 dpds 同时提供。
    :param dpds: 默认 None；逾期阈值或阈值列表。
    :param del_grey: 默认 False；是否逐标签剔除灰样本。
    :param leaf_only: 默认 False 返回全部节点效果；True 仅返回叶节点效果。
    :param feature_map: 默认 None；特征名称到中文含义的映射。
    :param overdue_operator: 默认 '>'；逾期比较符，支持 '>'、'>='、'<'、'<='。
    :param save: 默认 None；导出 Excel 路径。
    :param sheet: 默认 '组合策略汇总'；策略汇总工作表名称。
    :param excel_writer: 默认 None；可复用的 hscredit ExcelWriter。
    :param date: 默认 None；用于样本时间分布的日期字段。
    :param freq: 默认 'M'；日期统计频率。
    :param project_desc: 默认 None；项目目标和取样逻辑说明。
    :param data_source: 默认 None；数据来源说明。
    :param kwargs: 传给树 report 的评估参数，如 amount、round_digits。
    :return: 与输入数据集结构对应的中文效果表、列表或字典。
    """
    from ...excel import ExcelWriter, dataframe2excel
    from ...core.viz import plot_tree
    import matplotlib.pyplot as plt
    from PIL import Image

    if excel_writer is not None and not isinstance(excel_writer, ExcelWriter):
        raise TypeError("excel_writer 必须是 hscredit.excel.ExcelWriter 实例")
    if not isinstance(sheet, str) or not sheet or len(sheet) > 24 or any(c in sheet for c in "[]:*?/\\"):
        raise ValueError("sheet 必须是 1 至 24 字符的有效 Excel 工作表名称")
    frames = (
        datasets
        if isinstance(datasets, dict)
        else {
            f"数据集{i + 1}": frame
            for i, frame in enumerate(datasets if isinstance(datasets, (list, tuple)) else [datasets])
        }
    )
    if not frames:
        raise ValueError("导出挖掘报告至少需要一个数据集")
    full_reports, y_maps, date_values = {}, [], []
    for name, frame in frames.items():
        actual_target = target
        if target is None and overdue is None:
            classes = getattr(tree, "classes_", [0, 1])
            actual_target = (
                tree.target
                if set(classes).issubset({0, 1})
                else {f"类别_{value}": frame[tree.target].eq(value).astype(int) for value in classes}
            )
        prepared, bound = bind_targets(frame, actual_target, overdue, dpds, del_grey, overdue_operator)
        y_maps.append(bound.y_map(prepared, style="at"))
        if date is not None:
            if date not in frame:
                raise ValueError(f"数据集 {name} 缺少日期字段 {date}")
            date_values.append(date_groups(frame[date], freq))
        full_reports[name] = tree.report(
            prepared,
            target=bound,
            leaf_only=False,
            feature_map=feature_map,
            overdue_operator=overdue_operator,
            **kwargs,
        )
    results = {}
    for name, table in full_reports.items():
        leaf_column = ("分箱详情", "是否叶子") if isinstance(table.columns, pd.MultiIndex) else "是否叶子"
        results[name] = table.loc[table[leaf_column].eq("是")].reset_index(drop=True) if leaf_only else table
    writer = excel_writer if excel_writer is not None else ExcelWriter()
    basic = "1-基本信息" if sheet == "组合策略汇总" else f"{sheet}_基本信息"
    detail = "策略详情" if sheet == "组合策略汇总" else f"{sheet}_策略详情"
    names = [basic, sheet, detail]
    for name in names:
        if name in writer.workbook.sheetnames:
            del writer.workbook[name]
        writer.get_sheet_by_name(name)
    percent = [
        "好样本占比",
        "坏样本占比",
        "命中率",
        "坏样本率",
        "样本整体坏率",
        "坏账改善",
        "准确率",
        "精确率",
        "召回率",
        "F1分数",
    ]
    try:
        ws = writer.get_sheet_by_name(basic)
        row, _ = writer.insert_value2sheet(ws, (2, 2), value="一、基本信息", style="header_middle")
        description = project_desc or "使用树规则划分客群，并比较各数据集的规则命中效果"
        row, _ = writer.insert_value2sheet(ws, (row + 2, 2), value="1、项目目标", style="header_middle")
        row, _ = writer.insert_value2sheet(
            ws, (row, 2), value=description, style="middle", align={"horizontal": "left"}
        )
        info = [
            {"统计项": "模型名称", "统计内容": type(tree).__name__},
            {"统计项": "取样逻辑", "统计内容": project_desc or ""},
            {"统计项": "数据源", "统计内容": data_source or ""},
            {"统计项": "指标口径", "统计内容": "、".join(y_maps[0])},
            {"统计项": "规则范围", "统计内容": "仅叶节点" if leaf_only else "全部节点（父子规则存在重叠）"},
        ]
        if date is not None:
            dates = pd.concat([pd.to_datetime(frame[date]) for frame in frames.values()]).dropna()
            info.insert(
                0,
                {
                    "统计项": "样本区间",
                    "统计内容": "" if dates.empty else f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}",
                },
            )
        if kwargs.get("amount") is not None:
            info.append(
                {
                    "统计项": "金额口径",
                    "统计内容": f"策略表按 {kwargs['amount']} 加权，命中数为金额；样本统计与分类指标按件数",
                }
            )
        row, _ = dataframe2excel(
            pd.DataFrame(info),
            writer,
            sheet_name=basic,
            title="2、数据样本描述",
            start_row=row + 2,
            left_cols=["统计项", "统计内容"],
        )
        labels = list(y_maps[0])
        filtered = any(pd.isna(values).any() for y_map in y_maps for values in y_map.values())
        stats, pct = build_sample_stats_table(
            list(frames), y_maps, labels, flat_total_col="样本数", target_specific_totals=filtered, n_jobs=1
        )
        row, _ = dataframe2excel(
            stats,
            writer,
            sheet_name=basic,
            title="3、数据样本统计",
            start_row=row + 2,
            index=isinstance(stats.columns, pd.MultiIndex),
            percent_cols=pct,
            decimal=None,
        )
        if date is not None:
            distribution, pct = build_group_distribution_table(
                list(frames), y_maps, date_values, labels, target_specific_totals=filtered, n_jobs=1
            )
            if not isinstance(distribution.columns, pd.MultiIndex):
                distribution = distribution.reset_index()
            dataframe2excel(
                distribution,
                writer,
                sheet_name=basic,
                title="4、样本分布情况",
                start_row=row + 2,
                percent_cols=pct,
                decimal=None,
                merge=True,
                merge_column=["数据集"],
            )
        row = 2
        for name, full in full_reports.items():
            table = _strategy_table(full)
            leaf_col = ("分箱详情", "是否叶子") if isinstance(table.columns, pd.MultiIndex) else "是否叶子"
            if leaf_only:
                table = table.loc[table[leaf_col].eq("是")]
            row, _ = dataframe2excel(
                table,
                writer,
                sheet_name=sheet,
                title=f"组合策略: {name}",
                start_row=row,
                percent_cols=percent,
                condition_cols=["坏样本率", "LIFT值"],
                left_cols=["组合策略", "指标含义"],
                decimal=None,
            )
            row += 2
        # scorecardpipeline 的策略详情按“策略表 -> 树图”排列。
        training = (
            tree.get_rule_table(round_digits=False) if tree._data is not None else next(iter(full_reports.values()))
        )
        row, _ = dataframe2excel(
            _strategy_table(training),
            writer,
            sheet_name=detail,
            title="组合策略: 训练集" if tree._data is not None else "组合策略: 当前数据集",
            percent_cols=percent,
            condition_cols=["坏样本率", "LIFT值"],
            decimal=None,
        )
        with TemporaryDirectory(prefix="hscredit-tree-report-") as temporary:
            image_path = str(Path(temporary) / "tree.png")
            figure = plot_tree(tree, save=image_path)
            plt.close(figure)
            with Image.open(image_path) as image:
                size = (1400, round(1400 * image.height / image.width))
            row, _ = writer.insert_pic2sheet(writer.get_sheet_by_name(detail), image_path, (row + 2, 2), figsize=size)
            if hasattr(tree, "get_node_table"):
                dataframe2excel(
                    tree.get_node_table(),
                    writer,
                    sheet_name=detail,
                    title="完整节点信息",
                    start_row=row + 2,
                    percent_cols=["样本占比", "坏样本率"],
                    decimal=None,
                )
            for image in writer.get_sheet_by_name(detail)._images:
                image.ref = BytesIO(Path(image_path).read_bytes())
        _fit_report_layout(writer, names)
        if save is not None:
            images = [(image, image._data()) for ws in writer.workbook for image in ws._images]
            for image, payload in images:
                image.ref = BytesIO(payload)
            try:
                writer.save(str(save), close=excel_writer is None)
            finally:
                for image, payload in images:
                    image.ref = BytesIO(payload)
    finally:
        if excel_writer is None:
            writer.workbook.close()
    if isinstance(datasets, dict):
        return results
    return list(results.values()) if isinstance(datasets, (list, tuple)) else next(iter(results.values()))


def _fit_report_layout(writer, names):
    """复用模型报告标题布局和 ExcelWriter 自动列宽，沿用其默认最大宽度 50。"""
    from .._excel_layout import adjust_report_title_merges

    for name in names:
        ws = writer.get_sheet_by_name(name)
        ws.sheet_view.showGridLines = False
        writer.adjust_columns_width(ws)
        adjust_report_title_merges(ws)
        for row in ws:
            for cell in row:
                if cell.value is not None:
                    alignment = copy(cell.alignment)
                    alignment.wrap_text = False
                    cell.alignment = alignment
