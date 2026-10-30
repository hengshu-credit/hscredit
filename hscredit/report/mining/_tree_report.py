"""复用模型报告样本统计与 scorecardpipeline 策略报告布局的树挖掘导出。"""

from copy import copy, deepcopy
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
    first = [key(c) for c in ["节点编号", "是否叶子", "是否调整", "组合策略", "命中数", "命中率"] if key(c) in result]
    last = [key(c) for c in ["指标含义", "入参字段", "字段含义"] if key(c) in result]
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
    """标签只求值一次，在单页生成基本信息、挖掘摘要与订单/金额口径的树图及规则表。

    :param tree: 已拟合的树规则挖掘器。
    :param datasets: DataFrame、数据集列表/元组或名称到 DataFrame 的字典。
    :param target: 默认 None；支持标签列名、Series、ndarray、list、callable 或标签映射；
        Series 按索引对齐，ndarray/list 按行位置对应，callable 接收完整数据集；
        与 overdue + dpds 互斥，未提供标签时使用树的训练目标。
    :param overdue: 默认 None；逾期字段或字段列表，与 dpds 同时提供。
    :param dpds: 默认 None；逾期阈值或阈值列表。
    :param del_grey: 默认 False；是否逐标签剔除灰样本。
    :param leaf_only: 默认 False 返回全部节点效果；True 仅返回叶节点效果。
    :param feature_map: 默认 None；特征名称到中文含义的映射。
    :param overdue_operator: 默认 '>'；逾期比较符，支持 '>'、'>='、'<'、'<='。
    :param save: 默认 None；导出 Excel 路径。
    :param sheet: 默认 '组合策略汇总'；完整挖掘报告的单页名称。
    :param excel_writer: 默认 None；可复用的 hscredit ExcelWriter。
    :param date: 默认 None；用于样本时间分布的日期字段。
    :param freq: 默认 'M'；日期统计频率。
    :param project_desc: 默认 None；项目目标和取样逻辑说明。
    :param data_source: 默认 None；数据来源说明。
    :param kwargs: 传给树 report 的评估参数，如 amount、round_digits。
    :return: 与输入数据集结构对应的中文效果表、列表或字典。
    """
    from ...excel import ExcelWriter, dataframe2excel

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
    full_reports, order_reports, y_maps, date_values, plot_labels = {}, {}, [], [], {}
    amount = kwargs.get("amount")
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
        plot_labels[name] = bound.names(style="space")[0]
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
        order_reports[name] = (
            full_reports[name]
            if amount is None
            else tree.report(
                prepared,
                target=bound,
                leaf_only=False,
                feature_map=feature_map,
                overdue_operator=overdue_operator,
                **{k: v for k, v in kwargs.items() if k != "amount"},
            )
        )
    results = {}
    for name, table in full_reports.items():
        leaf_column = ("分箱详情", "是否叶子") if isinstance(table.columns, pd.MultiIndex) else "是否叶子"
        results[name] = table.loc[table[leaf_column].eq("是")].reset_index(drop=True) if leaf_only else table
    writer = excel_writer if excel_writer is not None else ExcelWriter()
    basic = sheet
    names = [sheet]
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
        if amount:
            for source, output in [
                ("样本总数", "样本金额"),
                ("好样本数", "好样本金额"),
                ("坏样本数", "坏样本金额"),
                ("坏样本率", "金额坏率"),
            ]:
                for label_index, label in enumerate(labels):
                    values = []
                    for table in full_reports.values():
                        if isinstance(table.columns, pd.MultiIndex):
                            groups = [col[0] for col in table if col[-1] == "坏样本率"]
                            column = (
                                ("分箱详情", source) if ("分箱详情", source) in table else (groups[label_index], source)
                            )
                        else:
                            column = source
                        values.append(table[column].iloc[0])
                    column = (output, label) if isinstance(stats.columns, pd.MultiIndex) else output
                    stats[column] = values
                    if output == "金额坏率":
                        pct.append(column)
        performance = _prediction_metrics(tree, frames, y_maps, amount)
        for basis in ["订单", "金额"]:
            for metric in ["AUC", "KS", "PSI"]:
                for label in labels:
                    key = (basis, label, metric)
                    if key not in performance:
                        continue
                    metric_name = metric if basis == "订单" else f"金额{metric}"
                    column = (metric_name, label) if isinstance(stats.columns, pd.MultiIndex) else metric_name
                    stats[column] = performance[key]
                    if metric == "KS":
                        pct.append(column)
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
            row, _ = dataframe2excel(
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
        writer.adjust_columns_width(ws, start_col=4, end_col=ws.max_column)
        training_data = tree._data if tree._data is not None else next(iter(frames.values()))
        if tree._data is None:
            training = next(iter(order_reports.values()))
            training_amount = next(iter(full_reports.values())) if amount else None
        else:
            training = tree.get_rule_table(training_data, round_digits=False)
            if amount and amount not in training_data:
                raise ValueError(f"模型未保留训练集金额字段 {amount}，请使用包含该字段的训练数据重新 fit")
            training_amount = tree.get_rule_table(training_data, amount=amount, round_digits=False) if amount else None
        selected = _select_rules(training, leaf_only)
        node_col = ("分箱详情", "节点编号") if isinstance(selected.columns, pd.MultiIndex) else "节点编号"
        node_order = selected[node_col].tolist()
        row = _write_mining_summary(tree, writer, sheet, row + 2, training, training_amount)
        with TemporaryDirectory(prefix="hscredit-tree-report-") as temporary:
            train_label = "训练集" if tree._data is not None else "首数据集"
            row = _write_tree_section(
                tree,
                training,
                writer,
                sheet,
                row + 2,
                f"三、策略详情：{train_label}（订单口径）",
                temporary,
                percent,
                node_order,
                False,
            )
            if amount:
                row = _write_tree_section(
                    tree,
                    training_amount,
                    writer,
                    sheet,
                    row + 2,
                    f"策略详情：{train_label}（金额口径，{amount}）",
                    temporary,
                    percent,
                    node_order,
                    True,
                )
            for name, orders in order_reports.items():
                row = _write_tree_section(
                    tree,
                    orders,
                    writer,
                    sheet,
                    row + 2,
                    f"四、最终树应用：{name}（订单口径）",
                    temporary,
                    percent,
                    node_order,
                    False,
                    plot_labels[name],
                )
                if amount:
                    row = _write_tree_section(
                        tree,
                        full_reports[name],
                        writer,
                        sheet,
                        row + 2,
                        f"最终树应用：{name}（金额口径，{amount}）",
                        temporary,
                        percent,
                        node_order,
                        True,
                        plot_labels[name],
                    )
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
        writer.adjust_columns_width(ws, start_col=4, end_col=ws.max_column)
        adjust_report_title_merges(ws)
        for row in ws:
            for cell in row:
                if cell.value is not None:
                    alignment = copy(cell.alignment)
                    alignment.wrap_text = False
                    cell.alignment = alignment


def _select_rules(table, leaf_only):
    column = ("分箱详情", "是否叶子") if isinstance(table.columns, pd.MultiIndex) else "是否叶子"
    return table.loc[table[column].eq("是")] if leaf_only else table


def _tree_score(tree, data, class_index=1):
    """使用最终树预测；手工二叉树复用叶规则路由与已存储的训练概率。"""
    if hasattr(tree, "predict_proba"):
        probability = tree.predict_proba(data)
        if probability.shape[1] == 1:
            classes = getattr(tree, "classes_", getattr(getattr(tree, "clf", None), "classes_", [0]))
            return np.full(len(data), float(classes[0] == 1))
        return probability[:, min(class_index, probability.shape[1] - 1)]
    from ...core.viz.tree_plots import _extract_tree_data

    nodes = {node["node_id"]: node for node in _extract_tree_data(tree)["nodes"]}
    values = np.full(len(data), nodes[0]["bad_rate"], dtype=float)
    for rule in tree.get_rules():
        node = int(rule.name.rsplit("_", 1)[1])
        values[np.asarray(rule.predict(data), dtype=bool)] = nodes[node]["bad_rate"]
    return values


def _prediction_metrics(tree, frames, y_maps, amount):
    """样本统计的 AUC/KS/PSI 直接复用 ModelReport，金额口径使用其 sample_weight。"""
    from ..model_report import ModelReport

    features = getattr(tree, "features_", None) or getattr(tree, "_feature_list", None)
    result = {}
    for label_index, label in enumerate(y_maps[0]):
        class_index = label_index if str(label).startswith("类别_") else 1
        for basis, weight in [("订单", None)] + ([("金额", amount)] if amount else []):
            eligible = {
                name: (frame, y_map[label])
                for (name, frame), y_map in zip(frames.items(), y_maps)
                if len(frame)
                and pd.notna(y_map[label]).any()
                and (weight is None or frame[weight].to_numpy()[np.asarray(pd.notna(y_map[label]))].sum() > 0)
            }
            metrics = pd.DataFrame()
            if eligible:
                report = ModelReport(
                    tree,
                    datasets=eligible,
                    features=features,
                    prediction_method=lambda X: _tree_score(tree, X, class_index),
                    score_direction="higher_risk",
                    sample_weight=weight,
                    del_grey=True,
                    n_jobs=1,
                )
                metrics = report.get_metrics().set_index("统计项")
            for metric in ["AUC", "KS", "PSI"]:
                if basis == "金额" and metric == "PSI":
                    continue
                if metric in metrics.index or metric in {"AUC", "KS"}:
                    values = [
                        metrics.loc[metric, name] if metric in metrics.index and name in metrics else np.nan
                        for name in frames
                    ]
                    result[(basis, label, metric)] = pd.to_numeric(pd.Series(values), errors="coerce").tolist()
    return result


def _tree_view(tree, table, amount=False):
    """固定节点、边和阈值，仅叠加当前数据集命中量与第一标签的效果。"""
    from ...core.viz.tree_plots import _extract_tree_data

    view = deepcopy(_extract_tree_data(tree))
    multi = isinstance(table.columns, pd.MultiIndex)
    meta = lambda name: ("分箱详情", name) if multi else name
    bad_col = next(c for c in table if (c[-1] if multi else c) == "坏样本率")
    label = bad_col[0] if multi else str(tree.target)
    metric = lambda name: (label, name) if multi else name
    # 剔灰模式的总数在各标签下独立保存。
    count_col = meta("样本总数") if meta("样本总数") in table else metric("样本总数")
    share_col = meta("样本占比") if meta("样本占比") in table else metric("样本占比")
    rows = table.set_index(meta("节点编号"))
    view["total_samples"] = rows.loc[0, count_col]
    view["overall_bad_rate"] = rows.loc[0, bad_col]
    for node in view["nodes"]:
        row = rows.loc[node["node_id"]]
        n, bad = row[count_col], row[metric("坏样本数")]
        node.update(
            n_samples=n,
            sample_pct=row[share_col],
            bad_count=bad,
            good_count=n - bad,
            bad_rate=row[bad_col],
            lift=row[metric("LIFT值")],
        )
        unit = "金额" if amount else "订单数"
        node["metric_rows"] = [
            (f"命中{unit}", f"{n:,.2f}" if amount else f"{n:,.0f}"),
            (f"{unit}占比", f"{row[share_col]:.2%}"),
            ("坏样本金额" if amount else "坏订单数", f"{bad:,.2f}" if amount else f"{bad:,.0f}"),
            ("金额坏率" if amount else "坏订单率", f"{row[bad_col]:.2%}"),
            ("LIFT值", f"{row[metric('LIFT值')]:.4f}"),
        ]
    return view, label


def _write_tree_section(tree, full, writer, sheet, row, title, temporary, percent, node_order, amount, plot_label=None):
    from ...excel import dataframe2excel
    from ...core.viz import plot_tree
    import matplotlib.pyplot as plt
    from PIL import Image

    view, label = _tree_view(tree, full, amount)
    path = Path(temporary) / f"tree-{len(writer.get_sheet_by_name(sheet)._images)}.png"
    figure = plot_tree(view, save=str(path))
    plt.close(figure)
    with Image.open(path) as image:
        size = (1400, round(1400 * image.height / image.width))
    node_col = ("分箱详情", "节点编号") if isinstance(full.columns, pd.MultiIndex) else "节点编号"
    table = _strategy_table(full).set_index(node_col, drop=False).loc[node_order].reset_index(drop=True)
    end, _ = dataframe2excel(
        table,
        writer,
        sheet_name=sheet,
        title=f"{title}；树图标签：{plot_label or label}",
        start_row=row,
        figures=[str(path)],
        figsize=size,
        percent_cols=percent,
        condition_cols=["坏样本率", "LIFT值"],
        decimal=None,
    )
    writer.get_sheet_by_name(sheet)._images[-1].ref = BytesIO(path.read_bytes())
    return end


def _write_mining_summary(tree, writer, sheet, row, training, training_amount):
    from ...excel import dataframe2excel
    from ...core.viz.tree_plots import _extract_tree_data

    nodes = _extract_tree_data(tree)["nodes"]
    leaves = sum(node["is_leaf"] for node in nodes)
    summary = pd.DataFrame(
        [
            {
                "规则数（含根节点）": len(nodes),
                "叶子节点数": leaves,
                "中间节点数": len(nodes) - leaves,
                "树深度（根为0）": max(node["depth"] for node in nodes),
                "人工调整节点数": int(
                    training[("分箱详情", "是否调整") if isinstance(training.columns, pd.MultiIndex) else "是否调整"]
                    .eq("是")
                    .sum()
                ),
            }
        ]
    )
    row, _ = dataframe2excel(summary, writer, sheet_name=sheet, title="二、规则挖掘摘要", start_row=row)
    model = tree if hasattr(tree, "get_params") else getattr(tree, "clf", getattr(tree, "_sklearn_clf", None))
    parameters = model.get_params(deep=False) if model is not None else {}
    params = pd.DataFrame([{"参数": name, "配置": str(value)} for name, value in parameters.items()])
    row, _ = dataframe2excel(params, writer, sheet_name=sheet, title="树的参数配置", start_row=row + 2)
    history = getattr(tree, "edit_history_", [])
    if history:
        row, _ = dataframe2excel(
            pd.DataFrame(history),
            writer,
            sheet_name=sheet,
            title="手工调整记录（节点编号为操作当时编号）",
            start_row=row + 2,
        )
    else:
        adjusted = [node for node in nodes if node["is_manual"]]
        records = pd.DataFrame(
            [{"节点编号": n["node_id"], "当前分裂": n["condition"], "阈值": n["threshold_text"]} for n in adjusted],
            columns=["节点编号", "当前分裂", "阈值"],
        )
        row, _ = dataframe2excel(records, writer, sheet_name=sheet, title="当前人工调整节点", start_row=row + 2)
    rows = []
    cuts = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5, np.inf]
    for basis, table in [("订单", training)] + ([("金额", training_amount)] if training_amount is not None else []):
        multi = isinstance(table.columns, pd.MultiIndex)
        leaf_col = ("分箱详情", "是否叶子") if multi else "是否叶子"
        for lift in [c for c in table if (c[-1] if multi else c) == "LIFT值"]:
            for low, high in zip(cuts, cuts[1:]):
                selected = table.loc[table[lift].ge(low) & table[lift].lt(high)]
                rows.append(
                    {
                        "口径": basis,
                        "标签": lift[0] if multi else tree.target,
                        "LIFT区间": f"[{low:g}, {high:g})",
                        "规则数": len(selected),
                        "叶子节点数": int(selected[leaf_col].eq("是").sum()),
                        "中间节点数": int(selected[leaf_col].eq("否").sum()),
                    }
                )
    row, _ = dataframe2excel(
        pd.DataFrame(rows), writer, sheet_name=sheet, title="训练集 LIFT 区间规则分布", start_row=row + 2
    )
    return row
