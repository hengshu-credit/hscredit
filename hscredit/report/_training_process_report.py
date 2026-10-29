"""面向阅读的训练过程页：摘要、筛选、试验对比和学习曲线。"""

from pathlib import Path
import numpy as np
import pandas as pd


def process_sheet_name(report, index=None):
    full = report.explain_config["enabled"] and report.explain_config["scope"] == "full"
    if index is None:
        index = 8 if full else 7
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 7
        or (full and index == 7)
    ):
        raise ValueError("training_sheet_index 必须是未占用的正整数：普通报告从 7 起，含模型解释时从 8 起")
    name = f"{index}-训练与调参过程"
    if len(name) > 31:
        raise ValueError("training_sheet_index 过长，工作表名称不能超过 31 个字符")
    return name


def process_tables(report):
    """原始过程仍由 get_training_process 提供，Excel 不重复展开对象和默认配置。"""
    raw = report.get_training_process()
    context = report.model_context_
    overview = {
        "模型": context.get("最终估计器", context.get("模型类型")),
        "任务": context.get("任务类型"),
        "目标": report._target_name,
        "预测方法": context.get("预测方法"),
        "输入特征数": len(report.feature_names),
    }
    history = raw.get("训练记录", pd.DataFrame())
    if not history.empty:
        overview["最终训练样本数"] = history.iloc[-1].get("样本数")
    settings = raw.get("搜索设置", pd.DataFrame())
    if not settings.empty:
        overview.update(dict(zip(settings["配置项"], settings["配置值"])))
    relation = raw["模型口径"].query("配置项 == '与调参关系'")
    if not relation.empty:
        overview["与搜索的关系"] = relation.iloc[0]["配置值"]
    result = {"训练概览": pd.DataFrame(overview.items(), columns=["项目", "内容"])}
    for name in ("预处理步骤", "特征筛选记录", "特征筛选阶段"):
        if name in raw and not raw[name].empty:
            table = raw[name].copy()
            if name == "特征筛选阶段":
                table = table[
                    [
                        column
                        for column in ("步骤", "阶段名称", "输入特征数", "选中特征数", "剔除特征数", "阈值")
                        if column in table
                    ]
                ]
            for column in table:
                table[column] = table[column].map(
                    lambda v: "、".join(map(str, v)) if isinstance(v, (list, tuple)) else v
                )
            result[name] = table
    trials = raw.get("调参记录", pd.DataFrame())
    if not trials.empty:
        metric_names = raw.get("搜索指标", pd.DataFrame()).get("指标", pd.Series(dtype=str)).tolist()
        rows = []
        for item in trials.to_dict("records"):
            row = {key: item[key] for key in ("试验编号", "状态", "搜索选中", "耗时秒") if key in item}
            for index, value in enumerate(item.get("各目标得分") or []):
                row[metric_names[index] if index < len(metric_names) else f"目标{index+1}"] = value
            actual = item.get("实际搜索参数") or {}
            row.update({f"参数：{key}": actual.get(key, value) for key, value in (item.get("采样参数") or {}).items()})
            if item.get("错误信息"):
                row["错误信息"] = item["错误信息"]
            rows.append(row)
        result["参数搜索对比"] = pd.DataFrame(rows)
    if "各折指标" in raw:
        result["验证结果"] = raw["各折指标"]
    for name, table in raw.items():
        if name.endswith("训练记录") and not table.empty:
            columns = [
                c
                for c in ("训练修订", "训练阶段", "开始时间", "样本数", "耗时秒", "最佳迭代", "最佳得分", "状态")
                if c in table
            ]
            result["最终模型训练" if name == "训练记录" else name] = table[columns].copy()
    if "最终评估" in raw and not raw["最终评估"].empty:
        result["最终模型评估"] = raw["最终评估"]
    return result


def write_process(report, writer, filepath, with_plots, sheet_name=None):
    from ..excel import dataframe2excel
    from ..core.viz.style import get_palette
    import matplotlib.pyplot as plt

    sheet = writer.get_sheet_by_name(sheet_name or process_sheet_name(report))
    sheet.sheet_view.showGridLines = False
    index = sheet.title.split("-")[0]
    writer.insert_value2sheet(sheet, "B2", f"{index}、训练与调参过程", style="header")
    row, section_index = 5, 0

    def add_table(title, table):
        nonlocal row, section_index
        if table.empty:
            return
        section_index += 1
        writer.insert_value2sheet(
            sheet,
            (row, 2),
            f"{index}.{section_index}、{title}",
            style="header",
            end_space=(row, max(4, len(table.columns) + 1)),
        )
        shown = table.apply(lambda c: c.map(lambda v: str(v) if isinstance(v, (dict, list, tuple)) else v))
        end, _ = dataframe2excel(
            shown, writer, sheet_name=sheet, start_row=row + 2, index=False, speed="fast", decimal=None
        )
        for column, name in enumerate(table.columns, 2):
            for cells in sheet.iter_rows(min_row=row + 3, max_row=end - 1, min_col=column, max_col=column):
                cell = cells[0]
                if isinstance(cell.value, (float, np.floating)):
                    cell.number_format = "General" if str(name).startswith("参数：") else "0.0000"
        row = end + 3

    for title, table in process_tables(report).items():
        add_table(title, table)
    raw = report.get_training_process()
    for title in ("选中试验曲线", "训练曲线"):
        curve = raw.get(title, pd.DataFrame())
        if curve.empty:
            continue
        keys = [k for k in ("试验编号", "折编号", "训练修订", "数据集", "指标") if k in curve]
        summary = (
            curve.groupby(keys, sort=False, dropna=False)
            .agg(记录轮数=("迭代", "size"), 首轮数值=("数值", "first"), 末轮数值=("数值", "last"))
            .reset_index()
        )
        add_table("最终模型学习曲线" if title == "训练曲线" else "选中试验学习曲线", summary)
        if not with_plots:
            continue
        folder = Path(filepath).parent / (Path(filepath).stem + "_assets")
        folder.mkdir(parents=True, exist_ok=True)
        for metric_index, (metric, values) in enumerate(curve.groupby("指标", sort=False)):
            figure, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
            try:
                groups = [k for k in keys if k != "指标"]
                for number, (group, points) in enumerate(values.groupby(groups, sort=False, dropna=False)):
                    labels = group if isinstance(group, tuple) else (group,)
                    label = " / ".join(
                        str(value) if key == "数据集" else f"{key}{value}" for key, value in zip(groups, labels)
                    )
                    axis.plot(
                        points["迭代"], points["数值"], label=label, color=get_palette()[number % len(get_palette())]
                    )
                axis.set(xlabel="迭代", ylabel=str(metric), title=f"{title}：{metric}")
                axis.legend()
                path = folder / f"{title}_{metric_index}.png"
                figure.savefig(path, dpi=140)
            finally:
                plt.close(figure)
            row, _ = writer.insert_pic2sheet(sheet, str(path), (row, 2), figsize=(900, 360))
            row += 3
    writer.insert_hyperlink2sheet(sheet, "B2", hyperlink="#'目录'!B2")
