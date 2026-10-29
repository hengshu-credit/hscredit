"""回归模型报告的连续目标、误差指标、分箱与原生 Excel 布局。"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import is_regressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..core.metrics import compute_bin_stats, psi
from ..core.models._contracts import validate_sample_weight

METRICS = {"mse": "均方误差", "rmse": "均方根误差", "mae": "平均绝对误差", "r2": "决定系数"}


def is_regression(model):
    """按估计器协议识别任务，不根据标签取值或 predict 方法猜测。"""
    while getattr(model, "steps", None):
        model = model.steps[-1][1]
    kind = getattr(model, "_estimator_type", None)
    if kind is not None:
        return kind == "regressor"
    return is_regressor(model) if callable(getattr(model, "__sklearn_tags__", None)) else False


def target_values(frame, value, name):
    """保留连续数值；Series 按索引对齐，数组按位置，缺失标签不进入监督指标。"""
    if isinstance(value, str):
        if value not in frame:
            raise ValueError(f"回归报告缺少目标字段: {value}")
        value = frame[value]
    elif callable(value):
        value = value(frame.copy(deep=True))
    if value is None or isinstance(value, dict):
        raise ValueError("回归报告需要单个连续目标，请提供 y 或 target 字段/一维数据/callable")
    if isinstance(value, pd.Series) and not value.index.equals(frame.index):
        if not frame.index.is_unique or not value.index.is_unique:
            raise ValueError("回归目标含重复索引时必须与数据完全同序")
        if len(value) != len(frame) or not frame.index.isin(value.index).all():
            raise ValueError("回归目标索引必须与数据完全匹配")
        value = value.reindex(frame.index)
    try:
        values = (
            value.to_numpy(dtype=float, na_value=np.nan)
            if isinstance(value, pd.Series)
            else np.asarray(value, dtype=float)
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("回归报告目标必须为数值") from exc
    if values.ndim != 1 or len(values) != len(frame) or not len(values):
        raise ValueError("回归报告目标必须是一维、非空且与数据等长")
    if np.isinf(values).any():
        raise ValueError("回归报告目标不允许正负无穷")
    return pd.Series(values, index=frame.index, name=name)


def configure(report, metrics, sample_weight, score_direction):
    if score_direction is not None:
        raise ValueError("回归报告不使用风险排序方向，请省略 score_direction")
    requested = list(METRICS) if metrics is None else [metrics] if isinstance(metrics, str) else list(metrics)
    if any(not isinstance(value, str) or value not in METRICS for value in requested):
        raise ValueError("回归报告 metrics 支持 mse、rmse、mae、r2；不能使用二分类金融指标")
    if len(set(requested)) != len(requested):
        raise ValueError("报告评估指标名称不能重复")
    report.metrics = requested
    report.sample_weight = copy.deepcopy(sample_weight)
    report.score_direction = None
    report._SUMMARY_METRICS = ["样本数", "均方根误差", "平均绝对误差", "决定系数"]


def error_values(y, prediction, weights=None):
    """与 net.evaluate 保持同一加权口径，R² 单样本返回 NaN。"""
    if not len(y):
        return {name: np.nan for name in METRICS.values()}
    weights = validate_sample_weight(weights, len(y))
    mse = float(mean_squared_error(y, prediction, sample_weight=weights))
    return {
        "均方误差": mse,
        "均方根误差": float(np.sqrt(mse)),
        "平均绝对误差": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "决定系数": float(r2_score(y, prediction, sample_weight=weights)) if len(y) > 1 else np.nan,
    }


def metric_table(report):
    train_y, _ = report._get_valid_target_arrays(report._train_key)
    train_weights = report._metric_weight(report._train_key)
    reference = float(np.average(train_y, weights=train_weights)) if len(train_y) else np.nan
    columns = {}
    for key, dataset in report._datasets.items():
        y, prediction = report._get_valid_target_arrays(key)
        weights = report._metric_weight(key)
        values = {"样本数": len(y), **error_values(y, prediction, weights)}
        values.update(
            目标均值=float(np.average(y, weights=weights)) if len(y) else np.nan,
            预测均值=float(np.average(prediction, weights=weights)) if len(y) else np.nan,
            平均残差=float(np.average(prediction - y, weights=weights)) if len(y) else np.nan,
            基准均方根误差=(
                float(np.sqrt(mean_squared_error(y, np.full(len(y), reference), sample_weight=weights)))
                if len(y) and np.isfinite(reference)
                else np.nan
            ),
        )
        if len(report._datasets) > 1:
            values["PSI"] = (
                np.nan
                if key == report._train_key
                else float(psi(report._datasets[report._train_key].prediction, dataset.prediction))
            )
        columns[dataset.label] = values
    return pd.DataFrame(columns).rename_axis("统计项").reset_index()


def bin_table(report, dataset, method, max_n_bins, margins, feature=None, amount=None, label=None, labels=None):
    """训练集确定等频切点，使用 compute_bin_stats 的连续目标口径；分箱表为件数口径。"""
    from ..core.binning import QuantileBinning

    if method != "quantile":
        raise ValueError("回归报告分箱目前支持 method='quantile'")
    if label is not None or labels:
        raise ValueError("回归报告目前只支持单个连续目标")
    if isinstance(max_n_bins, bool) or not isinstance(max_n_bins, (int, np.integer)) or max_n_bins < 1:
        raise ValueError("max_n_bins 必须为正整数")
    key = report._resolve_dataset_key(dataset)
    train, current = report._datasets[report._train_key], report._datasets[key]
    name = "预测值" if feature is None else feature
    train_values = train.prediction if feature is None else train.X[feature]
    values = current.prediction if feature is None else current.X[feature]
    frame = pd.DataFrame({name: np.asarray(train_values)})
    if len(frame) == 1:
        frame = pd.concat([frame, frame], ignore_index=True)
    # 等频切点不使用标签。交替标签仅满足公共分箱器统计接口的二分类校验，
    # 关闭单调及比例约束，舍弃该辅助统计，再以真实连续目标独立计算全部报告统计。
    binner = QuantileBinning(min_n_bins=1, max_n_bins=int(max_n_bins), min_bin_size=None, monotonic=False, n_jobs=1)
    if not pd.api.types.is_numeric_dtype(frame[name]):
        binner.set_params(category_order={name: list(pd.unique(frame[name].dropna()))})
    binner.fit(frame, np.arange(len(frame)) % 2)
    target_frame = pd.DataFrame({name: np.asarray(values)})
    bins = binner.transform(target_frame, metric="indices")[name].to_numpy()
    captions = binner.transform(target_frame, metric="bins")[name].to_numpy()
    y, prediction = np.asarray(current.y, dtype=float), current.prediction
    valid = np.isfinite(y)
    bins, captions, y, prediction = bins[valid], captions[valid], y[valid], prediction[valid]
    if not len(y):
        return pd.DataFrame(columns=["分箱", "分箱标签", "样本总数", "目标值均值", "预测均值"])
    caption_map = dict(zip(bins, captions))
    table = compute_bin_stats(bins, y, target_type="continuous", round_digits=False)
    columns = ["分箱", "样本总数", "样本占比", "目标值均值", "目标值标准差", "目标值最小值", "目标值最大值"]
    table = table[columns].copy()
    table.insert(1, "分箱标签", table["分箱"].map(caption_map))
    groups = [bins == value for value in table["分箱"]]
    if margins:
        total = compute_bin_stats(np.zeros(len(y), dtype=int), y, target_type="continuous", round_digits=False)
        total = total[columns].copy()
        total["分箱"] = "合计"
        total.insert(1, "分箱标签", "合计")
        table = pd.concat([table, total], ignore_index=True)
        groups.append(np.ones(len(y), dtype=bool))
    table["预测均值"] = [float(prediction[mask].mean()) for mask in groups]
    table["平均残差"] = [float((prediction[mask] - y[mask]).mean()) for mask in groups]
    table["均方根误差"] = [error_values(y[mask], prediction[mask])["均方根误差"] for mask in groups]
    table["平均绝对误差"] = [error_values(y[mask], prediction[mask])["平均绝对误差"] for mask in groups]
    if amount is not None:
        from ..core.metrics._semantics import validate_amounts

        amounts = validate_amounts(current.X[amount].to_numpy(), len(current.X))[valid]
        table["金额合计"] = [float(amounts[mask].sum()) for mask in groups]
    return table


def grouped_metrics(report, column, freq=None):
    from ..utils.dates import date_groups

    rows = []
    for key, dataset in report._datasets.items():
        values = date_groups(dataset.X[column], freq or "MS") if freq is not None else dataset.X[column]
        frame = pd.DataFrame({"分组": np.asarray(values), "位置": np.arange(len(values))})
        valid = report._get_target_valid_mask(key)
        all_weights = report._metric_weight(key)
        weights = None
        if all_weights is not None:
            weights = np.zeros(len(dataset.X))
            weights[valid] = all_weights
        for group, subset in frame.groupby("分组", dropna=False, sort=False):
            positions = subset["位置"].to_numpy()
            positions = positions[valid[positions]]
            y = np.asarray(dataset.y, dtype=float)[positions]
            p = dataset.prediction[positions]
            weight = None if weights is None else weights[positions]
            metrics = error_values(y, p, weight) if weight is None or weight.sum() > 0 else error_values([], [])
            rows.append(
                {"数据集": dataset.label, "时间" if freq is not None else column: group, "样本数": len(y), **metrics}
            )
    return pd.DataFrame(rows)


def export_excel(report, filepath, options):
    """沿用 ExcelWriter 主题及原子保存，单独组织连续目标报告章节。"""
    import matplotlib.pyplot as plt
    from ..excel import ExcelWriter, dataframe2excel
    from ..core.viz.style import get_palette
    from ._model_layout import configuration_table, fit_metadata_sheet, record_identifier

    result = report._execution_result
    for column in (options["date"], options["group"], options["amount"]):
        if column is not None:
            missing = [d.label for d in report._datasets.values() if column not in d.X]
            if missing:
                raise ValueError(f"请求字段 {column} 在以下数据集中缺失: {missing}")
    metrics = report.get_metrics()
    bins = {
        key: report.get_bin_table(
            key, method=options["method"], max_n_bins=options["max_n_bins"], amount=options["amount"]
        )
        for key in report._datasets
    }
    result.success("核心数据计算")
    result.current_section = "工作簿渲染"
    writer = ExcelWriter()
    from ._training_process_report import process_sheet_name

    training_name = process_sheet_name(report, options.get("training_sheet_index"))
    sheet_names = [
        "1-基本信息",
        "2-模型性能",
        "3-入模变量分析",
        "4-稳定性分析",
        "5-模型参数",
        "6-模型部署需求",
        training_name,
    ]
    positions, section_counts = {}, {}

    def section(sheet_name, title, table, **kwargs):
        ws = writer.get_sheet_by_name(sheet_name)
        ws.sheet_view.showGridLines = False
        row = positions.get(sheet_name, 2)
        if "-" in sheet_name and sheet_name.split("-")[0].isdigit():
            number, caption = sheet_name.split("-", 1)
            if row == 2:
                writer.insert_value2sheet(ws, "B2", f"{number}、{caption}", style="header")
                row = 5
            section_counts[sheet_name] = section_counts.get(sheet_name, 0) + 1
            title = f"{number}.{section_counts[sheet_name]}、{title}"
        writer.insert_value2sheet(
            ws, (row, 2), value=title, style="header", end_space=(row, max(4, len(table.columns) + 1))
        )
        printable = (
            table.map(lambda v: str(v) if isinstance(v, (dict, list, tuple)) else v)
            if hasattr(table, "map")
            else table.applymap(lambda v: str(v) if isinstance(v, (dict, list, tuple)) else v)
        )
        if "样本占比" in table:
            kwargs.setdefault("percent_cols", ["样本占比"])
        end, _ = dataframe2excel(
            printable,
            writer,
            sheet_name=ws,
            start_row=row + 2,
            start_col=2,
            index=False,
            auto_width=True,
            speed="fast",
            **kwargs,
        )
        positions[sheet_name] = max(end + 3, row + len(table) + 5)
        return ws

    contents = pd.DataFrame({"章节": sheet_names})
    section("目录", options["model_name"] or "回归模型评估报告", contents)
    for i, name in enumerate(sheet_names):
        writer.insert_hyperlink2sheet(writer.get_sheet_by_name("目录"), (5 + i, 2), hyperlink=f"#'{name}'!B2")
    basic = []
    for dataset in report._datasets.values():
        y = np.asarray(dataset.y, dtype=float)
        finite = np.isfinite(y)
        row = {
            "数据集": dataset.label,
            "输入样本数": len(y),
            "有效目标数": int(finite.sum()),
            "目标均值": float(y[finite].mean()) if finite.any() else np.nan,
        }
        if options["date"]:
            dates = pd.to_datetime(dataset.X[options["date"]])
            row.update(开始日期=dates.min(), 结束日期=dates.max())
        if options["amount"]:
            row["金额合计"] = float(dataset.X[options["amount"]].sum())
        basic.append(row)
    section("1-基本信息", "样本与目标", pd.DataFrame(basic))
    section(
        "1-基本信息",
        "评估口径",
        pd.DataFrame(
            {
                "配置项": ["任务", "目标", "残差", "基准预测", "分箱统计", "项目说明", "数据来源"],
                "配置值": [
                    "回归",
                    report._target_name,
                    "预测值减实际值",
                    "固定为首个训练数据集的加权目标均值",
                    "训练集确定切点，按件数统计；不继承评估权重",
                    options["project_desc"],
                    options["data_source"],
                ],
            }
        ),
    )
    section("2-模型性能", "预测误差与均值基准", metrics)
    for key, dataset in report._datasets.items():
        section("2-模型性能", f"{dataset.label}预测分箱", bins[key])
    if options["with_plots"]:
        result.current_section = "图形生成"
        folder = Path(filepath).parent / f"{Path(filepath).stem}_assets"
        folder.mkdir(parents=True, exist_ok=True)
        colors = get_palette()
        plotted = 0
        for i, (key, dataset) in enumerate(report._datasets.items()):
            y, p = report._get_valid_target_arrays(key)
            if not len(y):
                continue
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            try:
                axes[0].scatter(y, p, alpha=0.55, s=14, color=colors[0])
                bounds = [float(min(y.min(), p.min())), float(max(y.max(), p.max()))]
                axes[0].plot(bounds, bounds, color=colors[1], linestyle="--")
                axes[0].set(xlabel="实际值", ylabel="预测值", title=f"{dataset.label}：实际与预测")
                axes[1].scatter(p, p - y, alpha=0.55, s=14, color=colors[0])
                axes[1].axhline(0, color=colors[1], linestyle="--")
                axes[1].set(xlabel="预测值", ylabel="预测值减实际值", title="残差分布")
                image = folder / f"regression_{i}.png"
                figure.savefig(image, dpi=150)
            finally:
                plt.close(figure)
            ws = writer.get_sheet_by_name("2-模型性能")
            start = positions[ws.title]
            end, _ = writer.insert_pic2sheet(ws, str(image), (start, 2), figsize=(1000, 400))
            positions[ws.title] = end + 3
            plotted += 1
        if plotted:
            result.success("图形生成")
        else:
            result.skip("图形生成", "没有带有效连续目标的样本")
    if options["show_importance"] and report.explain_config["enabled"]:
        from ._shap_importance_report import write_importance

        sheet = writer.get_sheet_by_name("3-入模变量分析")
        writer.insert_value2sheet(sheet, "B2", "3、入模变量分析", style="header")
        section_counts[sheet.title] = 1
        positions[sheet.title] = write_importance(
            report, writer, sheet, 5, filepath, options["with_plots"], title="3.1、SHAP 特征重要性"
        )
    train = report._datasets[report._train_key]
    described = train.X[report.feature_names].describe(include="all").T.reset_index()
    described = described.rename(
        columns={
            "index": "变量名称",
            "count": "非空样本数",
            "mean": "均值",
            "std": "标准差",
            "min": "最小值",
            "max": "最大值",
            "unique": "唯一值数",
            "top": "众数",
            "freq": "众数样本数",
        }
    )
    if options["feature_map"]:
        described.insert(1, "变量说明", described["变量名称"].map(options["feature_map"]).fillna(""))
    section("3-入模变量分析", "训练变量描述", described)
    correlation = report.get_features_corr()
    if not correlation.empty:
        section("3-入模变量分析", "数值特征相关性", correlation.rename_axis("变量名称").reset_index())
    if options["feature_info"] is not None:
        section("3-入模变量分析", "变量信息", options["feature_info"])
    if options["show_importance"]:
        importance = report.get_feature_importance()
        if not importance.empty:
            section("3-入模变量分析", "原生特征重要性", importance.rename_axis("变量名称").reset_index())
    for feature in report.feature_names:
        section(
            "3-入模变量分析",
            f"{feature}训练分箱",
            report.get_feature_bin_table(
                feature, report._train_key, max_n_bins=options["max_n_bins"], method=options["method"]
            ),
        )
    stability = metrics.loc[metrics["统计项"].eq("PSI")]
    if not stability.empty:
        section("4-稳定性分析", "相对训练预测分布 PSI", stability)
    if options["date"]:
        section("4-稳定性分析", "分期预测误差", grouped_metrics(report, options["date"], options["freq"] or "MS"))
    if options["group"]:
        section("4-稳定性分析", "分组预测误差", grouped_metrics(report, options["group"]))
    if "4-稳定性分析" not in positions:
        section(
            "4-稳定性分析",
            "预测分布",
            pd.DataFrame(
                [
                    {
                        "数据集": d.label,
                        "预测均值": float(np.mean(d.prediction)),
                        "预测标准差": float(np.std(d.prediction)),
                    }
                    for d in report._datasets.values()
                ]
            ),
        )
    section(
        "5-模型参数",
        "模型配置",
        configuration_table(report.model_context_["最终模型参数"]),
        decimal=None,
    )
    section(
        "6-模型部署需求",
        "模型输入字段",
        pd.DataFrame(
            {"入参字段": report.feature_names, "数据类型": [str(train.X[name].dtype) for name in report.feature_names]}
        ),
    )
    if options["include_sample_records"]:
        sample_tables = []
        for dataset in report._datasets.values():
            fields = options["loc_cols"] or []
            fields = [fields] if isinstance(fields, str) else list(fields)
            selected = dataset.X[fields].iloc[:10].copy().reset_index(drop=True)
            for field in fields:
                selected[field] = selected[field].map(record_identifier)
            selected.insert(0, "数据集", dataset.label)
            selected["实际值"] = np.asarray(dataset.y)[:10]
            selected["预测值"] = dataset.prediction[:10]
            selected["残差"] = selected["预测值"] - selected["实际值"]
            sample_tables.append(selected)
        section("6-模型部署需求", "预测样例（各数据集前十行）", pd.concat(sample_tables, ignore_index=True))
        result.success("生产订单测试用例")
    from ._training_process_report import write_process

    write_process(report, writer, filepath, options["with_plots"], training_name)
    for name in ("5-模型参数", "6-模型部署需求", training_name):
        fit_metadata_sheet(writer.get_sheet_by_name(name), writer.style_sheet.column_dimensions["C"].width)
    for name in sheet_names:
        writer.insert_hyperlink2sheet(writer.get_sheet_by_name(name), "B2", hyperlink="#'目录'!B2")
        result.success(f"工作表：{name}")
    from ._model_layout import fit_report_titles

    for sheet in writer.workbook:
        if sheet.title not in {"初始化", "目录", "报告执行状态"}:
            fit_report_titles(sheet)
    result.success("工作表：目录")
    result.current_section = "工作簿发布"
    result.ensure_publishable()
    section("报告执行状态", "报告完整" if result.complete else "报告不完整", result.status_table())
    writer.move_sheet(writer.get_sheet_by_name("报告执行状态"), index=0)
    writer.save(filepath)
    return filepath
