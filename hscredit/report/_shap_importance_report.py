"""复用统一重要性综合图，保留原生重要性与 SHAP 各自的尺度。"""

from pathlib import Path
import numpy as np
import pandas as pd


def importance_table(report):
    explanation = report.get_model_explanation()
    if "失败原因" in explanation:
        raise ValueError(explanation["失败原因"])
    table = explanation["全局解释"][["特征", "平均绝对SHAP值", "SHAP重要性占比", "平均SHAP值"]].copy()
    native = report._get_raw_feature_importance()
    if native.notna().any():
        table.insert(1, "原生特征重要性", table["特征"].map(native))
    return table


def write_importance(report, writer, sheet, row, filepath, with_plots, title="1、SHAP 特征重要性"):
    from ..excel import dataframe2excel
    from ..core.viz import plot_model_feature_importance
    from ..core.viz.style import get_defaults
    import matplotlib.pyplot as plt

    if not report.explain_config["enabled"]:
        return row
    explanation = report.get_model_explanation()
    if "失败原因" in explanation:
        result = getattr(report, "_execution_result", None)
        if result is not None:
            result.failure("SHAP 特征重要性", explanation["失败原因"])
        writer.insert_value2sheet(sheet, (row, 2), title, style="header", end_space=(row, 3))
        end, _ = dataframe2excel(
            pd.DataFrame([{"失败原因": explanation["失败原因"]}]),
            writer,
            sheet_name=sheet,
            start_row=row + 2,
            index=False,
        )
        return end + 3
    table = importance_table(report)
    meta = explanation["元信息"]
    writer.insert_value2sheet(sheet, (row, 2), title, style="header", end_space=(row, len(table.columns) + 1))
    row, _ = dataframe2excel(
        table,
        writer,
        sheet_name=sheet,
        start_row=row + 2,
        index=False,
        decimal=None,
        auto_width=True,
        percent_cols=["SHAP重要性占比"],
    )
    info = {
        "解释数据": meta.get("解释数据集"),
        "解释样本数": meta["样本数"],
        "背景样本数": meta["背景数据"]["样本数"],
        "输出尺度": "坏样本概率" if meta["实际输出尺度"] == "probability" else "预测值",
        "SHAP算法": "排列式 SHAP" if meta["解释算法"] == "permutation" else meta["解释算法"],
        "最大解释次数": meta.get("最大解释次数"),
        "重要性定义": "各样本绝对 SHAP 值的均值；原生重要性保留模型自身口径",
    }
    row, _ = dataframe2excel(
        pd.DataFrame(info.items(), columns=["项目", "内容"]),
        writer,
        sheet_name=sheet,
        start_row=row + 2,
        index=False,
        auto_width=True,
    )
    if with_plots:
        result = explanation["解释结果"]
        data = result.data
        labels = None
        if report.explain_config["data"] is None:
            key = (
                report._resolve_dataset_key(report.explain_config["dataset"])
                if report.explain_config["dataset"] is not None
                else report._train_key
            )
            dataset = report._datasets[key]
            # 使用解释器同一抽样规则保存位置关系，重复索引不能通过 reindex 对齐标签。
            positions = pd.DataFrame({"位置": np.arange(len(dataset.X))}, index=dataset.X.index)
            sampled = explanation["解释器"]._sample(positions, report.explain_config["max_samples"])
            aligned = dataset.X.iloc[sampled["位置"].to_numpy()][data.columns]
            if not aligned.equals(data):
                raise ValueError("整合图的解释数据与标签抽样位置不一致")
            labels = np.asarray(report._get_y(key))[sampled["位置"].to_numpy()]
            if not np.isfinite(np.asarray(labels, dtype=float)).all():
                labels = None  # 显式缺失标签不伪造颜色或改变已解释的样本。
        native = report._get_raw_feature_importance().reindex(data.columns)
        has_native = native.notna().all() and np.isfinite(native).all()
        folder = Path(filepath).parent / (Path(filepath).stem + "_assets")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "shap_importance.png"
        figure = plot_model_feature_importance(
            report.model,
            data,
            labels,
            prediction_method={"probability": "predict_proba", "prediction": "predict", "score": "predict_score"}.get(
                result.model_output, result.model_output
            ),
            left_top_n=32,
            right_top_n=10,
            importance_source="raw" if has_native else "shap",
            shap_by_label=False,
            shap_values=result.values,
            feature_importance=native if has_native else None,
            random_state=report.explain_config["random_state"],
            title=f"模型特征重要性与 SHAP 综合分析（TOP{min(32, len(data.columns))}）",
            save=path,
            show=False,
            dpi=get_defaults()["dpi"],
        )
        try:
            # 按导出图片实际比例放入 Excel，避免综合看板被压扁。
            from PIL import Image

            with Image.open(path) as picture:
                width = 1200
                height = round(width * picture.height / picture.width)
            row, _ = writer.insert_pic2sheet(sheet, str(path), (row + 2, 2), figsize=(width, height))
        finally:
            plt.close(figure)
    return row + 3
