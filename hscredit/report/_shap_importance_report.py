"""原生重要性与 SHAP 分别呈现，不混用尺度或把均值方向当重要性。"""

from pathlib import Path
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
    from ..core.viz.style import get_palette
    import matplotlib.pyplot as plt

    if not report.explain_config["enabled"]:
        return row
    table = importance_table(report)
    meta = report.get_model_explanation()["元信息"]
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
        fields = [c for c in ("原生特征重要性", "平均绝对SHAP值") if c in table]
        shown = table.head(20).iloc[::-1]
        figure, axes = plt.subplots(
            1,
            len(fields),
            figsize=(6 * len(fields), max(3, 0.3 * len(shown) + 1.3)),
            squeeze=False,
            constrained_layout=True,
        )
        try:
            for index, field in enumerate(fields):
                axis = axes[0, index]
                axis.barh(shown["特征"], shown[field], color=get_palette()[index])
                axis.set_title("原生特征重要性" if field == "原生特征重要性" else "SHAP 特征重要性")
                axis.set_xlabel(field)
            folder = Path(filepath).parent / (Path(filepath).stem + "_assets")
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "shap_importance.png"
            figure.savefig(path, dpi=140)
        finally:
            plt.close(figure)
        row, _ = writer.insert_pic2sheet(
            sheet, str(path), (row + 2, 2), figsize=(600 * len(fields), max(300, 30 * len(shown) + 130))
        )
    return row + 3
