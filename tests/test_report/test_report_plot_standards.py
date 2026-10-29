"""报告图形使用统一样式，横向图组在最终列宽下保持一致。"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from PIL import Image
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression, LogisticRegression

from hscredit.report import ModelReport
from hscredit.core.viz.style import PRIMARY_COLORS, get_defaults, get_font_sizes


def test_plot_rows_align_after_all_tables_have_changed_column_widths(tmp_path, monkeypatch):
    X = pd.DataFrame({"变量甲": np.arange(40), "变量乙": np.arange(40) % 3})
    y = (X["变量甲"] > 19).astype(int)
    report = ModelReport(
        LogisticRegression().fit(X, y),
        datasets={"训练集": (X, y), "验证集": (X, y), "测试集": (X, y)},
        n_jobs=1,
    )
    picture = tmp_path / "排版用图.png"
    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot([0, 1], [0, 1])
    figure.savefig(picture)
    plt.close(figure)
    paths = {f"model_{key}": [str(picture)] * 4 for key in report._datasets}
    paths.update({f"feat_bin_{feature}": [str(picture)] * 3 for feature in X})
    paths.update({f"feat_hist_{feature}": [str(picture)] * 3 for feature in X})
    monkeypatch.setattr(report, "_export_plots", lambda *args, **kwargs: (paths, {}))
    output = tmp_path / "图组排版.xlsx"
    report.to_excel(output, with_plots=True, mode="strict")
    book = load_workbook(output)
    for name, group_size, row_count in [("2-模型性能", 4, 3), ("3-入模变量分析", 6, 2)]:
        rows = defaultdict(list)
        for image in book[name]._images:
            anchor = image.anchor
            rows[anchor._from.row].append((anchor._from.col, anchor._from.colOff, anchor.ext.cx, anchor.ext.cy))
        assert len(rows) == row_count
        arrangements = list(rows.values())
        assert all(len(row) == group_size for row in arrangements)
        assert all(row == arrangements[0] for row in arrangements)
        assert arrangements[0][0][0] == 1
        assert all(left[0] < right[0] for left, right in zip(arrangements[0], arrangements[0][1:]))
    book.close()


def test_new_report_plots_use_hscredit_axes_palette_and_export_defaults(tmp_path, monkeypatch):
    from hscredit.core.viz import utils

    saved = {}
    original = utils.save_figure

    def capture(figure, path, dpi=240):
        saved[Path(path).stem] = figure
        assert dpi == get_defaults()["dpi"]
        for axis in figure.axes:
            assert axis.spines["left"].get_edgecolor() == to_rgba(PRIMARY_COLORS[0])
            assert not axis.spines["top"].get_visible()
            assert not axis.spines["right"].get_visible()
            assert to_rgba(axis.xaxis.label.get_color()) == to_rgba(PRIMARY_COLORS[0])
            assert axis.title.get_fontsize() == get_font_sizes()["title"]
        original(figure, path, dpi=dpi)
        with Image.open(path) as image:
            np.testing.assert_allclose(image.info["dpi"], [240, 240], atol=0.1)

    monkeypatch.setattr(utils, "save_figure", capture)
    X = pd.DataFrame({"变量甲": np.arange(12), "变量乙": np.arange(12) % 3})
    y = X["变量甲"] * 2 + X["变量乙"]
    report = ModelReport(
        LinearRegression().fit(X, y),
        datasets={"训练集": (X, y)},
        n_jobs=1,
        explain_config={"enabled": True, "scope": "importance", "max_samples": 6, "max_background": 6},
    )
    report._training_process_snapshot["训练曲线"] = pd.DataFrame(
        {
            "训练修订": [1] * 3,
            "数据集": ["训练集"] * 3,
            "迭代": [1, 2, 3],
            "指标": ["损失"] * 3,
            "数值": [0.6, 0.5, 0.4],
        }
    )
    report.to_excel(tmp_path / "统一绘图.xlsx", with_plots=True, mode="strict")
    assert {"regression_0", "shap_importance", "训练曲线_0"} <= set(saved)
    axis = saved["训练曲线_0"].axes[0]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.6, 0.5, 0.4])
    assert not axis.get_legend().get_frame_on()
    assert all(bar.get_hatch() == "/" for axis in saved["shap_importance"].axes for bar in axis.patches)
