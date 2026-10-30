"""整合重要性图复用报告解释结果，TOP32 只限制展示且保持标签对齐。"""

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from hscredit.report import ModelReport
from hscredit.report._shap_importance_report import write_importance
from hscredit.excel import ExcelWriter
from hscredit.core import viz


class NoNativeRegressor(RegressorMixin, BaseEstimator):
    _supports_native_feature_importance = False

    def fit(self, X, y):
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X):
        return X.select_dtypes("number").sum(axis=1).to_numpy()


@pytest.mark.parametrize("native", [False, True])
def test_report_integrates_top32_without_reexplaining_or_truncating_table(tmp_path, monkeypatch, native):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(45, 40)), columns=[f"变量{i}" for i in range(40)])
    X.index = np.repeat(np.arange(15), 3)  # 标签不能简单按重复索引 reindex。
    y = np.arange(45, dtype=float) + 0.25
    model = (LinearRegression() if native else NoNativeRegressor()).fit(X, y)
    report = ModelReport(
        model,
        datasets={"验证": (X, y)},
        n_jobs=1,
        explain_config={
            "enabled": True,
            "scope": "importance",
            "dataset": "验证",
            "max_samples": 5,
            "max_background": 4,
            "max_evals": 81,
        },
    )
    explanation = report.get_model_explanation()
    expected = explanation["解释结果"]
    import shap

    def fail(*args, **kwargs):
        raise AssertionError("不得重复计算 SHAP")

    monkeypatch.setattr(shap, "Explainer", fail)
    original = viz.plot_model_feature_importance
    seen = []

    def capture(model, data, labels, **kwargs):
        pd.testing.assert_frame_equal(data, expected.data)
        np.testing.assert_allclose(kwargs["shap_values"], expected.values)
        selected = pd.DataFrame({"y": y}, index=X.index).sample(n=5, random_state=42).sort_index()
        np.testing.assert_array_equal(labels, selected.y)
        assert kwargs["importance_source"] == ("raw" if native else "shap")
        assert kwargs["shap_by_label"] is False
        fig = original(model, data, labels, **kwargs)
        axes = {ax.get_label(): ax for ax in fig.axes}
        assert len(axes["SHAP分布"].get_yticklabels()) == 32
        assert len(axes["特征重要性"].patches) == 32
        assert sum(ax.get_label() == "SHAP依赖" for ax in fig.axes) == 10
        means = np.abs(expected.values).mean(axis=0)
        values = report._get_raw_feature_importance().to_numpy() if native else means
        top = np.argsort(-values, kind="stable")[:32][::-1]
        assert [label.get_text() for label in axes["SHAP分布"].get_yticklabels()] == X.columns[top].tolist()
        np.testing.assert_allclose([p.get_width() for p in axes["特征重要性"].patches], values[top])
        seen.append(fig)
        return fig

    monkeypatch.setattr(viz, "plot_model_feature_importance", capture)
    writer = ExcelWriter()
    sheet = writer.get_sheet_by_name("变量分析")
    write_importance(report, writer, sheet, 2, tmp_path / "整合图.xlsx", True)
    assert len(seen) == 1 and len(sheet._images) == 1
    assert len(report.get_shap_importance()) == 40
    writer.workbook.close()


def test_cached_plot_supports_unlabelled_categorical_data_and_rejects_bad_shapes(monkeypatch):
    X = pd.DataFrame({"金额": [1.0, 2.0, np.nan], "类别": pd.Categorical(["甲", "乙", "甲"])})
    values = np.array([[0.2, 0.4], [0.1, -0.2], [0.0, 0.1]])
    fig = viz.plot_model_feature_importance(object(), X, shap_values=values, importance_source="shap", show=False)
    assert len(next(a for a in fig.axes if a.get_label() == "SHAP分布").get_yticklabels()) == 2
    assert sum(a.get_label() == "SHAP依赖" for a in fig.axes) == 2
    assert not any(a.get_ylabel() == "真实标签" for a in fig.axes)
    plt.close(fig)
    with pytest.raises(ValueError, match="形状"):
        viz.plot_model_feature_importance(object(), X, shap_values=values[:, :1], show=False)
    with pytest.raises(ValueError, match="全部字段"):
        viz.plot_model_feature_importance(object(), X, feature_importance=pd.Series({"金额": 1}), show=False)


def test_native_only_report_caps_top32_without_enabling_shap(monkeypatch):
    from hscredit.core.viz.utils import _threaded_agg_rendering

    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(60, 40)), columns=[f"变量{i}" for i in range(40)])
    y = X.sum(axis=1)
    report = ModelReport(LinearRegression().fit(X, y), datasets={"训练": (X, y)}, n_jobs=1)

    def reject_global_pyplot(*args, **kwargs):
        raise AssertionError("并行报告不能使用全局 pyplot 画布")

    monkeypatch.setattr(plt, "subplots", reject_global_pyplot)
    with _threaded_agg_rendering():
        fig = report._create_native_importance_figure()
    assert len(fig.axes[0].patches) == 32
    assert report._model_explanation_cache is None
    plt.close(fig)


def test_full_explanation_report_does_not_repeat_global_importance_graph(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from openpyxl import load_workbook

    X = pd.DataFrame({"金额": np.arange(30), "期限": np.arange(30) % 3})
    y = (X["金额"] > 12).astype(int)
    report = ModelReport(
        RandomForestClassifier(n_estimators=3, random_state=7).fit(X, y),
        datasets={"训练": (X, y)},
        n_jobs=1,
        explain_config={
            "enabled": True,
            "max_samples": 6,
            "max_background": 6,
            "n_bootstrap": 2,
            "representative_count": 1,
        },
    )
    path = tmp_path / "完整解释.xlsx"
    result = report.to_excel(path, with_plots=True, mode="strict", return_result=True)
    assert result.complete
    book = load_workbook(path)
    chart = (tmp_path / "完整解释_assets/shap_importance.png").read_bytes()
    assert sum(image._data() == chart for image in book["3-入模变量分析"]._images) == 1
    assert not (tmp_path / "完整解释_assets/feature_contribution.png").exists()
    assert not book["7-模型解释"]._images
    assert not book["5-模型参数"]._images
    book.close()
