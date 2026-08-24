import numpy as np
import pandas as pd
import pytest

from hscredit.core.models.explainability import model_explain_report


class _ImportanceModel:
    def __init__(self):
        self.feature_names_in_ = np.array(["age", "income", "score"])

    def get_feature_importances(self, importance_type="gain"):
        return pd.Series([0.2, 0.5, 0.3], index=self.feature_names_in_)


class _CoefModel:
    coef_ = np.array([[0.4, -0.1, 0.2]])


def test_model_explain_report_uses_model_feature_importance():
    report = model_explain_report(_ImportanceModel(), top_n=2)

    assert report["特征名"].tolist() == ["income", "score"]
    assert report["排名"].tolist() == [1, 2]
    assert set(["重要性", "重要性类型", "来源", "归一化重要性"]).issubset(report.columns)


def test_model_explain_report_falls_back_to_coefficients():
    X = pd.DataFrame(np.zeros((3, 3)), columns=["x1", "x2", "x3"])
    report = model_explain_report(_CoefModel(), X=X, importance_type="coef")

    assert report.loc[0, "特征名"] == "x1"
    assert report.loc[0, "影响方向"] == "正向"
    assert report.loc[2, "影响方向"] == "负向"


class _BrokenImportanceModel:
    feature_importances_ = np.array([0.2, 0.8])

    def get_feature_importances(self, importance_type="gain"):
        raise RuntimeError("模型重要性计算失败")


def test_model_explain_report_does_not_hide_real_model_errors():
    """已有公开方法的运行错误不能被 feature_importances_ 静默掩盖。"""
    with pytest.raises(RuntimeError, match="模型重要性计算失败"):
        model_explain_report(_BrokenImportanceModel())


def test_model_explain_report_rejects_nonpositive_top_n():
    """负数 head 语义不能冒充合法 top_n。"""
    with pytest.raises(ValueError, match="top_n"):
        model_explain_report(_ImportanceModel(), top_n=0)


def test_model_explain_report_preserves_nan_importance_without_integer_cast_failure():
    """缺失重要性可以保留，但不能在排名转整数时崩溃。"""

    class NanImportanceModel:
        def get_feature_importances(self, importance_type="gain"):
            return pd.Series([np.nan, 0.5], index=["缺失", "有效"])

    report = model_explain_report(NanImportanceModel())

    assert report.loc[report["特征名"] == "缺失", "排名"].isna().all()
    assert report.loc[report["特征名"] == "有效", "排名"].iloc[0] == 1
