import numpy as np
import pandas as pd

from hscredit.core.models.evaluation import model_explain_report


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
