import numpy as np
import pandas as pd

from hscredit.core.eda import model_drift_report


class _ProbabilityModel:
    def predict_proba(self, X):
        prob = 1 / (1 + np.exp(-X["x1"].to_numpy()))
        return np.column_stack([1 - prob, prob])


def test_model_drift_report_combines_score_and_feature_drift():
    rng = np.random.RandomState(7)
    X_base = pd.DataFrame({
        "x1": rng.normal(0, 1, 120),
        "x2": rng.normal(1, 1, 120),
    })
    X_target = pd.DataFrame({
        "x1": rng.normal(0.5, 1.2, 120),
        "x2": rng.normal(1, 1, 120),
    })
    y_base = (X_base["x1"] > 0.3).astype(int)
    y_target = (X_target["x1"] > 0.3).astype(int)

    report = model_drift_report(
        _ProbabilityModel(),
        X_base,
        X_target,
        y_base=y_base,
        y_target=y_target,
        features=["x1", "x2"],
        psi_bins=5,
    )

    assert set(report) == {"评分漂移", "特征漂移", "漂移摘要"}
    assert np.isfinite(report["评分漂移"]["PSI"])
    assert report["评分漂移"]["模型性能"]["基准KS"] >= 0
    assert report["特征漂移"]["特征名"].tolist()
    assert "评分PSI" in report["漂移摘要"]["指标"].tolist()
