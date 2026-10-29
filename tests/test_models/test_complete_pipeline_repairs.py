"""完整 Pipeline 的真实参数搜索、分箱拟合与报告元信息回归测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from hscredit import (
    ModelTuner,
    ScoreCard,
    RoundScoreCard,
    ProbabilityScoreCard,
    LogisticRegression,
    OptimalBinning,
    VarianceSelector,
    RulesClassifier,
)
from hscredit.core.binning import QuantileBinning
from hscredit.core.rules import Rule
from hscredit.report import ModelReport


@pytest.fixture
def data():
    rng = np.random.default_rng(43)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["评分", "次数"])
    return X, (X["评分"] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)


@pytest.mark.parametrize("cls", [ScoreCard, RoundScoreCard])
def test_raw_scorecard_tuning_refits_cloned_binning_inside_each_fold(cls, data, monkeypatch):
    X, y = data
    fitted_rows = []
    original = QuantileBinning.fit

    def tracked(self, X, y=None, **kwargs):
        fitted_rows.append(set(X.index))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(QuantileBinning, "fit", tracked)
    binner = OptimalBinning(method="quantile", max_n_bins=3, monotonic=False, n_jobs=1)
    model = Pipeline(
        [
            ("筛选", VarianceSelector(threshold=0.0, n_jobs=1)),
            ("模型", cls(binner=binner, lr_model=LogisticRegression(calculate_stats=False))),
        ]
    )
    tuner = ModelTuner(
        model,
        search_space={"模型__lr_model__C": [0.2, 2.0]},
        cv=2,
        fit_params={"模型__input_type": "raw"},
        n_jobs=1,
        random_state=42,
    )
    tuner.fit(X, y, n_trials=2, show_progress_bar=False)
    assert fitted_rows and all(len(rows) == 40 for rows in fitted_rows)
    best = tuner.get_best_model()
    assert len(fitted_rows[-1]) == 80
    assert not getattr(binner, "_is_fitted", False)
    report = ModelReport(best, datasets={"测试": (X, y)}, n_jobs=1)
    assert report._is_scorecard_model()
    assert not report._scorecard_points_table().empty
    assert not report._scorecard_formula_table().empty
    process = report.get_training_process()
    assert "特征筛选记录" in process and "预处理步骤" in process
    assert process["特征筛选记录"].iloc[0]["选中特征数"] == 2
    assert np.isfinite(best.predict_proba(X)).all()


def test_pipeline_probability_scorecard_reads_fitted_model_metadata(data):
    X, y = data
    card = ProbabilityScoreCard(model=LogisticRegression(calculate_stats=False), prefit=False)
    pipeline = Pipeline([("筛选", VarianceSelector(threshold=0.0, n_jobs=1)), ("模型", card)]).fit(X, y)
    report = ModelReport(pipeline, datasets={"测试": (X, y)}, n_jobs=1)
    assert report._is_scorecard_model()
    assert report._scorecard_lr_model() is card.model_
    assert not report._scorecard_lr_summary().empty
    assert not report._scorecard_scale_table().empty


def test_calibrated_inference_wrapper_supports_pipeline_prediction(data):
    from hscredit import ProbabilityCalibrator, CalibratedModel

    X, y = data
    base = LogisticRegression(calculate_stats=False).fit(X, y)
    calibrator = ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y)
    wrapper = CalibratedModel(base, calibrator)
    pipeline = Pipeline([("筛选", VarianceSelector(n_jobs=1).fit(X, y)), ("模型", wrapper)])
    np.testing.assert_allclose(pipeline.predict_proba(X), calibrator.predict_proba(X))
    with pytest.raises(ValueError, match="推理组合"):
        wrapper.fit(X, y)


@pytest.mark.parametrize("logic,expected", [("or", [0, 1, 1, 1]), ("and", [0, 0, 0, 1])])
@pytest.mark.parametrize("pipeline", [False, True])
def test_rules_logic_parameter_search_preserves_public_decision(logic, expected, pipeline):
    X = pd.DataFrame({"甲": [0, 1, 0, 1] * 8, "乙": [0, 0, 1, 1] * 8})
    y = np.array([0, 0, 0, 1] * 8)
    base = RulesClassifier([Rule("甲>0"), Rule("乙>0")], n_jobs=1)
    model = clone(base).set_params(logic=logic).fit(X, y)
    assert model.logic.value == logic
    np.testing.assert_array_equal(model.predict(X)[:4], expected)
    estimator = Pipeline([("筛选", VarianceSelector(n_jobs=1)), ("模型", base)]) if pipeline else base
    key = "模型__logic" if pipeline else "logic"
    tuner = ModelTuner(estimator, search_space={key: [logic]}, metric="accuracy", n_jobs=1, random_state=42)
    tuner.fit(X, y, n_trials=1, eval_set=(X, y), show_progress_bar=False)
    best = tuner.get_best_model()
    np.testing.assert_array_equal(best.predict(X)[:4], expected)
    assert tuner.best_score_ == pytest.approx(np.mean(np.asarray(expected * 8) == y))
    report = ModelReport(best, datasets={"验证": (X, y)}, metrics=["accuracy", "auc"], n_jobs=1)
    assert report.get_metrics().set_index("统计项").loc["Accuracy", "验证"] == pytest.approx(tuner.best_score_)
