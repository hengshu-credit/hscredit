"""真实放款数据上的全模型接口与指标口径验收。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import hscredit as hc
from hscredit.core.rules import Rule
from hscredit.report import ModelReport

MODELS = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "SVM",
    "DecisionTreeClassifier",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NGBoost",
    "ScoreCard",
    "RoundScoreCard",
    "ProbabilityScoreCard",
    "RulesClassifier",
    "ProbabilityCalibrator",
]


@pytest.fixture(scope="module")
def loan_data():
    path = Path(__file__).resolve().parents[2] / "examples/hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("真实放款验收数据未随仓库分发；请在本地提供 examples/hscredit_yyp.xlsx")
    data = pd.read_excel(path)
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = data["FPD"].to_numpy(dtype=int)
    train, valid = train_test_split(np.arange(len(y)), test_size=0.2, random_state=37, stratify=y)
    # 标度缩放只用于避免 LR/SVM 在原始量纲上迭代不收敛，评分卡另走真实 WOE。
    X = X.fillna(X.iloc[train].median())
    return X / X.iloc[train].std().replace(0, 1), y, train, valid, data["放款金额"].to_numpy(dtype=float)


def build_model(name):
    try:
        cls = getattr(hc, name)
    except (ImportError, ValueError) as exc:
        pytest.skip(f"可选后端不可用: {exc}")
    common = dict(target="FPD")
    if name == "LogisticRegression":
        return cls(max_iter=1500, calculate_stats=False, **common)
    if name in {"RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "NGBoost"}:
        return cls(n_estimators=4, n_jobs=1, random_state=37, **common)
    if name == "CatBoost":
        return cls(iterations=4, depth=2, n_jobs=1, random_state=37, allow_writing_files=False, **common)
    if name in {"SVM", "DecisionTreeClassifier"}:
        return cls(n_jobs=1, random_state=37, **common)
    if name in {"ScoreCard", "RoundScoreCard"}:
        return cls(**common)
    if name == "ProbabilityScoreCard":
        return cls(model=build_model("LogisticRegression"), prefit=False, **common)
    if name == "RulesClassifier":
        return cls(rules=[Rule("`衡枢鉴真分老客版` < 6"), Rule("`近六个月非银多头机构数` > 1")], n_jobs=1, **common)
    return cls(model=build_model("LogisticRegression"), calib_ratio=0.25, random_state=37, **common)


def model_data(name, loan_data):
    X, y, train, valid, weights = loan_data
    model = build_model(name)
    if name in {"ScoreCard", "RoundScoreCard"}:
        binner = hc.OptimalBinning(method="quantile", max_n_bins=4).fit(X.iloc[train], y[train])
        model.set_params(binner=binner)
        X = binner.transform(X, metric="woe")
    return model, X.iloc[train], y[train], X.iloc[valid], y[valid], weights[valid]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("input_mode", ["named_frame", "pairs"])
def test_all_models_share_eval_metrics_history_and_report(name, input_mode, loan_data):
    model, X, y, VX, vy, weights = model_data(name, loan_data)
    metric = hc.get_loss("cost_sensitive", fn_cost=7).metric(kind="business")
    eval_set = {"时间验证": VX.assign(FPD=vy)} if input_mode == "named_frame" else [(VX, vy)]
    model.fit(X, y, eval_set=eval_set, eval_metric=metric)
    expected = metric(vy, model.predict_proba(VX)[:, 1])
    actual = model.get_eval_results()
    assert actual.loc[actual["指标"] == metric.name, "数值"].iloc[0] == pytest.approx(expected)
    assert model.get_training_history().iloc[-1]["状态"] == "完成"
    weighted = model.evaluate(VX, vy, sample_weight=weights, metrics=metric)
    assert weighted[metric.name] == pytest.approx(metric(vy, model.predict_proba(VX)[:, 1], sample_weight=weights))
    report = model.model_report(datasets={"验证": (VX, vy)}, metrics=[metric], verbose=False)
    assert isinstance(report, ModelReport)
    table = report.get_metrics().set_index("统计项")
    assert table.loc[metric.name, "验证"] == pytest.approx(expected)
    assert "最终评估" in report.get_training_process()
    if name in {"GradientBoosting", "XGBoost", "LightGBM", "CatBoost", "NGBoost"}:
        assert not model.get_learning_curve().empty
    else:
        assert model.get_learning_curve().empty


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("cv", [None, 2])
def test_all_trainable_models_support_explicit_cv_or_single_validation(name, cv, loan_data):
    model, X, y, VX, vy, _ = model_data(name, loan_data)
    metric = hc.get_metric("classification_cost", fn_cost=7)
    tuner = hc.ModelTuner(model, search_space={}, metric=metric, cv=cv, n_jobs=1, random_state=37)
    tuner.fit(X, y, n_trials=1, show_progress_bar=False, eval_set={"时间验证": (VX, vy)} if cv is None else None)
    assert len(tuner.get_fold_models()) == (1 if cv is None else 2)
    best = tuner.get_best_model()
    assert best.tuner is tuner
    assert best.get_training_history().iloc[-1]["样本数"] == len(y)
    assert np.isfinite(best.predict_proba(VX)).all()


def test_ngboost_objective_drives_training_and_validation_metric(loan_data):
    model, X, y, VX, vy, weights = model_data("NGBoost", loan_data)
    loss = hc.get_loss("focal", alpha=0.8)
    metric = hc.get_metric("auc")
    model.set_params(objective=loss, eval_metric=metric, early_stopping_rounds=2)
    model.fit(X, y, eval_set={"监控": (X, y), "验证": (VX, vy)}, val_sample_weight=weights)
    assert model.native_params_["Score"].__name__ == f"NGBoost_{loss.name}"
    assert set(model.get_eval_results()["数据集"]) == {"监控", "验证"}
    assert model._evals_result["val"][metric.name][-1] >= 0  # 早停内部最小化，公开值恢复原方向。
    expected = metric(vy, model.predict_proba(VX)[:, 1], sample_weight=weights)
    assert model.get_eval_results().query("数据集 == '验证'")["数值"].iloc[0] == pytest.approx(expected)
    assert model.get_learning_curve().query("数据集 == '验证'")["数值"].iloc[-1] == pytest.approx(expected)
    tuner = hc.ModelTuner(build_model("NGBoost"), search_space={}, loss=loss, metric=metric, n_jobs=1)
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().native_params_["Score"].__name__ == f"NGBoost_{loss.name}"


@pytest.mark.parametrize("name", ["RandomForest", "ExtraTrees", "GradientBoosting", "SVM", "DecisionTreeClassifier"])
def test_unsupported_training_loss_is_not_silently_ignored(name, loan_data):
    model, X, y, _, _, _ = model_data(name, loan_data)
    model.set_params(objective=hc.get_loss("focal"))
    with pytest.raises(ValueError, match="不支持自定义 objective"):
        model.fit(X, y)


def test_nested_model_refit_invalidates_wrapped_report(loan_data):
    model, X, y, VX, vy, _ = model_data("ProbabilityScoreCard", loan_data)
    model.fit(X, y)
    report = model.model_report(datasets={"验证": (VX, vy)}, verbose=False)
    model.model_.fit(X, 1 - y)
    with pytest.raises(ValueError, match="模型状态已变化"):
        report.get_metrics()


@pytest.mark.parametrize("method", ["standard", "linear", "quantile", "boxcox"])
def test_probability_transforms_preserve_history(method, loan_data):
    model, X, y, VX, _, _ = model_data("LogisticRegression", loan_data)
    model.fit(X, y)
    transformer = hc.ScoreTransformer(method=method).fit(proba=model.predict_proba(X)[:, 1])
    assert transformer.get_training_history().iloc[-1]["样本数"] == len(y)
    assert np.isfinite(transformer.predict(model.predict_proba(VX)[:, 1])).all()
    assert transformer.get_learning_curve().empty


@pytest.mark.parametrize("name", ["PlattCalibrator", "IsotonicCalibrator", "BetaCalibrator", "HistogramCalibrator"])
def test_probability_calibration_algorithms_use_same_metric(name, loan_data):
    model, X, y, VX, vy, weights = model_data("LogisticRegression", loan_data)
    model.fit(X, y)
    calibrator = getattr(hc, name)().fit(y_prob=model.predict_proba(X)[:, 1], y_true=y)
    p = model.predict_proba(VX)[:, 1]
    metric = hc.get_metric("classification_cost", fn_cost=7)
    assert calibrator.evaluate(p, vy, metrics=metric, sample_weight=weights)[metric.name] == pytest.approx(
        metric(vy, calibrator.calibrate(p), sample_weight=weights)
    )
    assert calibrator.get_training_history().iloc[-1]["状态"] == "完成"


@pytest.mark.parametrize("method", ["linear", "quantile", "binning"])
def test_drift_transform_preserves_fit_history(method, loan_data):
    model, X, y, VX, vy, _ = model_data("LogisticRegression", loan_data)
    model.fit(X, y)
    calibrator = hc.ScoreDriftCalibrator(method=method).fit(model, VX, vy, X_reference=X, y_reference=y)
    assert calibrator.get_training_history().iloc[-1]["样本数"] == len(vy)
    assert np.isfinite(calibrator.predict_score(VX)).all()


def test_rules_classification_metrics_follow_logic_and_threshold():
    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1]})
    model = hc.RulesClassifier([Rule("a > 0"), Rule("b > 0")], logic="and", output_mode="individual", n_jobs=1)
    model.fit(X, [0, 0, 0, 1], eval_set=(X, [0, 0, 0, 1]), eval_metric="accuracy")
    assert model.evaluate(X, [0, 0, 0, 1], metrics="accuracy")["Accuracy"] == 1


def test_calibrated_model_reports_actual_calibrated_probability(loan_data):
    base, X, y, VX, vy, _ = model_data("LogisticRegression", loan_data)
    base.fit(X, y)
    calibrator = hc.ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y)
    model = hc.CalibratedModel(base, calibrator)
    metric = hc.get_metric("classification_cost", fn_cost=7)
    report = model.model_report(datasets={"验证": (VX, vy)}, metrics=metric, verbose=False)
    assert report.get_metrics().set_index("统计项").loc[metric.name, "验证"] == pytest.approx(
        model.evaluate(VX, vy, metrics=metric)[metric.name]
    )
    assert "底层模型训练记录" in report.get_training_process()


def test_gradient_boosting_final_refit_disables_internal_holdout(loan_data):
    model, X, y, VX, vy, _ = model_data("GradientBoosting", loan_data)
    model.set_params(n_estimators=12, n_iter_no_change=2)
    tuner = hc.ModelTuner(model, search_space={}, metric="auc", n_jobs=1)
    tuner.fit(X, y, n_trials=1, eval_set=(VX, vy), show_progress_bar=False)
    assert tuner.get_best_model().get_params()["n_iter_no_change"] is None


@pytest.mark.parametrize("name", MODELS)
def test_dataframe_search_to_strict_excel_report(name, loan_data, tmp_path):
    """从同一模型的 target 配置出发，连续执行各入口，不手工补传丢失的元数据。"""
    model, X, y, VX, vy, _ = model_data(name, loan_data)
    train, valid = X.assign(FPD=y), VX.assign(FPD=vy)
    metric = hc.get_metric("classification_cost", fn_cost=7)
    tuner = hc.ModelTuner(model, search_space={}, metric=metric, n_jobs=1, random_state=37)
    tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    values = best.evaluate(valid, metrics=metric)
    report = best.model_report(datasets={"验证": valid}, metrics=[metric], verbose=False, n_jobs=1)
    assert report.get_metrics().set_index("统计项").loc[metric.name, "验证"] == pytest.approx(values[metric.name])
    result = report.to_excel(
        str(tmp_path / f"{name}.xlsx"),
        with_plots=False,
        show_importance=False,
        include_sample_records=False,
        mode="strict",
        return_result=True,
    )
    assert result.complete
    assert "调参记录" in report.get_training_process()
