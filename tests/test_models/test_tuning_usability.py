"""调参入口的参数语义、数据隔离与简便使用回归测试。"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit.core.models import AutoTuner, ModelTuner
from hscredit.core.models.losses import AmountWeightedLoss, FocalLoss, make_metric


class WeightedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, bias=0.0):
        self.bias = bias

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.weights_ = None if sample_weight is None else np.asarray(sample_weight)
        self.rate_ = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.rate_ + self.bias, 0.001, 0.999))
        return np.column_stack([1 - p, p])


@pytest.fixture
def data():
    return np.arange(48).reshape(24, 2), np.tile([0, 1], 12)


def tuner(**kwargs):
    return ModelTuner(WeightedClassifier, search_space={}, cv=2, n_jobs=1, **kwargs)


def test_direction_comes_from_each_metric():
    obj = tuner(metric=["auc", "ks_diff"])
    assert obj.directions == ["maximize", "minimize"]
    assert tuner(metric=FocalLoss()).direction == "minimize"
    with pytest.raises(ValueError, match="direction|方向"):
        tuner(metric=lambda y, p: np.mean(p))


def test_loss_is_distinct_from_search_objective():
    from hscredit.core.models import XGBoost

    loss = FocalLoss()
    obj = ModelTuner(XGBoost, loss=loss, metric=None)
    assert obj.direction == "minimize"
    assert obj.fixed_params["objective"] is loss
    with pytest.raises(TypeError, match="loss="):
        ModelTuner(XGBoost, objective=loss)
    with pytest.raises(ValueError, match="仅支持"):
        tuner(loss=loss)


def test_bound_amounts_rejected_before_creating_trials(data):
    obj = tuner(metric=AmountWeightedLoss(amounts=np.arange(1, 25)).metric())
    with pytest.raises(ValueError, match="绑定金额"):
        obj.fit(*data, n_trials=1, show_progress_bar=False)
    assert obj.study_ is None


@pytest.mark.parametrize("bad_cv", [[], [([0, 0, 1], [2, 3])], [([0, 1], [1, 2])], [([0, 1], [24, 25])]])
def test_invalid_cv_rejected_consistently(data, bad_cv):
    obj = ModelTuner(WeightedClassifier, search_space={}, cv=bad_cv)
    with pytest.raises(ValueError, match="交叉验证"):
        obj.evaluate_trials(*data, trial_points=[{}])
    with pytest.raises(ValueError, match="交叉验证"):
        obj.fit(*data, n_trials=1, show_progress_bar=False)


def test_pipeline_routes_weights_and_reuses_nested_params(data):
    pipeline = Pipeline([("scale", StandardScaler()), ("model", WeightedClassifier())])
    obj = ModelTuner(pipeline, search_space={"model__bias": [0.0, 0.1]}, cv=2, random_state=8)
    weights = np.arange(1, 25)
    obj.fit(*data, sample_weight=weights, n_trials=1, show_progress_bar=False)
    for fold in obj.get_trial_result(0)["各折"]:
        np.testing.assert_array_equal(fold["模型"].steps[-1][1].weights_, weights[fold["训练位置"]])
    np.testing.assert_array_equal(obj.get_best_model().steps[-1][1].weights_, weights)
    assert len(obj.evaluate_trials(*data, trial_points=[{"model__bias": 0.1}])) == 1
    assert len(obj.evaluate_study_trials()) == 1


def test_fit_resume_rejects_new_data_before_overwriting_old_context(data):
    obj = tuner()
    obj.fit(*data, n_trials=1, show_progress_bar=False)
    original_X = obj._X
    obj.fit(*data, n_trials=1, show_progress_bar=False)
    assert len(obj.study_.trials) == 2
    with pytest.raises(ValueError, match="续跑数据"):
        obj.fit(data[0] + 1, data[1], n_trials=1, show_progress_bar=False)
    assert obj._X is original_X
    assert len(obj.study_.trials) == 2


def test_re_evaluation_does_not_reuse_previous_fit_weights(data):
    obj = tuner(random_state=8)
    obj.fit(*data, fit_params={"sample_weight": np.ones(24)}, n_trials=1, show_progress_bar=False)
    new_X, new_y = data[0][:12], data[1][:12]
    assert len(obj.evaluate_trials(new_X, new_y, trial_points=[{}])) == 1
    assert len(obj.evaluate_study_trials(X=new_X, y=new_y)) == 1
    assert len(obj._fit_params["sample_weight"]) == 24


def test_notebook_function_metric_survives_save_load_and_resume(data, tmp_path):
    namespace = {"__name__": "__main__", "np": np}
    exec("def error(y, p, scale=1.):\n    return float(np.mean((y-p)**2) * scale)", namespace)
    metric = make_metric(namespace["error"], name="概率误差", greater_is_better=False, scale=2.0)
    obj = tuner(metric=metric, random_state=8)
    obj.fit(*data, n_trials=1, show_progress_bar=False)
    obj.get_best_model()
    restored = ModelTuner.load(obj.save(tmp_path / "tuner.pkl"))
    restored.fit(*data, n_trials=1, show_progress_bar=False)
    assert len(restored.study_.trials) == 2
    restored.metrics[0].metric.kwargs["scale"] = 3.0
    with pytest.raises(ValueError, match="参数或指标发生变化"):
        restored.fit(*data, n_trials=1, show_progress_bar=False)


def test_storage_resume_checks_data_contract_across_new_tuners(data, tmp_path):
    storage = "sqlite:///" + (tmp_path / "search.db").as_posix()
    obj = tuner(storage=storage, study_name="same", random_state=8)
    obj.fit(*data, n_trials=1, show_progress_bar=False)
    restored = tuner(storage=storage, study_name="same", load_if_exists=True, random_state=8)
    with pytest.raises(ValueError, match="Study 的数据"):
        restored.fit(data[0] + 1, data[1], n_trials=1, show_progress_bar=False)
    assert len(restored.study_.trials) == 1


def test_custom_loss_drops_ineffective_adaptive_class_weight(data):
    from hscredit.core.models import XGBoost

    obj = ModelTuner(XGBoost, loss=FocalLoss(), metric=None, cv=2, n_jobs=1, random_state=8)
    obj.fit(*data, n_trials=1, show_progress_bar=False)
    assert "scale_pos_weight" not in obj.search_space
    with pytest.raises(ValueError, match="不能搜索"):
        ModelTuner(XGBoost, loss=FocalLoss(), search_space={"scale_pos_weight": [1, 5]})
    with pytest.raises(ValueError, match="类别权重"):
        ModelTuner(XGBoost(scale_pos_weight=5), loss=FocalLoss(), search_space={})


def test_auto_tuner_loads_only_requested_model(monkeypatch):
    module = importlib.import_module("hscredit.core.models")
    original_getattr = module.__getattr__

    def guarded_getattr(name):
        if name in {"XGBoost", "LightGBM", "CatBoost", "NGBoost"}:
            raise AssertionError("不应导入其他可选模型")
        return original_getattr(name)

    for name in ("XGBoost", "LightGBM", "CatBoost", "NGBoost"):
        monkeypatch.delattr(module, name, raising=False)
    monkeypatch.setattr(module, "__getattr__", guarded_getattr)
    obj = AutoTuner.create("lr", search_space={"C": [0.1, 1.0]})
    assert list(obj.search_space) == ["C"]


def test_dataset_example_validates_real_loan_data():
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    frame = pd.read_excel(path, usecols=["衡枢鉴真分老客版", "FPD"]).dropna()
    frame = frame.iloc[:400]
    obj = AutoTuner.create(
        "lr",
        target="FPD",
        search_space={"C": [0.1, 1.0]},
        fixed_params={"max_iter": 100},
        cv=2,
        random_state=8,
        n_jobs=1,
    )
    obj.fit(frame, n_trials=2, show_progress_bar=False)
    assert obj.get_best_model().predict_proba(frame[["衡枢鉴真分老客版"]]).shape == (len(frame), 2)
    assert len(obj.get_oof_predictions()) == len(frame)
