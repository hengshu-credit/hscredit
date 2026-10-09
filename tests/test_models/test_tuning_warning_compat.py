"""常规调参不触发已弃用的终止分析接口，同时保留真实用户警告。"""

import importlib
from types import SimpleNamespace
import warnings

import numpy as np
import optuna
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from hscredit.core.models.tuning import ModelTuner


def _tuner(**kwargs):
    return ModelTuner(
        LogisticRegression(max_iter=200),
        search_space={},
        metric="auc",
        cv=2,
        n_jobs=1,
        random_state=42,
        **kwargs,
    )


@pytest.mark.parametrize("version", ["4.9.0", "4.9.0rc1", "5.0.0", "6.0.0"])
def test_new_optuna_does_not_import_deprecated_terminator(monkeypatch, version):
    monkeypatch.setattr(optuna, "__version__", version)
    original = importlib.import_module

    def no_terminator(name, *args, **kwargs):
        if name == "optuna.terminator":
            raise AssertionError("普通搜索不应调用弃用模块")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_terminator)
    X, y = make_classification(n_samples=80, n_features=4, random_state=12)
    tuner = _tuner()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    assert not caught, [str(w.message) for w in caught]
    assert tuner.study_.metric_names == ["AUC"]
    trial = tuner.study_.trials[0]
    assert len(trial.user_attrs["各折指标"]) == 2
    assert len(trial.intermediate_values) == 2
    assert len(tuner.get_trial_result(0)["各折"]) == 2


@pytest.mark.parametrize(
    "version,enabled,expected", [("4.8.0", None, True), ("4.8.0", False, False), ("4.9.0", True, True)]
)
def test_legacy_reporting_uses_public_api_only_when_enabled(monkeypatch, version, enabled, expected):
    calls = []
    original = importlib.import_module

    def stub(name, *args, **kwargs):
        if name == "optuna.terminator":
            return SimpleNamespace(report_cross_validation_scores=lambda trial, scores: calls.append((trial, scores)))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(optuna, "__version__", version)
    monkeypatch.setattr(importlib, "import_module", stub)
    tuner = _tuner(record_terminator_scores=enabled)
    trial = object()
    tuner._report_terminator_scores(trial, [np.float64(0.6), np.float64(0.7)])
    assert bool(calls) is expected
    if expected:
        assert calls == [(trial, [0.6, 0.7])]


def test_metric_naming_does_not_suppress_unrelated_user_warnings():
    X, y = make_classification(n_samples=80, n_features=4, random_state=12)

    def callback(study, trial):
        warnings.warn("用户弃用提醒", FutureWarning)
        warnings.warn("用户实验功能", optuna.exceptions.ExperimentalWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _tuner().fit(X, y, n_trials=1, callbacks=[callback], show_progress_bar=False)
    assert [str(w.message) for w in caught] == ["用户弃用提醒", "用户实验功能"]


def test_existing_study_names_are_not_rewritten_on_resume(monkeypatch):
    X, y = make_classification(n_samples=80, n_features=4, random_state=12)
    tuner = _tuner()
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    def fail(names):
        raise AssertionError("无需重复设置相同指标名")

    monkeypatch.setattr(tuner.study_, "set_metric_names", fail)
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    assert len(tuner.study_.trials) == 2


def test_explicit_legacy_switch_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="record_terminator_scores"):
        _tuner(record_terminator_scores="auto")
