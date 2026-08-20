"""ModelReport 指定预测方法与 callable 契约测试。"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from hscredit.exceptions import SerializationError, ValidationError
from hscredit.report.model_report import ModelReport, auto_model_report


class _TrackingModel:
    """记录每个公共预测入口的真实调用次数。"""

    feature_names_in_ = np.array(["x"])
    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls = []

    def predict_proba(self, X):
        self.calls.append("predict_proba")
        positive = np.array([0.1, 0.2, 0.3])[: len(X)]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        self.calls.append("predict")
        return np.array([1.0, 0.0, 1.0])[: len(X)]

    def predict_score(self, X):
        self.calls.append("predict_score")
        return np.array([610.0, 620.0, 630.0])[: len(X)]

    def transform(self, X):
        self.calls.append("transform")
        return np.array([11.0, 12.0, 13.0])[: len(X)]


@pytest.fixture
def report_data():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0]}), pd.Series([0, 1, 0], name="target")


@pytest.mark.parametrize(
    ("method", "expected_method", "expected"),
    [
        ("predict_proba", "predict_proba", [0.1, 0.2, 0.3]),
        ("predict_prob", "predict_proba", [0.1, 0.2, 0.3]),
        ("predict", "predict", [1.0, 0.0, 1.0]),
        ("predict_score", "predict_score", [610.0, 620.0, 630.0]),
        ("transform", "transform", [11.0, 12.0, 13.0]),
    ],
)
def test_auto_model_report_uses_only_selected_method(method, expected_method, expected, report_data):
    """任何额外概率或评分调用都会改变用户选择的方法语义。"""
    X, y = report_data
    model = _TrackingModel()

    report = auto_model_report(model, X_train=X, y_train=y, method=method, verbose=False, n_jobs=1)
    dataset = report._datasets["训练集"]

    assert isinstance(report, ModelReport)
    assert report.model is model
    assert report.method == method
    assert model.calls == [expected_method]
    np.testing.assert_allclose(dataset.prediction, expected)
    assert dataset.y_proba is dataset.prediction
    assert dataset.score is dataset.prediction


def test_default_method_is_predict_proba(report_data):
    """未指定 method 时只能调用一次 predict_proba。"""
    X, y = report_data
    model = _TrackingModel()

    report = auto_model_report(model, X_train=X, y_train=y, verbose=False, n_jobs=1)

    assert report.method == "predict_proba"
    assert model.calls == ["predict_proba"]


def test_callable_receives_real_report_and_named_parameters_once(report_data):
    """worker 副本或参数探针会破坏 self 身份与严格单次执行。"""
    X, y = report_data
    model = _TrackingModel()
    seen = []

    def scorer(self, x, scale, n_jobs, offset):
        seen.append(self)
        self.custom_marker = "kept"
        return np.arange(len(x), dtype=float) * scale + n_jobs + offset

    report = auto_model_report(
        model,
        X_train=X,
        y_train=y,
        method=scorer,
        method_kwargs={"scale": 10.0},
        offset=2.0,
        n_jobs=1,
        verbose=False,
    )

    assert seen == [report]
    assert report.custom_marker == "kept"
    assert report.kwargs == {"offset": 2.0}
    np.testing.assert_allclose(report._datasets["训练集"].prediction, [3.0, 13.0, 23.0])
    assert model.calls == []


def test_callable_copy_and_pickle_restore_executable_method(report_data):
    """lambda 必须按 callable 载荷恢复，不能依赖不可执行的源码字符串。"""
    X, y = report_data
    factor = 7.0
    report = auto_model_report(
        _TrackingModel(),
        X_train=X,
        y_train=y,
        method=lambda self, x: np.arange(len(x), dtype=float) * factor,
        verbose=False,
        n_jobs=1,
    )

    copied = copy.deepcopy(report)
    restored = pickle.loads(pickle.dumps(report))

    assert callable(copied.method)
    assert callable(restored.method)
    np.testing.assert_allclose(copied.method(copied, X), [0.0, 7.0, 14.0])
    np.testing.assert_allclose(restored.method(restored, X), [0.0, 7.0, 14.0])


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("missing", "method"),
        (lambda self, x, required: np.zeros(len(x)), "required"),
        (lambda self, x: np.zeros((len(x), 2)), "一维"),
        (lambda self, x: np.zeros(len(x) - 1), "长度"),
        (lambda self, x: np.full(len(x), np.nan), "有限"),
    ],
)
def test_method_validation_is_chinese_and_transactional(method, message, report_data):
    """无效结果不能提交半成品数据集。"""
    X, y = report_data

    with pytest.raises((ValidationError, SerializationError), match=message):
        ModelReport(_TrackingModel(), X_train=X, y_train=y, method=method, n_jobs=1)
