"""net 模型的训练、表示、字段、任务及制品契约。"""

import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from hscredit import ResNet, FTTransformer, TabPFN


@pytest.fixture
def frame():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(65, 3)), columns=["分数", "次数", "金额"])
    X["类别"] = np.where(X["次数"] > 0, "甲", "乙")
    X.loc[2, "分数"] = np.nan
    X["全空"] = np.nan
    return X, (X["金额"] > 0).astype(int).to_numpy()


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rtdl_revisiting_models")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


def small(cls, **kwargs):
    return cls(
        **{
            "d_block": 16,
            "n_blocks": 1,
            "batch_size": 16,
            "max_epochs": 3,
            "early_stopping_rounds": None,
            "device": "cpu",
            "random_state": 7,
            **kwargs,
        }
    )


def test_net_import_is_lazy_without_optional_dependencies():
    code = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tabpfn', 'tabm', 'rtdl_num_embeddings', 'rtdl_revisiting_models'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
import hscredit
from hscredit.core import models
from hscredit.core.models import net
for name in ['ResNet', 'FTTransformer', 'TabM', 'TabPFN']:
    assert getattr(hscredit, name) is getattr(models, name) is getattr(net, name)
    getattr(net, name)()
assert not any(name in sys.modules for name in ['torch', 'tabpfn', 'tabm', 'rtdl_num_embeddings', 'rtdl_revisiting_models'])
from hscredit.exceptions import DependencyError
for cls in [net.ResNet, net.FTTransformer, net.TabM, net.TabPFN]:
    try:
        cls().fit([[1], [2], [3], [4], [5], [6], [7], [8]], [0, 1] * 4)
    except DependencyError:
        pass
    else:
        raise AssertionError(cls)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_torch_outputs_are_the_trained_head_and_roundtrip(cls, objective, frame, torch_runtime, tmp_path):
    X, y = frame
    target = y if objective == "binary" else X["金额"].to_numpy() * 123 + 5000
    model = small(cls, objective=objective, target="目标")
    assert is_regressor(model) == (objective == "regression")
    assert is_classifier(model) == (objective == "binary")
    assert clone(model).get_params() == model.get_params()
    model.fit(X.assign(目标=-99), target, eval_set={"验证": (X[:9], target[:9])})
    query = X.iloc[:7].copy()
    query.loc[0, "类别"] = "未知"
    embeddings = model.get_embeddings(query)
    assert embeddings.shape == (7, 16)
    assert_allclose(embeddings, model.transform(query))
    assert_allclose(embeddings, model.transform(query[query.columns[::-1]].assign(目标=100)))
    native = model.get_native_model()
    with torch_runtime.no_grad():
        raw = native.head(torch_runtime.tensor(embeddings)).numpy().ravel()
    if objective == "binary":
        probability = model.predict_proba(query)
        assert_allclose(probability.sum(axis=1), 1)
        assert_allclose(probability[:, 1], expit(raw), atol=1e-7)
        assert_array_equal(model.predict(query), (probability[:, 1] >= 0.5).astype(int))
        assert np.isfinite(model.predict_score(query)).all()
    else:
        assert not hasattr(model, "classes_")
        prediction = model.predict(query)
        assert_allclose(prediction, raw.astype(float) * model.target_scale_ + model.target_mean_)
        metric = model.evaluate(X, target, metrics="rmse")
        assert_allclose(metric["均方根误差"], np.sqrt(np.mean((target - model.predict(X)) ** 2)))
        with pytest.raises(ValueError, match="回归模型"):
            model.predict_proba(query)
    assert list(model.get_eval_results().columns) == ["数据集", "指标", "数值"]
    assert len(model.get_learning_curve()) == 6
    assert model.training_summary_["状态"] == "完成"
    model.save(tmp_path / "模型.pkl")
    restored = cls.load(tmp_path / "模型.pkl")
    assert_allclose(restored.predict(query), model.predict(query))
    assert_allclose(restored.transform(query), embeddings)


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_preprocessing_excludes_holdout_and_zero_weight_rows(cls, frame, torch_runtime):
    X, y = frame
    model = small(cls, early_stopping_rounds=2, validation_fraction=0.3).fit(X, y)
    positions = model._eval_train_indices_
    assert_allclose(model.preprocessor_.medians_[0], X.iloc[positions]["分数"].median())
    assert not set(positions) & set(model._eval_val_indices_)
    contaminated = X.copy()
    contaminated.loc[64, "分数"] = 1e6
    weights = np.ones(len(y))
    weights[-1] = 0
    weighted = small(cls).fit(contaminated, y, sample_weight=weights)
    reduced = small(cls).fit(X.iloc[:-1], y[:-1])
    assert_allclose(weighted.predict_proba(X.iloc[:5]), reduced.predict_proba(X.iloc[:5]))


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_torch_input_edges_and_failed_refit(cls, frame, torch_runtime):
    X, y = frame
    model = small(cls)
    with pytest.raises(NotFittedError):
        model.get_embeddings(X)
    model.fit(X, y)
    assert model.get_embeddings(X.iloc[:0]).shape == (0, 16)
    assert_allclose(model.predict_proba(X.iloc[:1]), model.predict_proba(X.iloc[:9])[:1], atol=1e-6)
    with pytest.raises(ValueError, match="缺少训练字段"):
        model.predict(X.drop(columns="分数"))
    with pytest.raises(ValueError, match="重复"):
        model.predict(pd.concat([X, X[["次数"]]], axis=1))
    with pytest.raises(ValueError, match="无穷"):
        model.predict(X.assign(分数=np.inf))
    with pytest.raises(ValueError):
        model.fit(X, np.ones(len(y)))
    with pytest.raises(NotFittedError):
        model.predict(X)
    assert model.training_summary_["状态"] == "失败"
    model.set_params(objective="regression").fit(X, X["金额"].to_numpy())
    assert not hasattr(model, "classes_")
    assert not hasattr(model, "scorecard_")


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_numeric_and_categorical_only_inputs(cls, frame, torch_runtime):
    X, y = frame
    for data in [X[["类别"]], X[["次数", "金额"]].to_numpy()]:
        model = small(cls).fit(data, y)
        assert model.transform(data).shape == (len(X), 16)
        assert np.isfinite(model.predict_proba(data)).all()


def test_early_stopping_restores_best_epoch_and_preserves_rng(frame, torch_runtime, monkeypatch):
    X, y = frame
    reference = small(ResNet, max_epochs=1).fit(X, y)
    model = small(ResNet, max_epochs=8, early_stopping_rounds=2)
    original = model._infer_arrays
    calls = []

    def increasing_validation(numeric, categorical, *, embeddings=False):
        if len(numeric) == 8 and not embeddings:
            calls.append(len(calls))
            return np.full(8, calls[-1], dtype=np.float32)
        return original(numeric, categorical, embeddings=embeddings)

    monkeypatch.setattr(model, "_infer_arrays", increasing_validation)
    state = torch_runtime.random.get_rng_state().clone()
    model.fit(X, y, eval_set=[(X[:8], np.tile([0, 1], 4))])
    assert torch_runtime.equal(state, torch_runtime.random.get_rng_state())
    assert model.n_iter_ == 3
    assert model.best_iteration_ == 1
    for key, value in model._model.state_dict().items():
        assert_allclose(value.numpy(), reference._model.state_dict()[key].numpy())


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_models_learn_signal_and_work_in_sklearn(cls, torch_runtime):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(96, 2))
    y = (X[:, 0] > 0).astype(int)
    model = small(cls, max_epochs=25, learning_rate=0.02, batch_size=32, dropout=0).fit(X, y)
    assert model.score(X, y) > 0.9
    assert model.evals_result_["train"]["损失"][-1] < model.evals_result_["train"]["损失"][0]
    regression = small(cls, objective="regression", max_epochs=25, learning_rate=0.02, batch_size=32, dropout=0)
    regression.fit(X, X[:, 0] * 12 + 1000)
    assert regression.score(X, X[:, 0] * 12 + 1000) > 0.75
    pipe = Pipeline([("表示", small(cls, max_epochs=1)), ("分类", LogisticRegression())])
    assert np.isfinite(cross_val_score(pipe, X, y, cv=2)).all()


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_weighted_loss_matches_independent_formula(cls, frame, torch_runtime, monkeypatch):
    X, y = frame
    weight = np.linspace(1, 5, len(y))
    monkeypatch.setattr(torch_runtime.optim.AdamW, "step", lambda self: None)
    model = small(cls, max_epochs=1, batch_size=128, dropout=0)
    model.fit(X, y, sample_weight=weight, eval_set={"train": (X, y), "训练集": (X, y)})
    model._model.train()
    numeric, categorical = model.preprocessor_.transform(X)
    with torch_runtime.no_grad():
        raw = model._model(torch_runtime.tensor(numeric), torch_runtime.tensor(categorical)).numpy().astype(float)
    expected = np.average(np.logaddexp(0, raw) - y * raw, weights=weight)
    assert_allclose(model.evals_result_["train"]["损失"], [expected], atol=1e-6)
    assert set(model.evals_result_) == {"train", "validation_0", "validation_1"}
    assert len(model.get_learning_curve()) == 3


class FakeClassifier:
    """只用于检验适配边界；真实权重验收在 net_models.ipynb 执行。"""

    def __init__(self, **kwargs):
        self.options = kwargs

    def fit(self, X, y):
        self.seen = X.copy()
        self.labels = np.asarray(y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = expit(np.asarray(X["金额"]))
        return np.column_stack((1 - p, p))

    def get_embeddings(self, X, data_source):
        assert data_source == "test"
        values = np.asarray(X[["次数", "金额"]])
        result = np.stack([values, values + 2])
        return result[:, 0, :] if len(X) == 1 else result


class FakeRegressor(FakeClassifier):
    def predict(self, X):
        return np.asarray(X["金额"]) * 5 + 100


@pytest.fixture
def fake_tabpfn(monkeypatch):
    module = importlib.import_module("hscredit.core.models.net.tabpfn")
    monkeypatch.setattr(module, "_load_tabpfn", lambda: (FakeClassifier, FakeRegressor))


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_tabpfn_task_dispatch_embeddings_and_native_options(objective, frame, fake_tabpfn, tmp_path):
    X, y = frame
    target = y if objective == "binary" else X["金额"] * 5 + 100
    model = TabPFN(objective=objective, target="目标", n_estimators=2, fit_mode="low_memory")
    assert clone(model).get_params() == model.get_params()
    model.set_params(fit_mode="fit_preprocessors")
    model.fit(X.assign(目标=-99), target, eval_set={"测试": (X[:9], target[:9])})
    assert (type(model.get_native_model()) is FakeRegressor) == (objective == "regression")
    assert "目标" not in model._model.seen.columns
    assert model._model.options["fit_mode"] == "fit_preprocessors"
    assert_allclose(model._model.labels, target)
    for query in [X[:1], X[:7], X.iloc[[9, 2, 4]]]:
        result = model.get_embeddings(query)
        assert result.shape == (len(query), 2)
        assert_allclose(result, np.asarray(query[["次数", "金额"]]) + 1)
        assert_allclose(result, model.get_embeddings(query, aggregate=None).mean(axis=0))
    assert_allclose(model.transform(X), model.transform(X[X.columns[::-1]]))
    assert model.get_learning_curve().empty
    if objective == "regression":
        assert model.score(X, target) == 1
        assert not is_classifier(model)
        with pytest.raises(ValueError, match="回归模型"):
            model.predict_proba(X)
    else:
        assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    model.save(tmp_path / "tabpfn.pkl")
    assert_allclose(TabPFN.load(tmp_path / "tabpfn.pkl").transform(X), model.transform(X))


def test_tabpfn_rejects_unsupported_weights_and_bad_embeddings(frame, fake_tabpfn, monkeypatch):
    X, y = frame
    with pytest.raises(ValueError, match="不支持 sample_weight"):
        TabPFN().fit(X, y, sample_weight=np.ones(len(y)))
    model = TabPFN().fit(X, y)
    with pytest.raises(ValueError, match="aggregate"):
        model.get_embeddings(X, aggregate="flatten")
    monkeypatch.setattr(model._model, "get_embeddings", lambda X, data_source: np.zeros((2, len(X) + 1, 8)))
    with pytest.raises(ValueError, match="embedding"):
        model.get_embeddings(X)


def test_tabpfn_new_preprocessing_parameter_replaces_deprecated_jobs(frame, monkeypatch):
    class NewClassifier(FakeClassifier):
        def __init__(self, *, n_preprocessing_jobs=1, **kwargs):
            super().__init__(n_preprocessing_jobs=n_preprocessing_jobs, **kwargs)

    module = importlib.import_module("hscredit.core.models.net.tabpfn")
    monkeypatch.setattr(module, "_load_tabpfn", lambda: (NewClassifier, FakeRegressor))
    X, y = frame
    model = TabPFN(n_jobs=2).fit(X, y)
    assert model.get_native_params()["n_preprocessing_jobs"] == 2
    assert "n_jobs" not in model.get_native_params()
    explicit = TabPFN(n_jobs=2, n_preprocessing_jobs=3).fit(X, y)
    assert explicit.get_native_params()["n_preprocessing_jobs"] == 3


@pytest.mark.parametrize("cls", [ResNet, FTTransformer, TabPFN])
def test_invalid_objective_is_explicit(cls):
    with pytest.raises(ValueError, match="objective"):
        cls(objective="multiclass").fit([[1], [2]], [0, 1])


@pytest.mark.parametrize("cls", [ResNet, FTTransformer])
def test_torch_rejects_misspelled_parameters(cls):
    with pytest.raises(ValueError, match="未知模型参数"):
        cls().set_params(learning_rata=0.01)
