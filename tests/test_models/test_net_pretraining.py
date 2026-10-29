"""X-only 掩码预训练的权重、字段、损失与监督迁移契约。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from hscredit import ResNet, FTTransformer, TabM, TabPFN
from hscredit.core.models.net._pretraining import random_mask
from .test_net_models import FakeClassifier, FakeRegressor


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(before)


@pytest.fixture
def frame():
    r = np.random.default_rng(6)
    X = pd.DataFrame(r.normal(size=(64, 2)), columns=["分数", "次数"])
    X["金额"] = X["分数"] * 10 + 100
    X["类别"] = np.where(X["次数"] > 0, "甲", "乙")
    X.loc[3, "分数"] = np.nan
    X["全空"] = np.nan
    return X


def model(cls, **kwargs):
    params = dict(
        d_block=16,
        n_blocks=1,
        max_epochs=2,
        batch_size=16,
        early_stopping_rounds=None,
        device="cpu",
        random_state=3,
        dropout=0,
        target="FPD",
    )
    if cls is TabM:
        params["k"] = 3
    return cls(**{**params, **kwargs})


@pytest.mark.parametrize("cls", [ResNet, FTTransformer, TabM])
def test_pretrain_updates_encoder_without_labels_and_reuses_weights(cls, frame, runtime, monkeypatch, tmp_path):
    m = model(cls)
    initial = {}
    original = m._initialize_network

    def initialize():
        original()
        initial.update({k: v.detach().cpu().clone() for k, v in m._model.state_dict().items()})

    monkeypatch.setattr(m, "_initialize_network", initialize)
    m.pretrain(frame.assign(FPD=np.arange(len(frame))))
    assert not m._is_fitted and "FPD" not in m.feature_names_in_
    parameter_names = set(dict(m._model.named_parameters()))
    assert any(
        not runtime.equal(v, initial[k])
        for k, v in m._pretrained_state_.weights.items()
        if k in parameter_names and not k.startswith("head.")
    )
    assert m.get_embeddings(frame[:4]).shape == (4, 16)
    with pytest.raises(NotFittedError):
        m.predict(frame)
    same = model(cls).pretrain(frame.assign(FPD="完全不同的标签"))
    assert_allclose(same.transform(frame), m.transform(frame))
    # 固定的预训练字段处理不会被监督数据重新拟合；来源模型保持不变。
    source_embedding = same.transform(frame).copy()
    receiver = model(cls, objective="regression")
    received = {}

    def train(tensors, eval_set):
        received.update({k: v.detach().cpu().clone() for k, v in receiver._model.state_dict().items()})

    monkeypatch.setattr(receiver, "_train", train)
    receiver.fit(frame.assign(分数=frame["分数"] + 100), frame["金额"], from_pretrained=same)
    for key, value in same._pretrained_state_.weights.items():
        assert runtime.equal(value, received[key])
    assert_allclose(receiver.preprocessor_.medians_, same.preprocessor_.medians_)
    assert_allclose(same.transform(frame), source_embedding)
    same.save(tmp_path / "预训练.pkl")
    restored = cls.load(tmp_path / "预训练.pkl")
    assert_allclose(restored.transform(frame), source_embedding)
    restored.fit(frame, (frame["次数"] > 0).astype(int))
    assert restored.predict_proba(frame).shape == (len(frame), 2)
    assert not hasattr(clone(same), "_pretrained_state_")


def test_masked_mse_matches_independent_calculation(runtime, monkeypatch):
    X = pd.DataFrame({"甲": [1.0, 2.0, 4.0, 8.0], "乙": [2.0, 3.0, 6.0, 7.0]})
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(runtime.optim.AdamW, "step", lambda self: None)
    m = model(FTTransformer, max_epochs=1, batch_size=4).pretrain(X, pretraining_ratio=1, sample_weight=weights)
    numeric, categorical = m.preprocessor_.transform(X)
    with runtime.no_grad():
        hidden = m._model.embeddings(runtime.zeros_like(runtime.tensor(numeric)), runtime.tensor(categorical))
        prediction = m._pretrain_decoder_["numeric"](hidden).numpy()
    expected = np.average(((prediction - numeric) ** 2).mean(axis=1), weights=weights)
    assert_allclose(m.evals_result_["train"]["重建损失"], [expected], rtol=1e-6)
    observed = np.array([[True, False, True], [False, False, False], [True, True, True]])
    mask = random_mask(observed, 0.01, np.random.default_rng(2))
    assert not (mask & ~observed).any()
    assert (mask.sum(axis=1)[[0, 2]] >= 1).all()


def test_pretrain_no_validation_leak_and_incompatible_transfer(frame, runtime):
    m = model(ResNet, early_stopping_rounds=2).pretrain(frame)
    expected = frame.iloc[m._eval_train_indices_]["分数"].median()
    assert_allclose(m.preprocessor_.medians_[0], expected)
    with pytest.raises(ValueError, match="类型"):
        model(FTTransformer).fit(frame, (frame["次数"] > 0).astype(int), from_pretrained=m)
    with pytest.raises(ValueError, match="结构"):
        model(ResNet, d_block=32).fit(frame, (frame["次数"] > 0).astype(int), from_pretrained=m)
    with pytest.raises(ValueError, match="pretrain"):
        model(ResNet).fit(frame, (frame["次数"] > 0).astype(int), from_pretrained=True)
    for ratio in [0, -1, 1.1, np.nan, True, "错误"]:
        with pytest.raises(ValueError, match="pretraining_ratio"):
            model(ResNet).pretrain(frame, pretraining_ratio=ratio)


@pytest.fixture
def pfn_backend(monkeypatch):
    import hscredit.core.models.net._pfn_training as training
    import hscredit.core.models.net.tabpfn as wrapper

    calls = []

    class FineClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _forward_with_loss(self, batch):
            return None

        def fit(self, X, y, **kwargs):
            calls.append((self.kwargs, X.copy(), np.asarray(y).copy(), kwargs))
            self.kwargs["experiment_logger"].log_epoch({"train/epoch": 0, "train/mean_loss": 0.25}, 1)
            self.finetuned_inference_classifier_ = FakeClassifier().fit(X, y)
            self.finetuned_inference_regressor_ = FakeRegressor().fit(X, y)
            return self

    monkeypatch.setattr(training, "load_finetuning", lambda: (FineClassifier, FineClassifier, object))
    monkeypatch.setattr(training, "model_spec", lambda estimator: SimpleNamespace(revision=len(calls)))
    monkeypatch.setattr(wrapper, "_load_tabpfn", lambda: (FakeClassifier, FakeRegressor))
    return calls


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_pfn_finetune_switch_and_config_overrides(objective, frame, pfn_backend):
    y = (frame["次数"] > 0).astype(int) if objective == "binary" else frame["金额"]
    config = {"max_epochs": 2, "learning_rate": 0.001, "early_stopping_rounds": 3}
    m = TabPFN(objective=objective, finetune=True, finetune_params=config, n_estimators=2)
    m.fit(frame, y, finetune_params={"max_epochs": 4}, eval_set={"验证": (frame[:9], y[:9])})
    options = pfn_backend[-1][0]
    assert options["epochs"] == 4 and options["learning_rate"] == 0.001
    assert options["early_stopping_patience"] == 3
    assert options["n_estimators_final_inference"] == 2
    assert (
        str(
            options["extra_classifier_kwargs" if objective == "binary" else "extra_regressor_kwargs"][
                "inference_precision"
            ]
        )
        == "torch.float32"
    )
    assert pfn_backend[-1][3]["X_val"].equals(frame[:9])
    assert config["max_epochs"] == 2
    assert m.finetune_ and m.n_iter_ == 1
    before = len(pfn_backend)
    m.fit(frame, y, finetune=False)
    assert not m.finetune_ and len(pfn_backend) == before
    assert m.predict(frame[:3]).shape == (3,)
    assert clone(m).get_params()["finetune_params__max_epochs"] == 2


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_pfn_x_only_pretraining_masks_targets_and_chains_weights(objective, frame, pfn_backend):
    m = TabPFN(objective=objective, target="FPD", finetune_params={"validation_fraction": 0}, random_state=2)
    m.pretrain(frame.assign(FPD="不得使用"), max_epochs=1, batch_size=32)
    assert len(pfn_backend) > 1 and "FPD" not in m.feature_names_in_
    assert (
        "model_path"
        in pfn_backend[1][0]["extra_classifier_kwargs" if objective == "binary" else "extra_regressor_kwargs"]
    )
    for _, X, y, _ in pfn_backend:
        assert "FPD" not in X and X.isna().all(axis=0).any()
        assert len(y) == len(X) and np.isfinite(y).all()
    with pytest.raises(NotFittedError):
        m.predict(frame)
    y = (frame["次数"] > 0).astype(int) if objective == "binary" else frame["金额"]
    m.fit(frame, y)
    assert isinstance(m.native_params_["model_path"], SimpleNamespace)
    assert m.predict(frame[:3]).shape == (3,)


def test_pfn_finetune_weight_guard(frame, pfn_backend):
    with pytest.raises(ValueError, match="sample_weight"):
        TabPFN(finetune=True).fit(frame, (frame["次数"] > 0).astype(int), sample_weight=np.ones(len(frame)))


def test_pfn_pretrain_rejects_duplicated_validation_aliases(frame, pfn_backend):
    with pytest.raises(ValueError, match="不能同时"):
        TabPFN(finetune_params={"validation_fraction": 0.1, "validation_split_ratio": 0.2}).pretrain(frame)


def test_pfn_optional_version_gate(monkeypatch):
    from hscredit.exceptions import DependencyError
    import hscredit.core.models.net._pfn_training as training

    monkeypatch.setattr(training, "version", lambda name: "2.0.9")
    with pytest.raises(DependencyError, match="9.2"):
        training.load_finetuning()


@pytest.mark.parametrize("pipeline", [False, True])
def test_pfn_tuner_full_refit_disables_internal_holdout(frame, pfn_backend, pipeline):
    from hscredit import ModelTuner
    from sklearn.pipeline import Pipeline

    y = (frame["次数"] > 0).astype(int)
    estimator = TabPFN(finetune=True, finetune_params={"max_epochs": 1, "validation_fraction": 0.2})
    tuner = ModelTuner(
        Pipeline([("模型", estimator)]) if pipeline else estimator,
        search_space={},
        n_jobs=1,
        fit_params={"模型__finetune" if pipeline else "finetune": None},
    )
    tuner.fit(
        frame.iloc[:48], y.iloc[:48], eval_set=(frame.iloc[48:], y.iloc[48:]), n_trials=1, show_progress_bar=False
    )
    tuner.get_best_model()
    options, X, _, fit_options = pfn_backend[-1]
    assert options["validation_split_ratio"] == 0 and options["early_stopping"] is False
    assert len(X) == 48 and not fit_options
