"""TabNet 决策表示、注意力、稀疏损失及完整 net 工作流契约。"""

import copy

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from hscredit import TabNet, ModelTuner, AutoTuner


@pytest.fixture(autouse=True)
def runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_tabnet")
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(before)


@pytest.fixture
def data():
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(65, 3)), columns=["甲", "乙", "丙"])
    X.insert(1, "类别", np.where(X["乙"] > 0, "A", "B"))
    X.loc[3, "甲"] = np.nan
    X["全空"] = np.nan
    return X, (X["丙"] > 0).astype(int).to_numpy()


def small(**kwargs):
    return TabNet(
        **{
            "n_d": 8,
            "n_a": 4,
            "n_steps": 2,
            "cat_embedding_dim": 3,
            "batch_size": 16,
            "max_epochs": 2,
            "early_stopping_rounds": None,
            "device": "cpu",
            "random_state": 7,
            **kwargs,
        }
    )


@pytest.mark.parametrize("objective", ["binary", "regression"])
@pytest.mark.parametrize("mask_type", ["sparsemax", "entmax"])
def test_head_attention_serialization_and_field_contract(objective, mask_type, data, runtime, tmp_path):
    X, y = data
    target = y if objective == "binary" else X["丙"].to_numpy() * 100 + 5000
    weight = np.linspace(1, 3, len(X))
    model = small(objective=objective, mask_type=mask_type, target="目标")
    assert is_classifier(model) == (objective == "binary")
    assert is_regressor(model) == (objective == "regression")
    assert clone(model).get_params() == model.get_params()
    model.fit(X.assign(目标=-999), target, sample_weight=weight, eval_set={"验证": (X[:8], target[:8])})
    hidden = model.transform(X)
    native = model.get_native_model()
    assert hidden.shape == (len(X), 8)
    assert_allclose(hidden, model.get_embeddings(X[X.columns[::-1]]))
    assert_allclose(model.get_embeddings(X, aggregate=None)[0], hidden)
    assert model.transform(X[:0]).shape == (0, 8)
    # float32 的单行/批量矩阵乘法存在舍入差异，按机器精度核对。
    assert_allclose(model.transform(X[:1]), hidden[:1], rtol=32 * np.finfo(np.float32).eps, atol=1e-6)
    numeric, categorical = model.preprocessor_.transform(X)
    with runtime.no_grad():
        raw = native.head(runtime.tensor(hidden)).numpy().ravel()
        # 直接核对官方各决策步，而不是调用包装层 embeddings 作为参照。
        joined = runtime.tensor(np.column_stack((numeric, categorical)), dtype=runtime.float32)
        embedded = native.backbone.embedder(joined)
        steps, _ = native.backbone.tabnet.encoder(embedded)
        assert_allclose(hidden, runtime.stack(steps).sum(dim=0).numpy(), atol=1e-6)
        masks, _ = native.backbone.forward_masks(joined)
    # 类别列位于原始字段中间；手工汇总 embedding 各维并恢复原顺序。
    weighted = (masks.numpy() * weight[:, None]).sum(axis=0)
    expected = np.array([weighted[0], weighted[4:].sum(), weighted[1], weighted[2], weighted[3]])
    expected /= expected.sum()
    assert_allclose(model.feature_importances_, expected, rtol=1e-6)
    assert_allclose(model.get_feature_importances().reindex(X.columns), expected, rtol=1e-6)
    if objective == "binary":
        assert_allclose(model.predict_proba(X)[:, 1], expit(raw), atol=1e-7)
        assert_allclose(model.predict(X), (expit(raw) >= 0.5).astype(int))
        assert np.isfinite(model.predict_score(X)).all()
    else:
        assert_allclose(model.predict(X), raw.astype(float) * model.target_scale_ + model.target_mean_)
        with pytest.raises(ValueError, match="回归"):
            model.predict_proba(X)
    model.save(tmp_path / "TabNet.pkl")
    restored = TabNet.load(tmp_path / "TabNet.pkl")
    assert_allclose(restored.transform(X), hidden)
    assert_allclose(restored.predict(X), model.predict(X))
    assert_allclose(copy.deepcopy(model).feature_importances_, expected, rtol=1e-6)
    assert len(model.get_learning_curve()) == 4
    with pytest.raises(ValueError, match="缺少训练字段"):
        model.predict(X.drop(columns="甲"))
    if objective == "binary":
        with pytest.raises(ValueError):
            model.fit(X, np.ones(len(X)))
        with pytest.raises(NotFittedError):
            model.predict(X)


def test_sparse_loss_matches_weighted_bce_and_entropy(data, runtime, monkeypatch):
    X, y = data
    weight = np.linspace(1, 3, len(X))
    monkeypatch.setattr(runtime.optim.AdamW, "step", lambda self: None)
    model = small(max_epochs=1, batch_size=128, lambda_sparse=0.2).fit(X, y, sample_weight=weight)
    model._model.train()
    n, c = model.preprocessor_.transform(X)
    with runtime.no_grad():
        raw = model._model(runtime.tensor(n), runtime.tensor(c)).numpy().astype(float)
    expected = np.average(np.logaddexp(0, raw) - y * raw, weights=weight)
    expected -= 0.2 * float(model._model.mask_loss)
    assert_allclose(model.evals_result_["train"]["损失"], [expected], rtol=1e-6)


def test_no_validation_or_zero_weight_leak_and_small_batches(data):
    X, y = data
    model = small(early_stopping_rounds=1).fit(X, y)
    assert_allclose(model.preprocessor_.medians_[0], X.iloc[model._eval_train_indices_]["甲"].median())
    weighted = small().fit(X, y, sample_weight=np.r_[np.ones(64), 0])
    reduced = small().fit(X.iloc[:64], y[:64])
    assert_allclose(weighted.predict_proba(X), reduced.predict_proba(X))
    assert_allclose(weighted.feature_importances_, reduced.feature_importances_)
    for frame in [X[["类别"]], X[["乙", "丙"]].to_numpy()]:
        m = small(batch_size=5, virtual_batch_size=2).pretrain(frame)
        m.fit(frame, y)
        assert np.isfinite(m.predict_proba(frame)).all()


@pytest.mark.parametrize(
    "option",
    [
        {"n_d": 0},
        {"n_a": True},
        {"n_steps": -1},
        {"batch_size": 1},
        {"virtual_batch_size": 1},
        {"n_independent": 0, "n_shared": 0},
        {"gamma": 0.9},
        {"lambda_sparse": -1},
        {"momentum": 1},
        {"epsilon": np.nan},
        {"mask_type": "other"},
    ],
)
def test_invalid_options_fail_in_chinese(option, data):
    X, y = data
    with pytest.raises(ValueError, match="必须|不能"):
        small(**option).fit(X, y)


@pytest.mark.parametrize("objective", ["binary", "regression"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_tune_custom_loss_optimizer_and_full_refit(objective, pipeline, data):
    X, y = data
    target = y if objective == "binary" else X["丙"].to_numpy() * 100 + 5000
    model = small(objective=objective, optimizer="sgd", optimizer_params={"momentum": 0.1})
    model.set_params(optimizer_params__momentum=0.2)
    estimator = Pipeline([("模型", model)]) if pipeline else model
    prefix = "模型__" if pipeline else ""
    loss = "bce" if objective == "binary" else "huber"
    estimator.set_params(**{prefix + "loss": loss})
    tuner = ModelTuner(
        estimator,
        search_space={prefix + "learning_rate": [0.01, 0.02]},
        n_jobs=1,
        metric="auc" if objective == "binary" else "rmse",
    )
    tuner.fit(X[:48], target[:48], eval_set=(X[48:], target[48:]), n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    assert np.isfinite(best.predict(X)).all()
    final = best[-1] if pipeline else best
    assert final.optimizer_params["momentum"] == 0.2
    assert final.training_summary_["状态"] == "完成"
    assert_allclose(final.preprocessor_.medians_[0], X.iloc[:48]["甲"].median())


def test_default_search_space_and_auto_alias():
    from hscredit.core.models.tuning._net import net_search_space

    space = net_search_space(TabNet, {})
    assert {"n_d", "n_a", "n_steps", "lambda_sparse"} <= set(space)
    assert not {"d_block", "dropout", "n_blocks"} & set(space)
    assert AutoTuner.create("tabnet").model_class is TabNet


def test_pretrain_uses_official_network_loss_and_exact_encoder_transfer(data, runtime):
    from pytorch_tabnet.pretraining import TabNetPretrainer
    from pytorch_tabnet.metrics import UnsupervisedLoss

    X, _ = data
    X = X.iloc[:64]
    model = small(max_epochs=2, cat_embedding_dim=2).pretrain(X)
    backend = model.pretrainer_
    assert isinstance(backend, TabNetPretrainer)
    assert backend.loss_fn is UnsupervisedLoss
    # 相同配置直接运行官方预训练器，独立核对整个训练结果。
    params = backend.get_params()
    official = TabNetPretrainer(**params)
    numeric, categorical = model.preprocessor_.transform(X)
    transformed = np.column_stack((numeric, categorical)).astype(np.float32)
    official.fit(transformed, max_epochs=2, batch_size=16, virtual_batch_size=128, patience=0, drop_last=False)
    assert_allclose(backend.history["loss"], official.history["loss"], rtol=0, atol=0)
    for key, value in official.network.encoder.state_dict().items():
        assert runtime.equal(value, model._model.backbone.tabnet.encoder.state_dict()[key])
    for key, value in official.network.embedder.state_dict().items():
        assert runtime.equal(value, model._model.backbone.embedder.state_dict()[key])
    # 官方方差归一化的重建公式，独立计算而非再次调用其 loss。
    backend.network.train()
    output, embedded, flags = backend.network(runtime.tensor(transformed))
    array = embedded.detach().numpy()
    means = array.mean(axis=0)
    means[means == 0] = 1
    variance = array.var(axis=0, ddof=1)
    variance[variance == 0] = means[variance == 0]
    errors = ((output.detach().numpy() - array) * flags.detach().numpy()) ** 2
    expected = ((errors / variance).sum(axis=1) / (flags.detach().numpy().sum(axis=1) + 1e-9)).mean()
    assert_allclose(backend.compute_loss(output, embedded, flags).item(), expected, rtol=1e-6)


def test_official_pretraining_early_stopping_weights_and_failed_state(data, runtime, tmp_path):
    X, y = data
    model = small(max_epochs=3, early_stopping_rounds=1).pretrain(
        X.iloc[:48],
        eval_set={"首先": X.iloc[48:56], "其次": X.iloc[56:]},
        sample_weight=np.r_[0, np.ones(47)],
    )
    assert model.pretrainer_.early_stopping_metric == "validation_0_unsup_loss_numpy"
    assert model.eval_names_ == ["首先", "其次"]
    assert_allclose(model.preprocessor_.medians_[0], X.iloc[1:48]["甲"].median())
    assert model.best_iteration_ == model.pretrainer_.best_epoch + 1
    assert model.n_iter_ == len(model.pretrainer_.history["loss"])
    assert np.isfinite(model.get_learning_curve()["数值"]).all()
    model.save(tmp_path / "预训练.pkl")
    restored = TabNet.load(tmp_path / "预训练.pkl")
    assert_allclose(restored.transform(X), model.transform(X))
    with pytest.raises(ValueError, match="至少需要两行"):
        model.pretrain(X, eval_set=X[:1])
    with pytest.raises(NotFittedError):
        model.get_embeddings(X)
    restored.fit(X, y)
    assert restored.predict_proba(X).shape == (len(X), 2)


def test_tuner_accepts_official_pretrained_source_and_hashes_snapshot(data, runtime, tmp_path):
    X, y = data
    encoder = small(max_epochs=1).pretrain(X.iloc[:48])
    fingerprint = ModelTuner._configuration_fingerprint({"from_pretrained": encoder})
    encoder.save(tmp_path / "来源.pkl")
    restored = TabNet.load(tmp_path / "来源.pkl")
    assert ModelTuner._configuration_fingerprint({"from_pretrained": restored}) == fingerprint
    source_embedding = encoder.transform(X).copy()
    tuner = ModelTuner(
        small(max_epochs=1),
        search_space={"learning_rate": [0.01, 0.02]},
        n_jobs=1,
        fit_params={"from_pretrained": encoder},
    )
    tuner.fit(X.iloc[:48], y[:48], eval_set=(X.iloc[48:], y[48:]), n_trials=1, show_progress_bar=False)
    assert np.isfinite(tuner.get_best_model().predict_proba(X)).all()
    assert_allclose(encoder.transform(X), source_embedding)
    assert ModelTuner._configuration_fingerprint({"from_pretrained": encoder}) == fingerprint
    key = next(name for name, value in restored._pretrained_state_.weights.items() if value.is_floating_point())
    restored._pretrained_state_.weights[key] += 0.01
    assert ModelTuner._configuration_fingerprint({"from_pretrained": restored}) != fingerprint
