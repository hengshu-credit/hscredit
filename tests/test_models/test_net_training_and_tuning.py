"""TabM 成员训练数学契约、损失梯度、优化器配置及分类/回归调参。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit import (
    ResNet,
    FTTransformer,
    TabM,
    TabPFN,
    FocalLoss,
    WeightedBCELoss,
    RankingAUCProxyLoss,
    ModelTuner,
    AutoTuner,
)
from hscredit.core.models.net._training import loss_values


@pytest.fixture(autouse=True)
def torch_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabm")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


@pytest.fixture
def data():
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["分数", "次数", "金额"])
    X["商品"] = np.where(X["次数"] > 0, "甲", "乙")
    return X, (X["分数"] > 0).to_numpy(dtype=int), X["金额"].to_numpy() * 100 + 2000


def small(cls=TabM, **kwargs):
    options = {
        "d_block": 16,
        "n_blocks": 1,
        "max_epochs": 2,
        "batch_size": 32,
        "early_stopping_rounds": None,
        "random_state": 5,
        "device": "cpu",
    }
    if cls is TabM:
        options["k"] = 3
    return cls(**{**options, **kwargs})


@pytest.mark.parametrize("objective", ["binary", "regression"])
@pytest.mark.parametrize("embeddings", [None, "linear_relu", "periodic"])
@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini", "tabm-packed"])
def test_tabm_native_members_predict_and_serialize(objective, embeddings, arch_type, data, torch_runtime, tmp_path):
    X, y, amount = data
    labels = y if objective == "binary" else amount
    model = small(objective=objective, num_embeddings=embeddings, arch_type=arch_type).fit(
        X, labels, eval_set={"验证": (X, labels)}
    )
    assert clone(model).get_params(deep=False) == model.get_params(deep=False)
    members = model.get_embeddings(X[:7], aggregate=None)
    assert members.shape == (3, 7, 16)
    assert_allclose(model.get_embeddings(X[:7]), members.mean(axis=0))
    with torch_runtime.no_grad():
        raw = model.get_native_model().head(torch_runtime.tensor(members.transpose(1, 0, 2))).numpy().squeeze(-1)
    if objective == "binary":
        probability = expit(raw.astype(float)).mean(axis=1)
        assert_allclose(model.predict_proba(X[:7])[:, 1], probability, atol=1e-7)
        assert_array_equal(model.predict(X[:7]), (probability >= 0.5).astype(int))
        # 成员概率应在 sigmoid 后平均，而不是先平均 logits。
        assert np.max(np.abs(probability - expit(raw.astype(float).mean(axis=1)))) > 1e-8
        full_probability = model.predict_proba(X)[:, 1]
        expected = np.mean(-y * np.log(full_probability) - (1 - y) * np.log1p(-full_probability))
    else:
        expected_prediction = raw.astype(float).mean(axis=1) * model.target_scale_ + model.target_mean_
        assert_allclose(model.predict(X[:7]), expected_prediction, atol=1e-4)
        expected = np.mean(((labels - model.predict(X)) / model.target_scale_) ** 2)
    assert_allclose(model.evals_result_["validation_0"]["损失"][-1], expected, atol=1e-6)
    assert model.get_embeddings(X[:0], aggregate=None).shape == (3, 0, 16)
    model.save(tmp_path / "TabM.pkl")
    restored = TabM.load(tmp_path / "TabM.pkl")
    assert_allclose(restored.predict(X[:7]), model.predict(X[:7]))
    assert_allclose(restored.transform(X[:7]), model.transform(X[:7]))


def test_member_loss_and_gradient_match_hand_calculation(torch_runtime):
    torch = torch_runtime
    logits = torch.tensor([[-2.0, 3.0], [1.0, -4.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    model = SimpleNamespace(loss_fn_=torch.nn.BCEWithLogitsLoss(reduction="none"))
    per_row = loss_values(model, logits, target)
    raw = logits.detach().numpy()
    manual = (np.logaddexp(0, raw) - target.numpy()[:, None] * raw).mean(axis=1)
    assert_allclose(per_row.detach().numpy(), manual)
    loss = (per_row * weights).sum() / weights.sum()
    loss.backward()
    expected_gradient = (expit(raw) - target.numpy()[:, None]) * weights.numpy()[:, None] / weights.sum().item() / 2
    assert_allclose(logits.grad.numpy(), expected_gradient)
    ensemble_loss = np.logaddexp(0, raw.mean(axis=1)) - target.numpy() * raw.mean(axis=1)
    assert not np.allclose(manual, ensemble_loss)


@pytest.mark.parametrize("criterion", [FocalLoss(alpha=0.75, gamma=2), WeightedBCELoss(pos_weight=2, neg_weight=1)])
def test_baseloss_adapter_values_and_gradient_independent_formula(criterion, torch_runtime):
    torch = torch_runtime
    z = np.array([-2.0, -0.3, 0.8, 2.1])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.array([1.0, 2.0, 0.0, 4.0])

    def formula(logits):
        p = expit(logits)
        if isinstance(criterion, FocalLoss):
            return -0.75 * y * (1 - p) ** 2 * np.log(p) - 0.25 * (1 - y) * p**2 * np.log1p(-p)
        return -2 * y * np.log(p) - (1 - y) * np.log1p(-p)

    logits = torch.tensor(z, requires_grad=True)
    loss = criterion.to_torch()(logits, torch.tensor(y))
    assert_allclose(loss.detach().numpy(), formula(z))
    (loss * torch.tensor(weights)).mean().backward()
    epsilon = 1e-6
    expected = (formula(z + epsilon) - formula(z - epsilon)) / (2 * epsilon) * weights / len(y)
    assert_allclose(logits.grad.numpy(), expected, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="小批次"):
        RankingAUCProxyLoss().to_torch()


@pytest.mark.parametrize("cls", [ResNet, FTTransformer, TabM])
@pytest.mark.parametrize(
    "optimizer,options",
    [
        ("adamw", {"betas": (0.8, 0.95)}),
        ("adam", {"eps": 1e-7}),
        ("sgd", {"momentum": 0.9}),
        ("rmsprop", {"alpha": 0.95}),
        ("adagrad", {"eps": 1e-8}),
    ],
)
def test_optimizer_and_loss_configuration_are_used(cls, optimizer, options, data):
    X, y, _ = data
    model = small(cls, loss="focal", loss_params={"gamma": 1.5}, optimizer=optimizer, optimizer_params=options)
    model.fit(X, y)
    assert type(model.optimizer_).__name__.lower() == optimizer
    for key, value in options.items():
        assert model.optimizer_.defaults[key] == value
    assert model.optimizer_.defaults["lr"] == model.learning_rate
    assert model.optimizer_.defaults["weight_decay"] == model.weight_decay
    assert model.optimizer_.param_groups[0]["params"][0] is next(model._model.parameters())


def test_native_module_custom_callable_and_nested_params(data, torch_runtime, tmp_path):
    torch = torch_runtime
    X, y, amount = data
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    options = {"momentum": 0.8}
    model = small(ResNet, loss=criterion, optimizer=torch.optim.SGD, optimizer_params=options)
    model.set_params(optimizer_params__momentum=0.9).fit(X, y)
    assert criterion.reduction == "mean"
    assert options == {"momentum": 0.8}
    assert model.optimizer_.defaults["momentum"] == 0.9
    assert clone(model).get_params()["optimizer_params__momentum"] == 0.9

    def squared(prediction, target, scale):
        return scale * (prediction - target).square()

    model = small(loss=squared, loss_params={"scale": 2.0}, objective="regression").fit(X, amount)
    model.save(tmp_path / "自定义损失.pkl")
    restored = TabM.load(tmp_path / "自定义损失.pkl")
    assert_allclose(restored.predict(X[:3]), model.predict(X[:3]))
    model.set_params(loss=torch.nn.HuberLoss, loss_params={"delta": 0.5}).fit(X, amount)
    assert model.loss_fn_.delta == 0.5


@pytest.mark.parametrize(
    "options,match",
    [
        ({"optimizer_params": {"lr": 0.1}}, "learning_rate"),
        ({"optimizer": "missing"}, "optimizer"),
        ({"loss_params": {"reduction": "mean"}}, "reduction"),
        ({"loss": lambda p, y: (p - y).square().mean()}, "逐样本"),
        ({"loss": lambda p, y: (p - y).square().detach()}, "计算图"),
    ],
)
def test_invalid_training_configuration_is_rejected(options, match, data):
    X, y, _ = data
    with pytest.raises(ValueError, match=match):
        small(ResNet, **options).fit(X, y)


@pytest.mark.parametrize("cls", [ResNet, FTTransformer, TabM])
@pytest.mark.parametrize("task", ["binary", "regression"])
def test_tune_routes_configuration_cv_scores_and_full_refit(cls, task, data):
    X, y, amount = data
    labels = y if task == "binary" else amount
    model = small(cls, objective=task, early_stopping_rounds=1, max_epochs=3)
    best = model.tune(
        X,
        labels,
        n_trials=1,
        cv=2,
        n_jobs=1,
        show_progress_bar=False,
        loss=FocalLoss() if task == "binary" else "huber",
        search_space={"optimizer": ["sgd"], "optimizer_params__momentum": [0.9], "learning_rate": [0.01]},
    )
    tuner = best.tuner
    assert best.optimizer_.defaults["momentum"] == 0.9
    assert best.early_stopping_rounds is None and best.validation_fraction == 0
    assert best.training_summary_["样本数"] == len(X)
    assert_allclose(best.preprocessor_.medians_, X[["分数", "次数", "金额"]].median().to_numpy())
    folds = tuner.get_trial_result(0)["各折"]
    assert best.max_epochs == int(np.ceil(np.median([f["最佳迭代"] for f in folds])))
    table = tuner.get_oof_predictions()
    assert_array_equal(table["验证次数"], np.ones(len(X)))
    if task == "regression":
        assert "预测值" in table and "预测概率" not in table
        expected = [np.sqrt(np.mean((f["真实标签"] - f["预测值"]) ** 2)) for f in folds]
        assert_allclose(tuner.best_score_, np.mean(expected))
        assert not any("LIFT" in str(c) for c in tuner.get_optimization_history().columns)
    else:
        assert "预测概率" in table


def test_regression_holdout_weights_custom_metric_and_pipeline(data):
    X, _, amount = data
    X = X.drop(columns="商品")
    model = Pipeline([("缩放", StandardScaler()), ("网络", small(FTTransformer, objective="regression"))])

    def metric(y, p, sample_weight=None):
        return np.average(np.abs(y - p), weights=sample_weight)

    tuner = ModelTuner(model, search_space={}, metric=metric, direction="minimize", n_jobs=1)
    weight = np.arange(1, 21)
    tuner.fit(
        X[:60],
        amount[:60],
        eval_set=(X[60:], amount[60:]),
        validation_weight=weight,
        n_trials=1,
        show_progress_bar=False,
    )
    table = tuner.get_validation_predictions()
    assert_allclose(tuner.best_score_, np.average(np.abs(table["真实标签"] - table["预测值"]), weights=weight))
    assert tuner.get_best_model().steps[-1][1].objective == "regression"


@pytest.mark.parametrize("name", ["resnet", "fttransformer", "tabm", "tabpfn"])
def test_auto_tuner_supports_net_spaces(name):
    tuner = AutoTuner.create(name, n_jobs=1)
    space = tuner._get_adaptive_search_space()
    assert space
    if name == "tabm":
        assert "k" not in space
    elif name == "tabpfn":
        assert set(space) == {"n_estimators", "softmax_temperature"}
    else:
        assert "learning_rate" in space
    reg = AutoTuner.create(name, model_params={"objective": "regression"}, n_jobs=1)
    assert reg.direction == "minimize" and reg.metric.name == "均方根误差"


def test_fixed_head_count_and_pretrained_boundary(data):
    X, y, _ = data
    tuner = ModelTuner(FTTransformer(n_heads=np.int64(3)), n_jobs=1)
    assert all(width % 3 == 0 for width in tuner._get_adaptive_search_space()["d_block"]["choices"])
    for params in [{"loss": FocalLoss()}, {"optimizer": "adam"}, {"max_epochs": 2}]:
        with pytest.raises(ValueError, match="预训练推理"):
            TabPFN(**params).fit(X, y)
    with pytest.raises(ValueError, match="TabPFN"):
        ModelTuner(TabPFN, loss=FocalLoss(), n_jobs=1)


def test_torch_loss_buffers_are_part_of_resume_contract(torch_runtime, data, tmp_path):
    import copy

    torch = torch_runtime
    X, y, _ = data
    first = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    second = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    fingerprint = ModelTuner._configuration_fingerprint
    assert fingerprint(first) == fingerprint(copy.deepcopy(first))
    assert fingerprint(first) != fingerprint(second)
    tuner = ModelTuner(small(ResNet, loss=first), search_space={}, n_jobs=1, random_state=4)
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    tuner.save(tmp_path / "网络调参.pkl")
    restored = ModelTuner.load(tmp_path / "网络调参.pkl")
    restored.fit(X, y, n_trials=1, show_progress_bar=False)
    restored.model_params["loss"] = second
    restored._refresh_fixed_params()
    with pytest.raises(ValueError, match="配置|口径|参数"):
        restored.fit(X, y, n_trials=1, show_progress_bar=False)


def test_legacy_net_parameters_restore_without_retraining(data, tmp_path):
    X, y, _ = data
    model = small(ResNet).fit(X, y)
    expected = model.predict_proba(X[:3])
    for key in ("loss", "loss_params", "optimizer", "optimizer_params"):
        delattr(model, key)
    model.save(tmp_path / "旧网络.pkl")
    restored = ResNet.load(tmp_path / "旧网络.pkl")
    assert restored.get_params()["optimizer"] == "adamw"
    assert_allclose(restored.predict_proba(X[:3]), expected)
