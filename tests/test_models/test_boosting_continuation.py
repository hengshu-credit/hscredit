"""提升树追加轮数、旧模型不变、字段对齐和原生参数兼容。"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from hscredit import XGBoost, LightGBM, CatBoost, NGBoost


def make(cls, rounds=3, **kwargs):
    options = {"random_state": 3, "validation_fraction": 0, "n_jobs": 1}
    if cls is CatBoost:
        options.update(iterations=rounds, depth=3, allow_writing_files=False)
    else:
        options["n_estimators"] = rounds
    if cls is XGBoost:
        options["max_depth"] = 3
    if cls is LightGBM:
        options.update(verbosity=-1, min_child_samples=5)
    if cls is NGBoost:
        options["natural_gradient"] = False
    return cls(**{**options, **kwargs})


def rounds(model):
    native = model.get_native_model()
    if isinstance(model, XGBoost):
        return native.get_booster().num_boosted_rounds()
    if isinstance(model, LightGBM):
        return native.booster_.current_iteration()
    if isinstance(model, CatBoost):
        return native.tree_count_
    return len(native.base_models)


@pytest.fixture
def data():
    r = np.random.default_rng(8)
    X = pd.DataFrame(r.normal(size=(96, 3)), columns=["分数", "金额", "次数"])
    return X, (X["分数"] + X["次数"] > 0.1).astype(int)


@pytest.mark.parametrize("cls", [XGBoost, LightGBM, CatBoost, NGBoost])
def test_continue_same_instance_adds_rounds_and_preserves_prefix(cls, data, tmp_path):
    X, y = data
    m = make(cls).fit(X, y)
    before = m.predict_proba(X)
    source = make(cls).fit(X, y)
    m.fit(X[X.columns[::-1]], y, continue_training=True)
    assert rounds(m) == 6 and list(m.feature_names_in_) == list(X.columns)
    native = m.get_native_model()
    if cls is XGBoost:
        prefix = native.predict_proba(X, iteration_range=(0, 3))
    elif cls is LightGBM:
        prefix = native.predict_proba(X, num_iteration=3)
    elif cls is CatBoost:
        prefix = native.predict_proba(X, ntree_end=3)
    else:
        prefix = native.predict_proba(X, max_iter=3)
    assert_allclose(prefix, before, rtol=1e-6, atol=1e-7)
    destination = make(cls, rounds=2).fit(X, y, init_model=source)
    assert rounds(source) == 3 and rounds(destination) == 5
    assert_allclose(source.predict_proba(X), before)
    destination.save(tmp_path / "继续训练.pkl")
    restored = cls.load(tmp_path / "继续训练.pkl")
    restored.fit(X, y, continue_training=True)
    assert rounds(restored) == 7
    m.fit(X, y)
    assert rounds(m) == 3


@pytest.mark.parametrize("cls", [XGBoost, LightGBM, CatBoost, NGBoost])
def test_continuation_rejects_invalid_source_or_schema(cls, data):
    X, y = data
    with pytest.raises(ValueError, match="已拟合"):
        make(cls).fit(X, y, continue_training=True)
    old = make(cls).fit(X, y)
    with pytest.raises(ValueError, match="只能选择"):
        old.fit(X, y, continue_training=True, init_model=old)
    source = make(cls).fit(X, y)
    with pytest.raises(ValueError, match="缺少"):
        make(cls).fit(X.drop(columns="分数"), y, init_model=source)


def test_xgboost_native_alias_and_catboost_category_continuation(data):
    X, y = data
    old = make(XGBoost).fit(X, y)
    new = make(XGBoost, rounds=2).fit(X, y, xgb_model=old)
    assert rounds(new) == 5
    X = X.assign(商品=np.where(X["次数"] > 0, "甲", "乙"))
    cat = make(CatBoost).fit(X, y, cat_features=["商品"])
    cat.fit(X[X.columns[::-1]], y, continue_training=True)
    assert rounds(cat) == 6 and cat.predict_proba(X).shape == (len(X), 2)


def test_ngboost_keeps_initial_distribution_on_new_label_distribution(data):
    X, y = data
    m = make(NGBoost).fit(X, y)
    initial = m.get_native_model().init_params.copy()
    changed = y.copy()
    changed.iloc[:40] = 1
    m.fit(X, changed, continue_training=True)
    assert_allclose(m.get_native_model().init_params, initial)
    m.set_params(learning_rate=0.2)
    with pytest.raises(ValueError, match="learning_rate"):
        m.fit(X, y, continue_training=True)


def test_xgboost_old_early_stopping_metadata_does_not_hide_new_trees(data):
    import xgboost as xgb

    X, y = data
    old = make(XGBoost).fit(X, y)
    old.get_native_model().get_booster().set_attr(best_iteration="0", best_score="0.5", best_ntree_limit="1")
    updated = make(XGBoost, rounds=2).fit(X, y, init_model=old)
    booster = updated.get_native_model().get_booster()
    assert booster.attr("best_iteration") is None
    assert_allclose(updated.predict_proba(X)[:, 1], booster.predict(xgb.DMatrix(X)))
    assert old.get_native_model().get_booster().attr("best_iteration") == "0"


def test_ngboost_old_validation_iteration_is_not_reused_without_validation(data):
    X, y = data
    model = make(NGBoost).fit(X, y)
    model.get_native_model().best_val_loss_itr = 1
    model.fit(X, y, continue_training=True)
    assert model.best_iteration_ is None


def test_ngboost_continuation_uses_current_early_stopping_configuration(data):
    X, y = data
    model = make(NGBoost, early_stopping_rounds=2).fit(X, y, eval_set=[(X, y)])
    model.set_params(early_stopping_rounds=None)
    model.fit(X, y, continue_training=True)
    assert rounds(model) == 6
    assert model.best_iteration_ is None
    assert "val" not in model.evals_result_


def test_ngboost_continuation_preserves_custom_loss_and_rejects_a_change(data):
    from hscredit import FocalLoss

    X, y = data
    source = make(NGBoost, objective=FocalLoss(gamma=1)).fit(X, y)
    before = source.predict_proba(X)
    destination = make(NGBoost, rounds=2, objective=FocalLoss(gamma=1)).fit(X, y, init_model=source)
    assert rounds(destination) == 5
    assert_allclose(destination.get_native_model().predict_proba(X, max_iter=3), before)
    with pytest.raises(ValueError, match="训练损失"):
        make(NGBoost, objective=FocalLoss(gamma=2)).fit(X, y, init_model=source)


def test_catboost_continuation_reports_native_cpu_requirement(data):
    X, y = data
    source = make(CatBoost).fit(X, y)
    with pytest.raises(ValueError, match="仅支持 CPU"):
        make(CatBoost, task_type="GPU").fit(X, y, init_model=source)
