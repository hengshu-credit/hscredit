"""SVM 与 DecisionTreeClassifier 统一模型契约测试。"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from hscredit.core.models import AutoTuner, BaseRiskModel, DecisionTreeClassifier, ModelTuner, RandomForest, SVM
from hscredit.exceptions import SerializationError, ValidationError


@pytest.fixture
def binary_frame():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=19,
    )
    return pd.DataFrame(X, columns=list("abcde")), y


@pytest.mark.parametrize("model", [SVM(kernel="linear"), DecisionTreeClassifier(max_depth=3)])
def test_new_sklearn_models_clone_fit_and_predict_probability(model, binary_frame):
    """缺少 clone 参数或概率接口都会破坏 sklearn 与评分卡组合。"""
    X, y = binary_frame

    fitted = clone(model).fit(X, y, sample_weight=np.ones(len(y)))
    probability = fitted.predict_proba(X)

    assert probability.shape == (len(X), 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert fitted.predict_score(X).shape == (len(X),)


def test_svm_uses_svc_with_probability_enabled(binary_frame):
    """SVM 若未启用 SVC 概率训练，统一 predict_proba 契约会在拟合后才崩溃。"""
    X, y = binary_frame

    model = SVM(kernel="linear", random_state=19).fit(X, y)
    importances = model.get_feature_importances()

    assert isinstance(model.get_native_model(), SVC)
    assert model.get_native_model().probability is True
    assert set(importances.index) == set("abcde")
    assert len(importances) == 5
    assert (importances >= 0).all()


def test_svm_rejects_probability_false_in_chinese():
    """禁止创建缺少概率能力的半兼容 SVM。"""
    with pytest.raises(ValidationError, match="probability.*True"):
        SVM(probability=False)


def test_nonlinear_svm_reports_missing_native_importance(binary_frame):
    """非线性核不能用伪造的零向量冒充特征重要性。"""
    X, y = binary_frame
    model = SVM(kernel="rbf").fit(X, y)

    with pytest.raises(ValidationError, match="非线性.*特征重要性"):
        model.get_feature_importances()


def test_decision_tree_wraps_native_estimator_without_unsupported_options(binary_frame):
    """包装器不能把 n_jobs 或 verbose 泄漏给不支持这些参数的决策树。"""
    X, y = binary_frame

    model = DecisionTreeClassifier(max_depth=2, random_state=19).fit(X, y)

    assert isinstance(model.get_native_model(), SklearnDecisionTreeClassifier)
    assert model.get_native_model().get_params()["max_depth"] == 2


@pytest.mark.parametrize("model", [SVM(n_jobs=2), DecisionTreeClassifier(n_jobs=2)])
def test_new_sklearn_models_preserve_wrapper_n_jobs_without_native_leak(model, binary_frame):
    """统一 n_jobs 必须可 clone，但不能传给不支持它的原生 estimator。"""
    X, y = binary_frame

    cloned = clone(model)
    fitted = cloned.fit(X, y)

    assert cloned.n_jobs == 2
    assert "n_jobs" not in fitted.get_native_model().get_params()


@pytest.mark.parametrize(
    ("model", "parameter", "value"),
    [
        (SVM(kernel="linear"), "C", 2.5),
        (DecisionTreeClassifier(), "max_depth", 5),
        (RandomForest(n_estimators=5), "max_depth", 4),
    ],
)
def test_sklearn_wrapper_set_params_reaches_native_estimator(model, parameter, value, binary_frame):
    """set_params 若只改包装器属性，调优和 Pipeline 会继续使用陈旧构造参数。"""
    X, y = binary_frame

    model.set_params(**{parameter: value}).fit(X, y)

    assert model.get_native_model().get_params()[parameter] == value


def test_svm_set_params_cannot_disable_probability():
    """初始化后的 set_params 也不能绕过统一概率契约。"""
    with pytest.raises(ValidationError, match="probability.*True"):
        SVM().set_params(probability=False)


@pytest.mark.parametrize("model_name", ["svm", "svc"])
def test_auto_tuner_resolves_svm_names(model_name):
    """SVM 字符串若缺少注册，会错误拒绝合法的新模型。"""
    assert AutoTuner.create(model_name).model_class is SVM


@pytest.mark.parametrize("model_name", ["decisiontree", "dt"])
def test_auto_tuner_resolves_decision_tree_names(model_name):
    """决策树字符串必须解析为 hscredit 包装器而不是 sklearn 原生类。"""
    assert AutoTuner.create(model_name).model_class is DecisionTreeClassifier


def test_new_sklearn_models_use_model_specific_adaptive_spaces():
    """未知模型默认 XGBoost 空间会向 SVC 和决策树传入完全无效的参数。"""
    svm_space = ModelTuner(SVM)._get_adaptive_search_space()
    tree_space = ModelTuner(DecisionTreeClassifier)._get_adaptive_search_space()

    assert {"C", "kernel", "gamma"} <= set(svm_space)
    assert {"criterion", "max_depth", "ccp_alpha"} <= set(tree_space)
    assert "n_estimators" not in svm_space
    assert "learning_rate" not in tree_space


def test_new_model_joblib_round_trip_uses_new_class_identity(tmp_path, binary_frame):
    """新制品必须记录新类身份，不能继续写出已删除的类路径。"""
    X, y = binary_frame
    fitted = RandomForest(n_estimators=5, random_state=19).fit(X, y)
    path = tmp_path / "forest.joblib"

    joblib.dump(fitted, path)
    loaded = joblib.load(path)

    assert type(loaded).__name__ == "RandomForest"
    assert type(loaded).__module__ == "hscredit.core.models.classical.sklearn_models"
    np.testing.assert_allclose(loaded.predict_proba(X), fitted.predict_proba(X))


def test_old_json_class_path_is_not_remapped(tmp_path):
    """硬重命名不能通过 JSON 类名映射暗中保留旧模型兼容。"""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "model_class": "hscredit.core.models.classical.sklearn_models.RandomForestRiskModel",
                "params": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises((AttributeError, SerializationError)):
        BaseRiskModel.load(path)
