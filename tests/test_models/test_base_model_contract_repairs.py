"""模型基类输入、状态、评估与序列化契约回归测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from hscredit.core.models import BaseRiskModel, RandomForest
from hscredit.core.models.base import _lift_score


@pytest.fixture
def fitted_forest():
    X, y = make_classification(
        n_samples=160,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        random_state=17,
    )
    frame = pd.DataFrame(X, columns=["年龄", "收入", "负债", "历史逾期"])
    model = RandomForest(n_estimators=40, random_state=17).fit(frame, y)
    return model, frame, y


def test_dataframe_prediction_reorders_training_fields_and_ignores_extras(fitted_forest):
    model, frame, _ = fitted_forest
    expected = model.predict_proba(frame)[:, 1]
    changed = frame[["历史逾期", "负债", "年龄", "收入"]].copy()
    changed["申请编号"] = np.arange(len(changed))

    actual = model.predict_proba(changed)[:, 1]

    np.testing.assert_allclose(actual, expected)


def test_dataframe_prediction_reports_missing_training_fields_in_chinese(fitted_forest):
    model, frame, _ = fitted_forest

    with pytest.raises(ValueError, match="缺少训练字段.*收入"):
        model.predict_proba(frame.drop(columns=["收入"]))


def test_ndarray_prediction_validates_feature_count(fitted_forest):
    model, frame, _ = fitted_forest

    with pytest.raises(ValueError, match="特征数量.*4.*3"):
        model.predict_proba(frame.to_numpy()[:, :3])


def test_explicit_y_overrides_dataframe_target_column():
    frame = pd.DataFrame(
        {
            "年龄": [20, 30, 40, 50],
            "收入": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 0, 1],
        }
    )
    explicit_y = np.array([0, 1, 1, 1])

    model = RandomForest(n_estimators=5, random_state=17, target="target").fit(frame, explicit_y)

    assert model.bad_rate_ == pytest.approx(0.75)
    assert model.feature_names_in_ == ["年龄", "收入"]


def test_unfitted_wrapper_raises_not_fitted_error():
    model = RandomForest(n_estimators=5)

    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 3)))


def test_lift_small_sample_uses_at_least_one_observation():
    y_true = np.array([0, 1] * 10)
    y_prob = np.linspace(0.0, 1.0, len(y_true))

    assert np.isfinite(_lift_score(y_true, y_prob, top_ratio=0.01))


def test_lift_uses_series_positions_not_index_labels():
    y_true = pd.Series([0, 1] * 10, index=np.arange(100, 120))
    y_prob = np.linspace(0.0, 1.0, len(y_true))

    assert np.isfinite(_lift_score(y_true, y_prob, top_ratio=0.1))


def test_evaluate_rejects_unknown_metrics(fitted_forest):
    model, frame, y = fitted_forest

    with pytest.raises(ValueError, match="不支持的评估指标"):
        model.evaluate(frame, y, metrics=["not-a-metric"])


def test_json_save_rejects_models_without_native_roundtrip(fitted_forest, tmp_path):
    model, _, _ = fitted_forest

    with pytest.raises(ValueError, match="不支持.*JSON"):
        model.save(tmp_path / "forest.json")


def test_native_json_roundtrip_must_restore_a_fitted_model(monkeypatch, tmp_path, fitted_forest):
    """JSON metadata without a native artifact must never masquerade as fitted."""
    model, _, _ = fitted_forest
    path = tmp_path / "broken.json"
    path.write_text(
        '{"model_class":"hscredit.core.models.classical.sklearn_models.RandomForest",'
        '"params":{},"classes_":[0,1],"n_features_in_":4,"feature_names_in_":["a","b","c","d"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少原生模型"):
        BaseRiskModel.load(path)
