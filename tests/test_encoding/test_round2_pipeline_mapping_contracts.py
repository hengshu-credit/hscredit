"""R02/R07：原生Pipeline目标隔离、类别协议和输出schema。"""

import json
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression as NativeLR
from sklearn.pipeline import Pipeline

from hscredit.core.encoders import (
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    CardinalityEncoder,
)

ENCODERS = [
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    CardinalityEncoder,
]


@pytest.mark.parametrize("cls", ENCODERS)
def test_declared_target_never_appears_in_default_output(cls):
    X = pd.DataFrame({"x": ["a", "b", "a", "b"], "target": [0, 1, 1, 0]})
    encoder = cls(target="target", n_jobs=1)
    transformed = encoder.fit_transform(X, X.target)
    assert "target" not in transformed.columns
    pd.testing.assert_frame_equal(encoder.transform(X), encoder.transform(X.drop(columns="target")))


@pytest.mark.parametrize("training_mode", ["in_sample", "oof"])
def test_native_sklearn_pipeline_predicts_without_label(training_mode):
    rng = np.random.RandomState(41)
    df = pd.DataFrame({"x": rng.choice(["A", "B", "C"], 90), "target": rng.randint(0, 2, 90)})
    pipeline = Pipeline(
        [
            ("encoder", TargetEncoder(target="target", training_mode=training_mode, cv=3, n_jobs=1)),
            ("model", NativeLR()),
        ]
    )
    pipeline.fit(df, df.target)
    assert pipeline[-1].feature_names_in_.tolist() == ["x"]
    assert pipeline.predict(df[["x"]]).shape == (90,)


@pytest.mark.parametrize("cls", ENCODERS)
def test_label_passthrough_is_explicit_legacy_mode(cls):
    X = pd.DataFrame({"x": ["a", "b", "a", "b"], "target": [0, 1, 1, 0]})
    encoder = clone(cls(target="target", passthrough_target=True, n_jobs=1))
    output = encoder.fit_transform(X)
    assert output.target.tolist() == X.target.tolist()
    assert "target" not in encoder.cols_


@pytest.mark.parametrize(
    "cls", [WOEEncoder, TargetEncoder, CountEncoder, OrdinalEncoder, QuantileEncoder, CatBoostEncoder]
)
def test_real_unknown_named_category_is_not_overwritten(cls):
    X = pd.DataFrame({"x": ["__UNKNOWN__"] * 3 + ["other"] * 5})
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    # 序数编码按类别排序；改名仍保持相对顺序，避免把正常排序变化误判为哨兵冲突。
    renamed = X.replace({"x": {"__UNKNOWN__": "__RENAMED__"}})
    left = cls(n_jobs=1).fit(X, y).transform(X)
    right = cls(n_jobs=1).fit(renamed, y).transform(renamed)
    np.testing.assert_allclose(left, right)


@pytest.mark.parametrize("cls", ENCODERS)
def test_typed_json_roundtrip_keeps_numeric_string_and_real_reserved_names(cls):
    X = pd.DataFrame({"x": pd.Series([1, 1, "1", "1", "__UNKNOWN__", "__UNKNOWN__", None, "missing"], dtype=object)})
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    original = cls(n_jobs=1).fit(X, y)
    encoded = json.loads(json.dumps(original.export_mapping(), ensure_ascii=False, allow_nan=False))
    restored = cls(n_jobs=1).import_mapping(encoded)
    pd.testing.assert_frame_equal(original.transform(X), restored.transform(X))
    for probe in [pd.DataFrame({"x": [1, "1", "new", None]}), X.iloc[:0]]:
        pd.testing.assert_frame_equal(original.transform(probe), restored.transform(probe))


@pytest.mark.parametrize("mode", ["in_sample", "oof"])
def test_drop_invariant_schema_and_woe_metadata(mode):
    X = pd.DataFrame({"constant": ["fixed"] * 12, "x": ["a", "b"] * 6, "n": np.arange(12), "target": [0, 1] * 6})
    encoder = WOEEncoder(
        cols=["constant", "x"], target="target", drop_invariant=True, training_mode=mode, cv=3, n_jobs=1
    )
    trained = encoder.fit_transform(X)
    inference = encoder.transform(X.drop(columns="target"))
    assert list(trained.columns) == list(inference.columns) == ["x", "n"]
    assert trained.attrs["hscredit_encoding"] == inference.attrs["hscredit_encoding"] == "woe"
    assert encoder.get_feature_names_out().tolist() == ["x", "n"]


def test_legacy_mapping_import_remains_supported():
    payload = {
        "encoder_type": "TargetEncoder",
        "cols": ["x"],
        "cols_": ["x"],
        "mapping_": {"x": {"a": 0.2, "b": 0.8, "__UNKNOWN__": 0.5}},
        "handle_unknown": "value",
        "handle_missing": "value",
        "extra_state": {"global_mean_": 0.5},
    }
    restored = TargetEncoder(n_jobs=1).import_mapping(payload)
    assert restored.transform(pd.DataFrame({"x": ["a", "b", "new"]})).x.tolist() == [0.2, 0.8, 0.5]


def test_future_mapping_version_and_wrong_encoder_rejected_transactionally():
    encoder = TargetEncoder(n_jobs=1).fit(pd.DataFrame({"x": ["a", "b"]}), [0, 1])
    before = encoder.transform(pd.DataFrame({"x": ["a", "b"]}))
    payload = encoder.export_mapping()
    payload["version"] = 999
    with pytest.raises(ValueError, match="版本"):
        encoder.import_mapping(payload)
    pd.testing.assert_frame_equal(before, encoder.transform(pd.DataFrame({"x": ["a", "b"]})))
    payload = encoder.export_mapping()
    payload["encoder_type"] = "WOEEncoder"
    with pytest.raises(ValueError, match="类型"):
        encoder.import_mapping(payload)


def test_count_real_other_and_rare_unknown_are_distinct():
    X = pd.DataFrame({"x": ["__OTHER__"] * 4 + ["rare"] + ["known"] * 3})
    encoder = CountEncoder(min_group_size=2, n_jobs=1).fit(X)
    assert encoder.transform(pd.DataFrame({"x": ["__OTHER__", "rare", "new"]})).x.tolist() == [4, 1, 0]
    restored = CountEncoder(n_jobs=1).import_mapping(json.loads(json.dumps(encoder.export_mapping(), allow_nan=False)))
    pd.testing.assert_frame_equal(encoder.transform(X), restored.transform(X))
    strict = CountEncoder(min_group_size=2, handle_unknown="error", n_jobs=1).fit(X)
    with pytest.raises(Exception, match="未知类别"):
        strict.transform(pd.DataFrame({"x": ["new"]}))


@pytest.mark.parametrize(
    "cls", [WOEEncoder, TargetEncoder, CountEncoder, OrdinalEncoder, QuantileEncoder, CatBoostEncoder]
)
def test_explicit_legacy_export_and_import(cls):
    X = pd.DataFrame({"x": ["a", "a", "b", None]})
    original = cls(n_jobs=1).fit(X, [0, 1, 1, 0])
    restored = cls(n_jobs=1).import_mapping(original.export_mapping(legacy=True))
    pd.testing.assert_frame_equal(original.transform(X), restored.transform(X))


def test_woe_legacy_missing_value_and_new_exact_type_semantics():
    legacy = WOEEncoder(n_jobs=1).load({"x": {"1": 0.2, "nan": 0.7}})
    assert legacy.transform(pd.DataFrame({"x": [1, None]})).x.tolist() == [0.2, 0.7]
    current = WOEEncoder(n_jobs=1).fit(pd.DataFrame({"x": [1, 1, 2, 2]}), [0, 0, 1, 1])
    assert current.transform(pd.DataFrame({"x": ["1"]})).x.iloc[0] == 0.0


def test_count_typed_nan_json_roundtrip_and_numpy_bool_config():
    X = pd.DataFrame(
        {"x": pd.Series([float("nan"), np.float32("nan"), np.float64("nan"), None, pd.NA, pd.NaT, "x"], dtype=object)}
    )
    original = CountEncoder(n_jobs=1, normalize=np.bool_(False)).fit(X)
    restored = CountEncoder(n_jobs=1).import_mapping(json.loads(json.dumps(original.export_mapping(), allow_nan=False)))
    assert restored.normalize == np.bool_(False)
    pd.testing.assert_frame_equal(original.transform(X), restored.transform(X))


def test_old_pickle_new_parameter_migration_preserves_legacy_mode():
    encoder = TargetEncoder(target="target", n_jobs=1).fit(pd.DataFrame({"x": ["a", "b"], "target": [0, 1]}))
    del encoder.passthrough_target
    with pytest.warns(FutureWarning, match="旧编码器"):
        old = pickle.loads(pickle.dumps(encoder))
    assert clone(old).passthrough_target is True
    assert old.transform(pd.DataFrame({"x": ["a"], "target": [0]})).columns.tolist() == ["x", "target"]


def test_sparse_onehot_label_policy_and_schema():
    X = pd.DataFrame({"x": ["a", "b", "a", "b"], "target": [0, 1, 1, 0]})
    current = OneHotEncoder(target="target", sparse_output=True, return_df=False, n_jobs=1)
    assert current.fit_transform(X).shape[1] == 2
    assert current.transform(X[["x"]]).shape[1] == 2
    legacy = OneHotEncoder(target="target", passthrough_target=True, sparse_output=True, return_df=False, n_jobs=1)
    assert legacy.fit_transform(X).shape[1] == 3


def test_gbm_native_pipeline_and_frozen_leaf_schema():
    pytest.importorskip("xgboost")
    from hscredit.core.encoders import GBMEncoder

    X = pd.DataFrame({"x": np.arange(40), "target": [0, 1] * 20})
    encoder = GBMEncoder(
        target="target", model_type="xgboost", output_type="onehot", n_estimators=3, max_depth=2, n_jobs=1
    )
    pipeline = Pipeline([("encoder", encoder), ("model", NativeLR())]).fit(X, X.target)
    assert "target" not in pipeline[-1].feature_names_in_
    assert pipeline.predict(X[["x"]].iloc[:1]).shape == (1,)
    assert encoder.transform(X[["x"]].iloc[:1]).columns.tolist() == encoder.get_feature_names_out().tolist()
    with pytest.raises(NotImplementedError, match="完整|save_artifact"):
        encoder.export_mapping()


def test_all_invariant_oof_returns_empty_features_without_leaking_label():
    frame = pd.DataFrame({"x": ["a"] * 6, "target": [0, 1] * 3})
    encoder = TargetEncoder(target="target", drop_invariant=True, training_mode="oof", n_jobs=1)
    assert encoder.fit_transform(frame).shape == (6, 0)
    assert encoder.transform(frame[["x"]]).shape == (6, 0)


def test_invalid_loaded_label_policy_cannot_enable_passthrough():
    encoder = TargetEncoder(target="target", n_jobs=1).fit(pd.DataFrame({"x": ["a", "b"], "target": [0, 1]}))
    payload = encoder.export_mapping()
    payload["parameters"]["passthrough_target"] = {"type": "str", "value": "False"}
    with pytest.raises(ValueError, match="布尔"):
        encoder.import_mapping(payload)
    assert encoder.passthrough_target is False


def test_identity_dependent_decimal_nan_json_is_explicitly_rejected():
    from decimal import Decimal
    frame = pd.DataFrame({"x": pd.Series([Decimal("NaN"), Decimal("NaN"), "x"], dtype=object)})
    encoder = CountEncoder(n_jobs=1).fit(frame)
    assert encoder.transform(frame).x.tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="对象身份"):
        encoder.export_mapping()
