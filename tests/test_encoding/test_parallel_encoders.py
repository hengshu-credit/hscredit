"""编码器列级并行的契约与串并行一致性测试。"""

import inspect
import pickle
from collections import Counter
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core import encoders
from hscredit.core.encoders import (
    CardinalityEncoder,
    CatBoostEncoder,
    CountEncoder,
    GBMEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    TargetEncoder,
    WOEEncoder,
)
from hscredit.exceptions import ParallelExecutionError
from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET, parallel_execute


ENCODER_CLASSES = [
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    GBMEncoder,
    CardinalityEncoder,
]


@pytest.fixture
def encoder_xy():
    index = pd.Index([11, 13, 17, 19, 23, 29, 31, 37], name="样本")
    X = pd.DataFrame(
        {
            "a": ["乙", "甲", np.nan, "乙", "丙", "甲", "丙", "乙"],
            "b": [2, 1, 2, np.nan, 3, 1, 3, 2],
            "透传": pd.Series([8, 7, 6, 5, 4, 3, 2, 1], index=index, dtype="int64"),
        },
        index=index,
    )
    y = pd.Series([0, 1, 1, 0, 1, 0, 1, 0], index=index, name="FPD")
    return X, y


def _parallel_kwargs(cls):
    kwargs = {
        "n_jobs": 0.5,
        "parallel_backend": "threading",
        "parallel_config": {"batch_size": 2},
    }
    if cls is OrdinalEncoder:
        kwargs["mapping"] = {}
    elif cls is CardinalityEncoder:
        kwargs["special_values"] = []
    elif cls is GBMEncoder:
        kwargs["model_params"] = {}
    return kwargs


def test_all_exported_encoders_expose_explicit_parallel_parameters():
    """删除任一具体构造器的公共参数时，本测试应失败。"""
    exported = [getattr(encoders, name) for name in encoders.__all__]
    concrete = [cls for cls in exported if cls in ENCODER_CLASSES]
    assert concrete == ENCODER_CLASSES

    missing = {}
    for cls in concrete:
        params = inspect.signature(cls.__init__).parameters
        absent = [name for name in ("n_jobs", "parallel_backend", "parallel_config") if name not in params]
        wrong_default = [
            name
            for name, expected in (("n_jobs", -1), ("parallel_backend", None), ("parallel_config", None))
            if name in params and params[name].default != expected
        ]
        if absent or wrong_default:
            missing[cls.__name__] = {"缺失": absent, "默认值错误": wrong_default}
    assert missing == {}


@pytest.mark.parametrize("cls", ENCODER_CLASSES)
def test_encoder_parallel_params_survive_get_params_clone_and_pickle(cls):
    """构造器未原样保存公共参数或可变参数时，clone/pickle 契约会失败。"""
    kwargs = _parallel_kwargs(cls)
    encoder = cls(**kwargs)
    params = encoder.get_params(deep=False)
    assert params["n_jobs"] == 0.5
    assert params["parallel_backend"] == "threading"
    assert params["parallel_config"] is kwargs["parallel_config"]

    cloned = clone(encoder)
    assert cloned.get_params(deep=False)["parallel_config"] == {"batch_size": 2}
    assert cloned.parallel_config is not encoder.parallel_config

    restored = pickle.loads(pickle.dumps(encoder))
    assert restored.get_params(deep=False)["parallel_config"] == {"batch_size": 2}


@pytest.mark.parametrize(
    "encoder_factory",
    [
        lambda n, b: CountEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
        lambda n, b: WOEEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
        lambda n, b: TargetEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
        lambda n, b: OneHotEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
    ],
)
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_encoder_parallel_fit_transform_matches_serial(encoder_factory, backend, encoder_xy):
    """列 worker、状态提交或结果合并乱序时，本测试应失败。"""
    X, y = encoder_xy
    serial = encoder_factory(1, None).fit(X, y)
    parallel = encoder_factory(2, backend).fit(X, y)

    assert serial.export_mapping() == parallel.export_mapping()
    pd.testing.assert_frame_equal(serial.transform(X), parallel.transform(X), check_exact=True)
    pd.testing.assert_frame_equal(serial.fit_transform(X, y), parallel.fit_transform(X, y), check_exact=True)


@pytest.mark.parametrize("backend", [None, "threading", "loky"])
def test_parallel_missing_unknown_dtype_index_and_onehot_order(backend, encoder_xy):
    """缺失/未知策略、dtype、索引或独热列序被调度改变时，本测试应失败。"""
    X, _ = encoder_xy
    train = X[["a", "b", "透传"]]
    probe = pd.DataFrame(
        {"a": ["未知", np.nan], "b": [99, np.nan], "透传": [4, 5]},
        index=pd.Index([101, 103], name="样本"),
    )
    n_jobs = 1 if backend is None else 2

    count = CountEncoder(cols=["a", "b"], n_jobs=n_jobs, parallel_backend=backend).fit(train)
    count_out = count.transform(probe)
    assert count_out.index.equals(probe.index)
    assert list(count_out.columns) == ["a", "b", "透传"]
    assert count_out["透传"].dtype == np.dtype("int64")
    assert count_out.loc[101, ["a", "b"]].tolist() == [0, 0]

    onehot = OneHotEncoder(cols=["a", "b"], n_jobs=n_jobs, parallel_backend=backend).fit(train)
    out = onehot.transform(probe)
    assert out.index.equals(probe.index)
    assert list(out.columns) == ["透传"] + onehot.feature_names_
    assert out["透传"].dtype == np.dtype("int64")
    assert all(np.issubdtype(dtype, np.integer) for dtype in out[onehot.feature_names_].dtypes)


class _FailingCountEncoder(CountEncoder):
    """仅用于验证基类列事务。"""

    def _fit_column(self, column, values, y=None):
        if column == "坏列":
            raise RuntimeError("列拟合失败")
        return super()._fit_column(column, values, y)


def test_failed_parallel_refit_preserves_complete_previous_model():
    """worker 失败后若提交部分新状态或破坏旧模型，本测试应失败。"""
    X1 = pd.DataFrame({"好列": ["a", "a", "b", "b"], "坏列": [1, 1, 2, 2]})
    encoder = _FailingCountEncoder(cols=["好列"], n_jobs=2, parallel_backend="threading").fit(X1)
    before_mapping = encoder.export_mapping()["mapping_"]
    before_output = encoder.transform(X1[["好列"]])

    encoder.cols = ["好列", "坏列"]
    with pytest.raises(ParallelExecutionError, match="坏列"):
        encoder.fit(X1)

    assert encoder.export_mapping()["mapping_"] == before_mapping
    pd.testing.assert_frame_equal(encoder.transform(X1[["好列"]]), before_output)


@pytest.mark.parametrize(
    "factory",
    [
        lambda n, b: OrdinalEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
        lambda n, b: QuantileEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
        lambda n, b: CatBoostEncoder(
            cols=["a", "b"], sigma=0.05, random_state=42, n_jobs=n, parallel_backend=b
        ),
        lambda n, b: CardinalityEncoder(
            cols=["a", "b"], max_categories=3, special_values=[3], n_jobs=n, parallel_backend=b
        ),
    ],
)
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_remaining_column_encoders_match_all_learned_state(factory, backend, encoder_xy):
    """遗漏任一列级映射或元数据提交时，本测试应失败。"""
    X, y = encoder_xy
    serial = factory(1, None).fit(X[["a", "b"]], y)
    parallel = factory(2, backend).fit(X[["a", "b"]], y)

    assert serial.export_mapping() == parallel.export_mapping()
    for attr in serial._EXTRA_STATE_ATTRS:
        left = getattr(serial, attr)
        right = getattr(parallel, attr)
        if isinstance(left, dict) and left and all(isinstance(v, pd.Series) for v in left.values()):
            assert list(left) == list(right)
            for column in left:
                pd.testing.assert_series_equal(left[column], right[column], check_exact=True)
        else:
            assert left == right
    for attr in ("category_counts_", "special_counts_"):
        if hasattr(serial, attr):
            left = getattr(serial, attr)
            right = getattr(parallel, attr)
            assert list(left) == list(right)
            for column in left:
                if isinstance(left[column], pd.Series):
                    pd.testing.assert_series_equal(left[column], right[column], check_exact=True)
                else:
                    assert left[column] == right[column]

    pd.testing.assert_frame_equal(
        serial.transform(X[["a", "b"]], y),
        parallel.transform(X[["a", "b"]], y),
        check_exact=True,
    )


MISSING_LIKE_FACTORIES = [
    ("count", lambda n, b: CountEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("woe", lambda n, b: WOEEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("target", lambda n, b: TargetEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("onehot", lambda n, b: OneHotEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("ordinal", lambda n, b: OrdinalEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("quantile", lambda n, b: QuantileEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("catboost", lambda n, b: CatBoostEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
    ("cardinality", lambda n, b: CardinalityEncoder(cols=["a"], n_jobs=n, parallel_backend=b)),
]


def _missing_key_signature(mapping):
    return [
        (type(key).__module__, type(key).__name__, repr(key), value)
        for key, value in mapping.items()
    ]


@pytest.mark.parametrize("name,factory", MISSING_LIKE_FACTORIES)
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_missing_like_keys_keep_serial_mapping_export_import_and_transform(name, factory, backend):
    """将 None/NaT/NA 误当浮点 NaN 归一时，本测试应失败。"""
    X = pd.DataFrame(
        {"a": pd.Series([None, None, float("nan"), pd.NaT, pd.NA, "文本"], dtype=object)}
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    serial = factory(1, None).fit(X, y)
    parallel = factory(2, backend).fit(X, y)
    assert _missing_key_signature(serial.mapping_["a"]) == _missing_key_signature(parallel.mapping_["a"])
    assert _missing_key_signature(serial.export_mapping()["mapping_"]["a"]) == _missing_key_signature(
        parallel.export_mapping()["mapping_"]["a"]
    )

    expected = serial.transform(X)
    pd.testing.assert_frame_equal(expected, parallel.transform(X), check_exact=True)
    restored = factory(1, None).import_mapping(parallel.export_mapping())
    pd.testing.assert_frame_equal(expected, restored.transform(X), check_exact=True)

    if name == "count":
        signature = _missing_key_signature(serial.mapping_["a"])
        assert ("builtins", "NoneType", "None", 2) in signature
        assert any(module == "pandas._libs.tslibs.nattype" for module, _, _, _ in signature)
        assert any(type_name == "NAType" for _, type_name, _, _ in signature)
        assert expected["a"].tolist() == [2, 2, 1, 1, 1, 1]


def test_float_nan_normalization_does_not_capture_other_scalar_types():
    """bool、Decimal、complex 或 pandas missing 标量被泛化时，本测试应失败。"""
    from decimal import Decimal

    mapping = {True: 1, Decimal("1.5"): 2, complex(1, 2): 3, None: 4, pd.NaT: 5, pd.NA: 6}
    normalized = CountEncoder._canonicalize_nan_keys(mapping)
    assert list(normalized.items()) == list(mapping.items())


def _typed_float_nan_signature(mapping):
    return [
        (type(key).__module__, type(key).__name__, value)
        for key, value in mapping.items()
        if type(key) is float or isinstance(key, np.floating)
        if np.isnan(key)
    ]


def test_pandas_value_counts_keeps_float_nan_buckets_by_scalar_type():
    """typed NaN 被错误视作同一 pandas 分组时，本基线刻画应失败。"""
    values = [
        float("nan"),
        np.float32("nan"),
        float("nan"),
        np.float64("nan"),
        None,
        pd.NaT,
        pd.NA,
        "x",
    ]
    series = pd.Series(values, dtype=object)
    counts = series.value_counts(dropna=False, sort=False)

    assert _typed_float_nan_signature(counts.to_dict()) == [
        ("builtins", "float", 2),
        ("numpy", "float32", 1),
        ("numpy", "float64", 1),
    ]
    assert series.map(counts).tolist() == [2, 1, 2, 1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "n_jobs,backend",
    [(1, None), (2, "threading"), (2, "loky")],
)
def test_count_encoder_preserves_typed_float_nan_buckets_through_roundtrips(n_jobs, backend):
    """fit 提交覆盖 typed NaN 桶或 transform 按对象身份查找时，本测试应失败。"""
    X = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    float("nan"),
                    np.float32("nan"),
                    float("nan"),
                    np.float64("nan"),
                    None,
                    pd.NaT,
                    pd.NA,
                    "x",
                ],
                dtype=object,
            )
        }
    )
    encoder = CountEncoder(cols=["a"], n_jobs=n_jobs, parallel_backend=backend).fit(X)
    expected_signature = [
        ("builtins", "float", 2),
        ("numpy", "float32", 1),
        ("numpy", "float64", 1),
    ]
    expected_values = [2, 1, 2, 1, 1, 1, 1, 1]

    assert _typed_float_nan_signature(encoder.mapping_["a"]) == expected_signature
    assert encoder.transform(X)["a"].tolist() == expected_values

    exported = pickle.loads(pickle.dumps(encoder.export_mapping()))
    assert _typed_float_nan_signature(exported["mapping_"]["a"]) == expected_signature
    restored = CountEncoder(cols=["a"]).import_mapping(exported)
    assert _typed_float_nan_signature(restored.mapping_["a"]) == expected_signature
    assert restored.transform(X)["a"].tolist() == expected_values

    loaded = pickle.loads(pickle.dumps(encoder))
    assert _typed_float_nan_signature(loaded.mapping_["a"]) == expected_signature
    assert loaded.transform(X)["a"].tolist() == expected_values


@pytest.mark.parametrize(
    "n_jobs,backend",
    [(1, None), (2, "threading"), (2, "loky")],
)
def test_non_float_nan_scalars_keep_pandas_count_and_map_semantics(n_jobs, backend):
    """Decimal/complex NaN 被 typed-float 逻辑规范化或误合并时，本测试应失败。"""
    X = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    Decimal("NaN"),
                    Decimal("NaN"),
                    complex(float("nan"), 0),
                    complex(float("nan"), 0),
                    True,
                    Decimal("1.5"),
                    "x",
                ],
                dtype=object,
            )
        }
    )
    baseline = X["a"].value_counts(dropna=False, sort=False)
    baseline_signature = _missing_key_signature(baseline.to_dict())
    assert [(module, name, value) for module, name, _, value in baseline_signature] == [
        ("decimal", "Decimal", 1),
        ("decimal", "Decimal", 1),
        ("builtins", "complex", 2),
        ("builtins", "bool", 1),
        ("decimal", "Decimal", 1),
        ("builtins", "str", 1),
    ]
    assert X["a"].map(baseline).tolist() == [1, 1, 2, 2, 1, 1, 1]

    encoder = CountEncoder(cols=["a"], n_jobs=n_jobs, parallel_backend=backend).fit(X)
    learned = [item for item in _missing_key_signature(encoder.mapping_["a"]) if item[2] != "'__UNKNOWN__'"]
    assert Counter(learned[:6]) == Counter(baseline_signature)
    assert encoder.transform(X)["a"].tolist() == [1, 1, 2, 2, 1, 1, 1]


@pytest.mark.parametrize(
    "n_jobs,backend",
    [(1, None), (2, "threading"), (2, "loky")],
)
@pytest.mark.parametrize("handle_missing", ["value", "return_nan", "error"])
@pytest.mark.parametrize("handle_unknown", ["value", "return_nan", "error"])
def test_count_typed_nan_obeys_missing_policy_before_unknown_policy(
    n_jobs,
    backend,
    handle_missing,
    handle_unknown,
):
    """typed NaN 仍泄露计数或回落 unknown 策略时，本矩阵应失败。"""
    X = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    float("nan"),
                    np.float32("nan"),
                    float("nan"),
                    np.float64("nan"),
                    None,
                    pd.NaT,
                    pd.NA,
                    "x",
                ],
                dtype=object,
            )
        }
    )
    kwargs = {
        "cols": ["a"],
        "handle_missing": handle_missing,
        "handle_unknown": handle_unknown,
        "n_jobs": n_jobs,
        "parallel_backend": backend,
    }

    if handle_missing == "error":
        with pytest.raises(ValueError, match="列'a'包含缺失值"):
            CountEncoder(**kwargs).fit(X)

        encoder = CountEncoder(**kwargs).fit(pd.DataFrame({"a": ["x", "x"]}))
        for typed_nan in (float("nan"), np.float32("nan"), np.float64("nan")):
            probe = pd.DataFrame({"a": pd.Series([typed_nan], dtype=object)})
            with pytest.raises(ValueError, match="列'a'包含缺失值"):
                encoder.transform(probe)
        return

    encoder = CountEncoder(**kwargs).fit(X)
    typed_signature = _typed_float_nan_signature(encoder.mapping_["a"])
    assert [(module, name) for module, name, _ in typed_signature] == [
        ("builtins", "float"),
        ("numpy", "float32"),
        ("numpy", "float64"),
    ]
    if handle_missing == "value":
        assert [value for _, _, value in typed_signature] == [2, 1, 1]
    else:
        assert all(pd.isna(value) for _, _, value in typed_signature)

    def assert_policy(candidate):
        transformed = candidate.transform(X)["a"].tolist()
        if handle_missing == "value":
            assert transformed[:4] == [2, 1, 2, 1]
        else:
            assert all(pd.isna(value) for value in transformed[:4])
        assert transformed[4:] == [1, 1, 1, 1]

        unseen_typed = pd.DataFrame({"a": pd.Series([np.float16("nan")], dtype=object)})
        unseen_value = candidate.transform(unseen_typed)["a"].iloc[0]
        if handle_missing == "value":
            assert unseen_value == 0
        else:
            assert pd.isna(unseen_value)

        unknown = pd.DataFrame({"a": ["未训练类别"]})
        if handle_unknown == "error":
            with pytest.raises(ParallelExecutionError, match="列'a'包含未知类别"):
                candidate.transform(unknown)
        else:
            unknown_value = candidate.transform(unknown)["a"].iloc[0]
            if handle_unknown == "value":
                assert unknown_value == 0
            else:
                assert pd.isna(unknown_value)

    assert_policy(encoder)
    exported = pickle.loads(pickle.dumps(encoder.export_mapping()))
    assert_policy(CountEncoder(cols=["a"]).import_mapping(exported))
    assert_policy(pickle.loads(pickle.dumps(encoder)))


def test_target_noise_fixed_seed_matches_thread_and_loky(encoder_xy):
    X, y = encoder_xy
    columns = ["a", "b"]
    serial = TargetEncoder(cols=columns, noise=0.1, random_state=42, n_jobs=1).fit(X, y)
    expected = serial.transform(X, y)
    for backend in ("threading", "loky"):
        parallel = TargetEncoder(
            cols=columns, noise=0.1, random_state=42, n_jobs=2, parallel_backend=backend
        ).fit(X, y)
        pd.testing.assert_frame_equal(expected, parallel.transform(X, y), check_exact=True)


@pytest.mark.parametrize("cls", [TargetEncoder, CatBoostEncoder])
def test_none_random_state_uses_local_rng_without_consuming_global_state(cls, encoder_xy):
    X, y = encoder_xy
    kwargs = {"noise": 0.1} if cls is TargetEncoder else {"sigma": 0.1}
    encoder = cls(cols=["a", "b"], random_state=None, n_jobs=2, parallel_backend="threading", **kwargs).fit(X, y)
    np.random.seed(20260805)
    before = np.random.get_state()
    encoder.transform(X, y)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_first_failed_fit_has_no_partial_state():
    X = pd.DataFrame({"好列": ["a", "b"], "坏列": [1, 2]})
    encoder = _FailingCountEncoder(cols=["好列", "坏列"], n_jobs=2, parallel_backend="threading")
    with pytest.raises(ParallelExecutionError, match="坏列"):
        encoder.fit(X)
    assert encoder.mapping_ == {}
    assert encoder.cols_ is None
    assert encoder._is_fitted is False


class _FinalizeFailingCountEncoder(CountEncoder):
    def _fit(self, X, y=None):
        super()._fit(X, y)
        if "最终列" in (self.cols_ or []):
            raise RuntimeError("最终提交前失败")


def test_finalize_failure_preserves_previous_complete_model():
    X = pd.DataFrame({"a": ["x", "x", "y"], "最终列": [1, 2, 3]})
    encoder = _FinalizeFailingCountEncoder(cols=["a"], n_jobs=2, parallel_backend="threading").fit(X)
    before = encoder.export_mapping()
    encoder.cols = ["a", "最终列"]
    with pytest.raises(RuntimeError, match="最终提交前失败"):
        encoder.fit(X)
    assert encoder.export_mapping()["mapping_"] == before["mapping_"]


def test_loky_refit_failure_preserves_previous_complete_model():
    X = pd.DataFrame({"好列": ["a", "a", "b"], "坏列": [1, 2, 3]})
    encoder = _FailingCountEncoder(cols=["好列"], n_jobs=2, parallel_backend="loky").fit(X)
    before = encoder.export_mapping()["mapping_"]
    encoder.cols = ["好列", "坏列"]
    with pytest.raises(ParallelExecutionError, match="坏列"):
        encoder.fit(X)
    assert encoder.export_mapping()["mapping_"] == before


def test_successful_fit_preserves_mutable_parameter_identity():
    ordinal_mapping = {"a": {"x": 1, "y": 2}}
    ordinal = OrdinalEncoder(cols=["a"], mapping=ordinal_mapping, n_jobs=2).fit(
        pd.DataFrame({"a": ["x", "y"]})
    )
    assert ordinal.mapping is ordinal_mapping

    special_values = ["特殊"]
    cardinality = CardinalityEncoder(cols=["a"], special_values=special_values, n_jobs=2).fit(
        pd.DataFrame({"a": ["特殊", "x", "y"]})
    )
    assert cardinality.special_values is special_values


class _FakeRiskModel:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, **kwargs):
        return self


def _gbm_child_worker(task):
    model_type, public_n_jobs, model_params = task
    encoder = GBMEncoder(
        model_type=model_type,
        n_estimators=2,
        n_jobs=public_n_jobs,
        model_params=model_params,
    )
    encoder.classes_ = np.asarray([0, 1])
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    getattr(encoder, f"_fit_{model_type}")(X, y)
    key = "thread_count" if model_type == "catboost" else "n_jobs"
    return encoder.model_.params[key]


@pytest.mark.parametrize(
    "model_type,model_name,worker_key",
    [
        ("xgboost", "XGBoostRiskModel", "n_jobs"),
        ("lightgbm", "LightGBMRiskModel", "n_jobs"),
        ("catboost", "CatBoostRiskModel", "thread_count"),
    ],
)
def test_gbm_children_use_active_nested_budget(monkeypatch, model_type, model_name, worker_key):
    import hscredit.core.models as models

    monkeypatch.setattr(models, model_name, _FakeRiskModel)
    explicit = {worker_key: 99}
    task = (model_type, -1, explicit)
    token = _ACTIVE_BUDGET.set(ParallelBudget(4, 0))
    try:
        multi = parallel_execute(
            _gbm_child_worker,
            [task, task],
            n_jobs=2,
            parallel_backend="threading",
            has_parallel_children=True,
        )
        single = parallel_execute(
            _gbm_child_worker,
            [task],
            n_jobs=1,
            parallel_backend="threading",
            has_parallel_children=True,
        )
        explicit_one = parallel_execute(
            _gbm_child_worker,
            [(model_type, 99, {worker_key: 1})],
            n_jobs=1,
            parallel_backend="threading",
            has_parallel_children=True,
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert multi == [2, 2]
    assert single == [4]
    assert explicit_one == [1]
    assert explicit == {worker_key: 99}


@pytest.mark.parametrize(
    "model_type,model_name,worker_key",
    [
        ("xgboost", "XGBoostRiskModel", "n_jobs"),
        ("lightgbm", "LightGBMRiskModel", "n_jobs"),
        ("catboost", "CatBoostRiskModel", "thread_count"),
    ],
)
def test_gbm_top_level_explicit_workers_and_model_params_identity(monkeypatch, model_type, model_name, worker_key):
    import hscredit.core.models as models

    monkeypatch.setattr(models, model_name, _FakeRiskModel)
    params = {worker_key: 3, "自定义": "保留"}
    encoder = GBMEncoder(model_type=model_type, n_jobs=1, model_params=params)
    encoder.classes_ = np.asarray([0, 1])
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0, 1])
    getattr(encoder, f"_fit_{model_type}")(X, y)
    assert encoder.model_.params[worker_key] == 3
    assert encoder.model_params is params
    assert params == {worker_key: 3, "自定义": "保留"}


def test_xgboost_serial_thread_loky_raw_model_parity():
    pytest.importorskip("xgboost")
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + X["b"] * 0.25 > 0).astype(int))

    def factory(n_jobs, backend):
        return GBMEncoder(
            cols=list(X.columns),
            model_type="xgboost",
            n_estimators=5,
            max_depth=2,
            output_type="leaves",
            random_state=42,
            n_jobs=n_jobs,
            parallel_backend=backend,
        )

    serial = factory(1, None).fit(X, y)
    expected = serial.transform(X)
    expected_raw = bytes(serial.model_._model.get_booster().save_raw())
    for backend in ("threading", "loky"):
        parallel = factory(2, backend).fit(X, y)
        pd.testing.assert_frame_equal(expected, parallel.transform(X), check_exact=True)
        assert bytes(parallel.model_._model.get_booster().save_raw()) == expected_raw
