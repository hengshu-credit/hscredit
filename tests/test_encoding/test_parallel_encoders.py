"""编码器列级并行的契约与串并行一致性测试。"""

import inspect
import pickle

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
