"""分箱器公共并行配置与结果一致性测试。"""

import inspect

import hscredit.core.binning.optimal_binning as optimal_binning_module
import numpy as np
import pandas as pd
from sklearn.base import clone

from hscredit.core import binning
from hscredit.core.binning import BaseBinning, OptimalBinning, OptimalBinning2D, UniformBinning
from hscredit.core.metrics import compute_bin_stats


PARALLEL_PARAMETERS = {
    "n_jobs": -1,
    "parallel_backend": None,
    "parallel_config": None,
}


def test_all_exported_binners_expose_parallel_parameters():
    missing = []
    wrong_defaults = []
    for name in binning.__all__:
        cls = getattr(binning, name)
        if inspect.isclass(cls) and issubclass(cls, BaseBinning):
            params = inspect.signature(cls.__init__).parameters
            absent = set(PARALLEL_PARAMETERS) - set(params)
            if absent:
                missing.append(name)
                continue
            for parameter, expected in PARALLEL_PARAMETERS.items():
                if params[parameter].default != expected:
                    wrong_defaults.append((name, parameter, params[parameter].default))

    assert missing == []
    assert wrong_defaults == []


def test_optimal_binning_2d_exposes_parallel_parameters():
    params = inspect.signature(OptimalBinning2D.__init__).parameters
    assert {name: params[name].default for name in PARALLEL_PARAMETERS} == PARALLEL_PARAMETERS


def test_all_exported_binners_store_parallel_parameters():
    unavailable = []
    for name in binning.__all__:
        cls = getattr(binning, name)
        if not inspect.isclass(cls) or not (issubclass(cls, BaseBinning) or cls is OptimalBinning2D):
            continue
        if inspect.isabstract(cls):
            continue
        try:
            binner = cls(
                n_jobs=2,
                parallel_backend="threading",
                parallel_config={"batch_size": 1},
            )
        except ImportError:
            unavailable.append(name)
            continue
        params = binner.get_params(deep=False)
        assert params["n_jobs"] == 2, name
        assert params["parallel_backend"] == "threading", name
        assert params["parallel_config"] == {"batch_size": 1}, name

    assert set(unavailable) <= {"ORBinning", "CPSATBinning"}


def test_numpy_int8_category_sort_key_does_not_overflow():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    bins = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    result = compute_bin_stats(bins, y)
    assert not result.empty


def test_uniform_parallel_parameters_survive_sklearn_clone():
    binner = UniformBinning(
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )
    cloned = clone(binner)

    assert cloned.n_jobs == 2
    assert cloned.parallel_backend == "threading"
    assert cloned.parallel_config == {"batch_size": 1}


def test_optimal_binning_forwards_parallel_parameters_to_method_binner():
    X = pd.DataFrame(
        {
            "数值一": [1, 2, 3, 4, 5, 6, 7, 8],
            "数值二": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    binner = OptimalBinning(
        method="uniform",
        n_jobs=1,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    ).fit(X, y)

    assert binner._binner.n_jobs == 1
    assert binner._binner.parallel_backend == "threading"
    assert binner._binner.parallel_config == {"batch_size": 1}


def test_prebinning_empty_splits_fallback_inherits_parallel_parameters(monkeypatch):
    X = pd.DataFrame({"数值": [1, 2, 3, 4, 5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    parent = OptimalBinning(
        method="best_iv",
        max_n_bins=3,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )
    created = []

    class RecordingOptimalBinning(OptimalBinning):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(optimal_binning_module, "OptimalBinning", RecordingOptimalBinning)

    parent._fit_with_method_and_prebins(X, y, pre_splits={})

    assert len(created) == 1
    fallback = created[0]
    assert fallback.n_jobs == 2
    assert fallback.parallel_backend == "threading"
    assert fallback.parallel_config == {"batch_size": 1}


def test_optimal_binning_2d_forwards_parallel_parameters_to_axis_binners():
    X = pd.DataFrame(
        {
            "横轴": [1, 2, 3, 4, 5, 6, 7, 8],
            "纵轴": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    binner = OptimalBinning2D(
        n_jobs=1,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    ).fit(X, y)

    for axis_binner in (binner.binner_x_, binner.binner_y_):
        assert axis_binner.n_jobs == 1
        assert axis_binner.parallel_backend == "threading"
        assert axis_binner.parallel_config == {"batch_size": 1}
