"""具体分箱器的用户规则和并行分派回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import (
    BestIVBinning,
    BestKSBinning,
    BestLiftBinning,
    CartBinning,
    ChiMergeBinning,
    CPSATBinning,
    GeneticBinning,
    KernelDensityBinning,
    KMeansBinning,
    MDLPBinning,
    MonotonicBinning,
    OptimalBinning,
    ORBinning,
    QuantileBinning,
    SmoothBinning,
    TargetBadRateBinning,
    TreeBinning,
    UniformBinning,
)


DIRECT_BINNER_CLASSES = [
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    CartBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    ORBinning,
    CPSATBinning,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
]


def _method_specific_params(binner_cls):
    if binner_cls is GeneticBinning:
        return {"population_size": 8, "generations": 2}
    if binner_cls is KernelDensityBinning:
        return {"n_grid_points": 100}
    if binner_cls in {ORBinning, CPSATBinning}:
        return {"time_limit": 2, "num_workers": 1}
    return {}


@pytest.fixture
def numeric_data():
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(
        {
            "fixed": np.arange(120, dtype=float),
            "ordinary": rng.normal(size=120),
        }
    )
    y = pd.Series((X["ordinary"] + rng.normal(scale=0.4, size=len(X)) > 0).astype(int))
    return X, y


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES)
def test_direct_binners_apply_strict_rules_and_fit_ordinary_fields(binner_cls, numeric_data):
    X, y = numeric_data
    expected = np.array([20.123456, 60.5, 100.75])
    try:
        binner = binner_cls(
            user_splits={"fixed": expected.tolist()},
            strict_user_splits=True,
            max_n_bins=5,
            min_n_bins=2,
            random_state=7,
            n_jobs=2,
            parallel_backend="threading",
            **_method_specific_params(binner_cls),
        ).fit(X, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    np.testing.assert_array_equal(binner.splits_["fixed"], expected)
    assert "ordinary" in binner.splits_
    assert "ordinary" in binner.bin_tables_


def test_rule_and_ordinary_fields_are_submitted_in_one_parallel_batch(monkeypatch, numeric_data):
    X, y = numeric_data
    binner = BestIVBinning(
        user_splits={"fixed": [20.0, 60.0, 100.0]},
        strict_user_splits=True,
        n_jobs=2,
    )
    original = BestIVBinning._parallel_execute
    submissions = []

    def recording_execute(self, function, tasks, **kwargs):
        task_list = list(tasks)
        submissions.append([(task[0], task[3]) for task in task_list])
        return original(self, function, task_list, **kwargs)

    monkeypatch.setattr(BestIVBinning, "_parallel_execute", recording_execute)
    binner.fit(X, y)

    assert submissions == [
        [
            ("fixed", "_fit_common_user_split_feature"),
            ("ordinary", "_fit_feature"),
        ]
    ]


def test_direct_user_splits_match_serial_threading_and_loky(numeric_data):
    X, y = numeric_data
    common = {
        "user_splits": {"fixed": [20.0, 60.0, 100.0]},
        "strict_user_splits": True,
        "max_n_bins": 5,
        "random_state": 11,
    }
    models = [
        BestIVBinning(**common, n_jobs=1).fit(X, y),
        BestIVBinning(**common, n_jobs=3, parallel_backend="threading").fit(X, y),
        BestIVBinning(**common, n_jobs=3, parallel_backend="loky").fit(X, y),
    ]

    for feature in X.columns:
        for model in models[1:]:
            np.testing.assert_allclose(models[0].splits_[feature], model.splits_[feature], rtol=0, atol=0)
            pd.testing.assert_frame_equal(models[0].bin_tables_[feature], model.bin_tables_[feature])


def test_non_strict_rules_filter_round_and_user_splits_take_priority(numeric_data):
    X, y = numeric_data
    binner = QuantileBinning(
        user_splits={"fixed": [-99.0, 20.123456, 999.0]},
        split_points={"fixed": [50.0], "ordinary": [0.0]},
        strict_user_splits=False,
        decimal=3,
        n_jobs=2,
    ).fit(X, y)

    np.testing.assert_array_equal(binner.splits_["fixed"], np.array([20.123]))
    np.testing.assert_array_equal(binner.splits_["ordinary"], np.array([0.0]))


def test_optimal_binning_supports_split_points_alias(numeric_data):
    X, y = numeric_data
    expected = np.array([20.123456, 60.5, 100.75])
    binner = OptimalBinning(
        method="best_iv",
        split_points={"fixed": expected.tolist()},
        strict_user_splits=True,
        n_jobs=2,
    ).fit(X, y)

    np.testing.assert_array_equal(binner.splits_["fixed"], expected)
    assert "ordinary" in binner.splits_


@pytest.mark.parametrize("binner_cls", [UniformBinning, BestIVBinning])
def test_direct_binners_support_strict_and_non_strict_categorical_rules(binner_cls):
    X = pd.DataFrame({"category": ["a"] * 30 + ["b"] * 30 + ["c"] * 30 + ["d"] * 30})
    y = pd.Series([0] * 25 + [1] * 5 + [0] * 20 + [1] * 10 + [0] * 10 + [1] * 20 + [0] * 5 + [1] * 25)
    groups = [["a"], ["b"], ["c"], ["d"]]

    strict = binner_cls(
        user_splits={"category": groups},
        strict_user_splits=True,
        max_n_bins=4,
        min_n_bins=2,
        n_jobs=2,
    ).fit(X, y)
    assert strict.splits_["category"] == groups

    non_strict = binner_cls(
        user_splits={"category": groups},
        strict_user_splits=False,
        max_n_bins=2,
        min_n_bins=2,
        n_jobs=2,
    ).fit(X, y)
    assert non_strict.splits_["category"] == [["a", "b"], ["c", "d"]]
