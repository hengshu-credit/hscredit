"""统一目标参数、预定义指标口径与约束兼容策略。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import OptimalBinning, OptimalBinning2D, CustomObjectives, ORBinning
from hscredit.core.binning._objectives import evaluate_callback, resolve_objective
from hscredit.core.metrics import compute_bin_stats
from hscredit.exceptions import ParallelExecutionError


@pytest.fixture
def sample():
    X = pd.DataFrame({"数值": np.repeat(np.arange(6, dtype=float), 20)})
    y = pd.Series(np.concatenate([np.r_[np.ones(k), np.zeros(20 - k)] for k in [2, 4, 6, 10, 14, 18]]).astype(int))
    return X, y


def settings(method):
    return (
        {"population_size": 10, "generations": 4}
        if method == "genetic"
        else {"n_prebins": 6, "max_candidates": 8, "time_limit": 5}
    )


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
@pytest.mark.parametrize("name", CustomObjectives.names())
def test_named_and_callable_objectives_select_identical_rules(method, name, sample):
    X, y = sample
    common = dict(
        method=method, max_n_bins=3, min_bin_size=0.05, monotonic=False, random_state=42, n_jobs=1, **settings(method)
    )
    params = {"iv_weight": 0.8}
    named = OptimalBinning(objective=name, objective_params=params, **common).fit(X, y)
    callback = OptimalBinning(objective=CustomObjectives.get(name, **params), **common).fit(X, y)
    np.testing.assert_array_equal(named.splits_["数值"], callback.splits_["数值"])
    assert params == {"iv_weight": 0.8}
    assert clone(named).objective_params == params
    assert name in OptimalBinning.available_objectives(method)


@pytest.mark.parametrize("name", CustomObjectives.names())
def test_composite_iv_is_positive_and_matches_independent_formula(name):
    # 两箱各 5 人，好坏数分别为 (4,1)、(1,4)。总体好坏各 5 人。
    bins = np.repeat([0, 1], 5)
    y = np.asarray([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    weight = "ks_weight" if name == "max_ks_iv" else "variance_weight" if name == "woe_variance_iv" else "lift_weight"
    callback = CustomObjectives.get(name, **{weight: 0.0, "iv_weight": 1.0})
    expected = 2 * (0.8 - 0.2) * np.log(0.8 / 0.2)
    score = evaluate_callback(callback, bins, y)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(compute_bin_stats(bins, y, round_digits=False)["分档IV值"].sum())


def test_callback_rows_keep_counts_rates_and_chinese_metric_values():
    bins = np.repeat([0, 1], 5)
    y = np.asarray([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    def score(rows, total_good, total_bad, penalty):
        assert (total_good, total_bad) == (5, 5)
        assert rows[0]["count"] == rows[0]["样本总数"] == 5
        assert rows[0]["bad_rate"] == pytest.approx(0.2)
        assert rows[0]["bad_dist"] == pytest.approx(0.2)
        assert rows[0]["woe"] == pytest.approx(np.log(0.2 / 0.8))
        assert rows[0]["iv"] == rows[0]["分档IV值"]
        return sum(row["iv"] for row in rows) - penalty

    _, callback = resolve_objective(score, {"penalty": 0.25})
    assert evaluate_callback(callback, bins, y) == pytest.approx(1.2 * np.log(4) - 0.25)


@pytest.mark.parametrize("method", ["or_tools", "cp_sat"])
def test_custom_objective_can_choose_single_or_multiple_bins(method, sample):
    X, y = sample

    def prefer_count(rows, total_good, total_bad, desired):
        return -abs(len(rows) - desired)

    for desired in [1, 3]:
        binner = OptimalBinning(
            method=method,
            objective=prefer_count,
            objective_params={"desired": desired},
            min_n_bins=1,
            max_n_bins=3,
            n_jobs=1,
            monotonic=False,
            **settings(method),
        ).fit(X, y)
        assert binner.n_bins_["数值"] == desired
        assert binner.solver_info_["数值"]["后端"] == "enumeration"


@pytest.mark.parametrize("method", OptimalBinning.VALID_METHODS)
def test_legacy_preserves_native_results_for_impossible_direction(method, sample):
    X, y = sample
    extra = settings(method) if method in {"or_tools", "cp_sat", "genetic"} else {}
    binner = OptimalBinning(
        method=method,
        min_n_bins=3,
        max_n_bins=3,
        monotonic="descending",
        constraint_policy="legacy",
        n_jobs=1,
        random_state=42,
        **extra,
    ).fit(X, y)
    assert binner.constraint_violations_.get("数值")
    assert len(binner.transform(X)) == len(X)
    assert clone(binner).constraint_policy == "legacy"
    assert not any("状态" in col or "原因" in col for col in binner.get_bin_table("数值").columns)


@pytest.mark.parametrize("method", ["tree", "cp_sat", "or_tools", "quantile"])
def test_raise_still_rejects_impossible_constraints(method, sample):
    X, y = sample
    extra = settings(method) if method in {"or_tools", "cp_sat"} else {}
    with pytest.raises((ValueError, ParallelExecutionError), match="约束|单调"):
        OptimalBinning(
            method=method,
            min_n_bins=3,
            max_n_bins=3,
            monotonic="descending",
            constraint_policy="raise",
            n_jobs=1,
            **extra,
        ).fit(X, y)


def test_legacy_categorical_and_2d_constraints_preserve_population():
    X = pd.DataFrame({"类别": ["甲"] * 90 + ["乙"] * 10})
    y = pd.Series([0, 1] * 50)
    model = OptimalBinning(method="quantile", min_bin_size=0.2, constraint_policy="legacy", n_jobs=1).fit(X, y)
    assert model.constraint_violations_["类别"]
    assert model.get_bin_table("类别")["样本总数"].sum() == 100
    two = OptimalBinning2D(min_bin_size=1000, constraint_policy="legacy", n_jobs=1).fit(
        pd.DataFrame({"甲": np.arange(20.0), "乙": np.arange(20.0)}), pd.Series([0, 1] * 10)
    )
    assert two.constraint_violations_
    assert two.binner_x_.constraint_policy == two.binner_y_.constraint_policy == "legacy"
    assert len(two.transform(pd.DataFrame({"甲": np.arange(20.0), "乙": np.arange(20.0)}))) == 20


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
def test_callback_errors_are_not_relaxed_by_legacy_policy(method, sample):
    X, y = sample

    def broken(rows, total_good, total_bad):
        raise RuntimeError("回调故意失败")

    with pytest.raises((ValueError, ParallelExecutionError), match="回调故意失败"):
        OptimalBinning(
            method=method, objective=broken, constraint_policy="legacy", n_jobs=1, random_state=42, **settings(method)
        ).fit(X, y)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, [1.0]])
def test_objective_must_return_finite_scalar(value):
    with pytest.raises(ValueError, match="有限实数"):
        evaluate_callback(lambda rows, good, bad: value, [0, 0, 1, 1], [0, 1, 0, 1])


def test_invalid_parameters_and_fixed_objective_methods_are_explicit():
    with pytest.raises(ValueError, match="objective_params"):
        ORBinning(objective="max_lift_iv", objective_params={"typo": 1})
    with pytest.raises(ValueError, match="固定算法目标"):
        OptimalBinning(method="tree", objective="iv")
    with pytest.raises(ValueError, match="constraint_policy"):
        OptimalBinning(constraint_policy="ignore")
    with pytest.raises(ValueError, match="custom_objective"):
        ORBinning(objective="iv", custom_objective=lambda s, g, b: 1.0)


def test_original_custom_objective_style_remains_supported(sample):
    X, y = sample
    callback = CustomObjectives.max_lift_iv(lift_weight=0.2)
    a = ORBinning(objective="custom", custom_objective=callback, n_jobs=1, **settings("or_tools")).fit(X, y)
    b = ORBinning(objective=callback, n_jobs=1, **settings("or_tools")).fit(X, y)
    np.testing.assert_array_equal(a.splits_["数值"], b.splits_["数值"])


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
def test_direct_classes_clone_persist_and_accept_callable_parameters(method, sample):
    import cloudpickle

    X, y = sample
    cls = OptimalBinning._method_classes()[method]
    obj = cls(
        objective="max_lift_iv",
        objective_params={"iv_weight": 0.7},
        constraint_policy="legacy",
        max_n_bins=3,
        n_jobs=1,
        random_state=42,
        **settings(method),
    ).fit(X, y)
    restored = cloudpickle.loads(cloudpickle.dumps(obj))
    pd.testing.assert_frame_equal(obj.transform(X), restored.transform(X))
    assert clone(obj).objective_params == {"iv_weight": 0.7}
    restored.set_params(
        objective=lambda rows, good, bad, penalty: sum(s["iv"] for s in rows) - penalty,
        objective_params={"penalty": 0.1},
    )
    restored.fit(X, y)
    assert len(restored.transform(X)) == len(X)


def test_objective_callback_respects_woe_clip_for_genetic(sample):
    X, y = sample

    def clipped(rows, good, bad):
        assert max(abs(s["woe"]) for s in rows) <= 0.2 + 1e-12
        return sum(s["iv"] for s in rows)

    model = OptimalBinning(
        method="genetic", objective=clipped, woe_clip=0.2, random_state=42, n_jobs=1, **settings("genetic")
    ).fit(X, y)
    assert model._binner.woe_clip == 0.2


def test_new_parameters_have_safe_defaults_for_old_pickled_state():
    binner = ORBinning()
    del binner.constraint_policy
    del binner.objective_params
    assert clone(binner).constraint_policy == "raise"
    assert clone(binner).objective_params is None


def test_legacy_does_not_relax_input_or_explicit_bad_rate_validation(sample):
    X, y = sample
    with pytest.raises(ValueError, match="min_n_bins"):
        OptimalBinning(min_n_bins=5, max_n_bins=2, constraint_policy="legacy")
    with pytest.raises((ValueError, ParallelExecutionError), match="约束|可行"):
        OptimalBinning(
            method="target_bad_rate",
            target_bad_rates=[0.0, 1.0],
            strict_mode=True,
            merge_empty_bins=False,
            constraint_policy="legacy",
            n_jobs=1,
        ).fit(X, y)


def test_two_dimensional_axis_accepts_registered_objective_options(sample):
    X, y = sample
    two_x = pd.DataFrame({"数值": X["数值"], "另一轴": np.arange(len(X), dtype=float)})
    binner = OptimalBinning2D(
        max_n_bins=3,
        n_jobs=1,
        x_params={
            "method": "cp_sat",
            "objective": "max_lift_iv",
            "objective_params": {"lift_weight": 0.2},
            "n_prebins": 6,
            "time_limit": 5,
        },
    ).fit(two_x, y)
    assert binner.binner_x_.objective == "max_lift_iv"
    assert binner.binner_x_._binner.n_prebins == 6


def test_legacy_keeps_fixed_groups_when_count_cap_cannot_be_met():
    X = pd.DataFrame({"类别": list("AAAABBBBCCCCDDDD")})
    y = pd.Series([0, 1] * 8)
    model = OptimalBinning(
        method="quantile",
        user_splits={"类别": [["A"], ["B"], ["C"], ["D"]]},
        user_splits_fixed={"类别": [True, False, False, True]},
        max_n_bins=2,
        constraint_policy="legacy",
        n_jobs=1,
    ).fit(X, y)
    assert model.splits_["类别"] == [["A"], ["B", "C"], ["D"]]
    assert model.constraint_violations_["类别"]


def test_objective_and_constraint_options_reach_prebinning(sample):
    X, y = sample
    obj = OptimalBinning(
        method="cp_sat",
        objective="max_ks_iv",
        objective_params={"ks_weight": 0.2},
        prebinning="quantile",
        prebinning_params={"max_n_bins": 6},
        constraint_policy="legacy",
        max_n_bins=3,
        monotonic="descending",
        n_jobs=1,
        time_limit=2,
    ).fit(X, y)
    assert obj._prebinner.constraint_policy == "legacy"
    assert obj._binner._objective_name == "max_ks_iv"
    assert obj.constraint_violations_


@pytest.mark.parametrize("cuts", [[], [0.0, 1.0], [1.0, 1.0, 2.0], [0.5, 1.5]])
def test_prefix_objective_context_matches_raw_row_statistics(cuts):
    from hscredit.core.binning._objectives import make_objective_scorer

    x = np.asarray([2.0, 0.0, 1.0, 2.0, 0.0, 1.0])
    y = np.asarray([1, 0, 1, 0, 1, 0])
    seen = []

    def objective(rows, good, bad):
        seen.append(rows)
        return sum(s["iv"] for s in rows)

    raw = evaluate_callback(objective, np.searchsorted(cuts, x, side="right"), y, woe_clip=0.3)
    counted = make_objective_scorer(objective, x, y, woe_clip=0.3)(cuts)
    assert raw == pytest.approx(counted)
    pd.testing.assert_frame_equal(pd.DataFrame(seen[0]), pd.DataFrame(seen[1]))


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
@pytest.mark.parametrize("named", [False, True])
def test_artifact_roundtrip_preserves_named_and_local_callbacks(method, named, sample, tmp_path):
    X, y = sample
    objective = "max_lift_iv" if named else lambda rows, good, bad: sum(s["iv"] for s in rows)
    model = OptimalBinning(
        method=method,
        objective=objective,
        constraint_policy="legacy",
        max_n_bins=3,
        random_state=42,
        n_jobs=1,
        **settings(method),
    ).fit(X, y)
    path = tmp_path / f"{method}.joblib"
    model.save_artifact(path)
    restored = OptimalBinning.load_artifact(path, trusted=True)
    pd.testing.assert_frame_equal(model.transform(X), restored.transform(X))
    assert restored.constraint_policy == "legacy"
    if not named:
        assert callable(restored.objective)
        clone(restored).fit(X, y)


def test_two_dimensional_artifact_keeps_callable_axis_config(sample, tmp_path):
    X, y = sample
    both = pd.DataFrame({"甲": X["数值"], "乙": np.arange(len(X), dtype=float)})
    score = lambda rows, good, bad: sum(s["iv"] for s in rows)
    model = OptimalBinning2D(
        max_n_bins=3, n_jobs=1, x_params={"method": "cp_sat", "objective": score, "n_prebins": 6, "time_limit": 5}
    ).fit(both, y)
    path = tmp_path / "two.joblib"
    model.save_artifact(path)
    restored = OptimalBinning2D.load_artifact(path, trusted=True)
    assert callable(restored.x_params["objective"])
    assert restored._X is None and restored._y is None
    pd.testing.assert_frame_equal(model.transform(both), restored.transform(both))


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
def test_custom_objective_serial_and_process_parallel_are_consistent(method, sample):
    X, y = sample
    frame = pd.DataFrame({"甲": X["数值"], "乙": X["数值"] + 100.0})
    params = dict(
        method=method,
        objective="max_ks_iv",
        objective_params={"ks_weight": 0.3},
        max_n_bins=3,
        random_state=42,
        **settings(method),
    )
    serial = OptimalBinning(n_jobs=1, **params).fit(frame, y)
    parallel = OptimalBinning(n_jobs=2, parallel_backend="loky", **params).fit(frame, y)
    pd.testing.assert_frame_equal(serial.transform(frame), parallel.transform(frame))


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
def test_constant_feature_still_evaluates_its_single_feasible_partition(method):
    seen = []

    def score(rows, good, bad):
        seen.append(len(rows))
        return len(rows)

    X = pd.DataFrame({"数值": [1.0] * 20})
    y = pd.Series([0, 1] * 10)
    model = OptimalBinning(method=method, objective=score, n_jobs=1, random_state=42, **settings(method)).fit(X, y)
    assert seen and set(seen) == {1}
    assert model.n_bins_["数值"] == 1


@pytest.mark.parametrize("method", ["or_tools", "cp_sat", "genetic"])
def test_single_class_ordinary_rows_do_not_silently_bypass_custom_objective(method):
    seen = []

    def score(rows, good, bad):
        seen.append((good, bad))
        return -abs(len(rows) - 3)

    X = pd.DataFrame({"数值": [*np.arange(10.0), 99.0]})
    y = pd.Series([0] * 10 + [1])
    model = OptimalBinning(
        method=method,
        objective=score,
        min_n_bins=1,
        max_n_bins=3,
        min_bin_size=1,
        special_codes=[99.0],
        n_jobs=1,
        random_state=42,
        **settings(method),
    ).fit(X, y)
    assert seen and set(seen) == {(10, 0)}
    assert model.n_bins_["数值"] == 3
