"""分箱器严格公共契约的回归测试。"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import (
    BestIVBinning,
    BestKSBinning,
    BestLiftBinning,
    CPSATBinning,
    CartBinning,
    ChiMergeBinning,
    GeneticBinning,
    KMeansBinning,
    KernelDensityBinning,
    MDLPBinning,
    MonotonicBinning,
    ORBinning,
    OptimalBinning,
    QuantileBinning,
    SmoothBinning,
    TargetBadRateBinning,
    TreeBinning,
    UniformBinning,
)
from hscredit.core.binning._categorical import assign_category_groups
from hscredit.core.binning._contracts import (
    MISSING_BIN,
    SPECIAL_BIN,
    UNKNOWN_BIN,
    parse_numerical_user_splits,
    resolve_user_splits_fixed,
    validate_handle_unknown,
)
from hscredit.core.binning._candidate_search import search_candidate_splits


DIRECT_BINNER_CLASSES = [
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    ORBinning,
    CPSATBinning,
    CartBinning,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
]


@pytest.mark.parametrize(
    ("values", "expected_splits", "expected_missing_bin"),
    [
        ([np.nan, 10, 20], [10.0, 20.0], 0),
        ([10, np.nan, 20], [10.0, 20.0], 1),
        ([10, 20, np.nan], [10.0, 20.0], 2),
        ([10, 20], [10.0, 20.0], None),
    ],
)
def test_numeric_missing_marker_position_selects_ordinary_bin(values, expected_splits, expected_missing_bin):
    """删除位置解析会使手工指定的数值缺失箱归属丢失。"""
    splits, missing_bin = parse_numerical_user_splits("age", values)

    np.testing.assert_array_equal(splits, expected_splits)
    assert missing_bin == expected_missing_bin


@pytest.mark.parametrize(
    "values",
    [
        [20, 10],
        [10, 10],
        [10, np.inf],
        [np.nan, None, 10],
    ],
)
def test_numeric_user_splits_reject_ambiguous_rules(values):
    """乱序、重复、非有限切点或多个缺失标记不能静默改变规则。"""
    with pytest.raises(ValueError, match="切分点|缺失标记"):
        parse_numerical_user_splits("age", values)


def test_fixed_mask_excludes_numeric_missing_marker():
    """数值缺失位置不是切分节点，不能占用固定掩码位置。"""
    masks = resolve_user_splits_fixed(
        {"age": [10, np.nan, 20]},
        {"age": [True, False]},
    )

    assert masks == {"age": [True, False]}


def test_fixed_mask_supports_global_and_per_feature_forms():
    """删除任一种正式形式都会破坏全局、字段级或节点级固定能力。"""
    rules = {
        "age": [10, 20],
        "city": [["北京"], ["上海", "深圳"]],
    }

    assert resolve_user_splits_fixed(rules, True) == {
        "age": [True, True],
        "city": [True, True],
    }
    assert resolve_user_splits_fixed(rules, {"age": False, "city": [True, False]}) == {
        "age": [False, False],
        "city": [True, False],
    }


@pytest.mark.parametrize("value", [True, False, "-3", "ignore", "RAISE", 1.5, None])
def test_handle_unknown_rejects_non_integer_bin_numbers(value):
    """除精确的 raise 策略外，布尔值和非整数不能伪装成有效箱号。"""
    with pytest.raises(ValueError, match="handle_unknown.*整数箱号.*raise"):
        validate_handle_unknown(value)


@pytest.mark.parametrize("value", [-3, -2, -1, 0, 99, np.int64(3)])
def test_handle_unknown_accepts_every_integer_bin_number(value):
    """任意整数都能作为候选箱号，是否存在由拟合后的特征箱表校验。"""
    assert validate_handle_unknown(value) == int(value)


def test_handle_unknown_accepts_raise_policy():
    """删除 raise 分支会使用户无法选择在预测期未知类别上立即失败。"""
    assert validate_handle_unknown("raise") == "raise"


def test_category_assignment_uses_user_rule_before_special_and_missing_policy():
    """特殊值和缺失策略不能覆盖用户显式指定的类别箱。"""
    values = pd.Series([np.nan, "SPECIAL", "A", "UNKNOWN"], name="city")
    groups = [[np.nan, "A"], ["SPECIAL"]]

    actual = assign_category_groups(
        "city",
        values,
        groups,
        special_codes=[np.nan, "SPECIAL"],
        missing_separate=True,
        handle_unknown=UNKNOWN_BIN,
    )

    assert actual.tolist() == [0, 1, 0, UNKNOWN_BIN]


def test_reserved_bin_constants_are_distinct():
    """保留箱编号重叠会使转换阶段无法区分缺失、特殊和未知值。"""
    assert (MISSING_BIN, SPECIAL_BIN, UNKNOWN_BIN) == (-1, -2, -3)


def test_uniform_binner_exposes_only_formal_user_rule_parameters():
    """重新加入旧别名会让分箱器再次出现两套互斥公共契约。"""
    parameters = inspect.signature(UniformBinning.__init__).parameters

    assert {"user_splits", "user_splits_fixed", "special_codes", "max_n_bins"} <= set(parameters)
    assert {"n_bins", "split_points", "strict_user_splits", "special_values"}.isdisjoint(parameters)
    assert UniformBinning().handle_unknown == UNKNOWN_BIN


def test_numpy_fit_and_transform_share_generated_feature_names():
    """transform 重新生成整数列名会使 NumPy 特征绕过已经拟合的规则。"""
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.tile([0, 1], 10)

    fitted = UniformBinning(max_n_bins=3).fit(X, y)
    result = fitted.transform(X, metric="indices")

    assert list(result.columns) == ["feature_0", "feature_1"]
    assert set(fitted.splits_) == {"feature_0", "feature_1"}
    assert result.to_numpy().max() <= 2


def test_explicit_y_still_removes_target_column_and_fit_transform_returns_only_features():
    """显式 y 不能使 DataFrame 中的目标列泄漏为模型特征。"""
    df = pd.DataFrame({"x": np.arange(20, dtype=float), "target": np.tile([0, 1], 10)})
    external_y = 1 - df["target"]

    fitted = UniformBinning(max_n_bins=3).fit(df, external_y)
    transformed = UniformBinning(max_n_bins=3).fit_transform(df, external_y)

    assert fitted.feature_names_in_.tolist() == ["x"]
    assert list(transformed.columns) == ["x"]


def test_missing_target_is_rejected_before_statistics_conversion():
    """缺失目标不能进入整数坏样本计数并产生极大负数。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, np.nan, 1, 0], name="target")

    with pytest.raises(ValueError, match="目标变量.*缺失"):
        UniformBinning().fit(X, y)


@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES, ids=lambda cls: cls.__name__)
def test_direct_binner_common_signature_is_strict(binner_cls):
    """任一直接分箱器保留旧参数都会形成绕过统一契约的入口。"""
    parameters = inspect.signature(binner_cls.__init__).parameters

    assert {"max_n_bins", "user_splits", "user_splits_fixed", "special_codes", "handle_unknown"} <= set(parameters)
    assert {"n_bins", "split_points", "strict_user_splits", "special_values"}.isdisjoint(parameters)
    assert parameters["handle_unknown"].default == UNKNOWN_BIN


def test_optimal_binning_signature_and_method_kwargs_follow_sklearn_contract():
    """统一入口必须保留直接 ``**kwargs`` 用法，同时能被 clone/set_params 完整复刻。"""
    parameters = inspect.signature(OptimalBinning.__init__).parameters
    binner = OptimalBinning(method="cart", max_depth=3, min_samples_leaf=7)

    assert {"max_n_bins", "user_splits", "user_splits_fixed", "special_codes", "handle_unknown"} <= set(parameters)
    assert {"n_bins", "split_points", "strict_user_splits", "special_values"}.isdisjoint(parameters)
    assert binner.get_params()["max_depth"] == 3
    assert clone(binner).get_params()["min_samples_leaf"] == 7

    binner.set_params(max_depth=5)
    assert binner.get_params()["max_depth"] == 5


def test_optimal_binning_rejects_removed_or_unknown_kwargs():
    """旧别名和拼写错误不能再被 kwargs 静默吞掉。"""
    with pytest.raises((TypeError, ValueError)):
        OptimalBinning(n_bins=4)
    with pytest.raises(ValueError, match="无效"):
        OptimalBinning().set_params(not_a_binning_parameter=1)


def test_missing_is_learned_into_closest_bad_rate_bin_and_reused_by_transform():
    """missing_separate=False 必须按训练坏样本率学习归属，而不是在转换时临时猜测。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1, 1], name="target")
    binner = UniformBinning(
        user_splits={"x": [2.5]},
        user_splits_fixed=True,
        missing_separate=False,
    ).fit(X, y)

    transformed = binner.transform(pd.DataFrame({"x": [np.nan, 1.0, 4.0]}), metric="indices")

    assert binner._missing_bin_targets_["x"] == 1
    assert transformed["x"].tolist() == [1, 0, 1]
    assert MISSING_BIN not in binner.bin_tables_["x"]["分箱"].tolist()


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ([np.nan, 2.5], 0),
        ([2.5, np.nan], 1),
    ],
)
def test_numeric_user_nan_marker_has_priority_over_special_and_missing_separate(rule, expected):
    """user_splits 中 NaN 的位置必须覆盖 special_codes 和 missing_separate。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1], name="target")
    binner = UniformBinning(
        user_splits={"x": rule},
        user_splits_fixed=True,
        special_codes=[np.nan],
        missing_separate=True,
    ).fit(X, y)

    actual = binner.transform(pd.DataFrame({"x": [np.nan]}), metric="indices")

    assert actual.iloc[0, 0] == expected


def test_numeric_rule_export_distinguishes_separate_and_ordinary_missing_bins():
    """导出规则中的 NaN 只能表达普通箱位置，不能继续兼任独立 -1 箱标记。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1, 1], name="target")

    separate = UniformBinning(
        user_splits={"x": [2.5]},
        user_splits_fixed=True,
        missing_separate=True,
    ).fit(X, y)
    learned = UniformBinning(
        user_splits={"x": [2.5]},
        user_splits_fixed=True,
        missing_separate=False,
    ).fit(X, y)

    assert separate.export_rules()["x"] == [2.5]
    learned_rule = learned.export_rules()["x"]
    assert learned_rule[0] == 2.5
    assert pd.isna(learned_rule[1])


def test_imported_numeric_nan_rule_immediately_controls_transform_and_round_trips():
    """import_rules 不能只删除 NaN；导入后应立即按其位置映射缺失值。"""
    binner = UniformBinning(missing_separate=True).import_rules({"x": [np.nan, 2.5]})

    transformed = binner.transform(pd.DataFrame({"x": [np.nan, 1.0, 4.0]}), metric="indices")

    assert transformed["x"].tolist() == [0, 0, 1]
    exported = binner.export_rules()["x"]
    assert pd.isna(exported[0])
    assert exported[1:] == [2.5]


def test_update_numeric_nan_rule_recomputes_stats_with_explicit_missing_target():
    """update 提供训练数据时也必须按新规则重算缺失箱统计。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1], name="target")
    binner = UniformBinning(max_n_bins=2).fit(X, y)

    binner.update({"x": [2.5, np.nan]}, X=X, y=y)

    assert binner.transform(pd.DataFrame({"x": [np.nan]})).iloc[0, 0] == 1
    assert MISSING_BIN not in binner.bin_tables_["x"]["分箱"].tolist()


def test_missing_and_special_reserved_bins_are_kept_when_not_claimed_by_user_rule():
    """未被用户规则认领时，缺失和特殊值应稳定进入 -1 与 -2。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 999.0, np.nan]})
    y = pd.Series([0, 1, 0, 1], name="target")
    binner = UniformBinning(max_n_bins=2, special_codes=[999.0], missing_separate=True).fit(X, y)

    actual = binner.transform(pd.DataFrame({"x": [np.nan, 999.0]}), metric="indices")

    assert actual["x"].tolist() == [MISSING_BIN, SPECIAL_BIN]


def test_default_unknown_bin_is_recorded_and_custom_unknown_requires_existing_bin():
    """默认 -3 必须可转换；自定义未知箱号若未出现在训练表中必须拟合报错。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "B"]})
    y = pd.Series([0, 1, 0, 1], name="target")
    default = UniformBinning(max_n_bins=2).fit(X, y)
    custom = UniformBinning(max_n_bins=2, handle_unknown=0).fit(X, y)

    assert UNKNOWN_BIN in default._recorded_bins_["city"]
    assert default.transform(pd.DataFrame({"city": ["C"]})).iloc[0, 0] == UNKNOWN_BIN
    assert custom.transform(pd.DataFrame({"city": ["C"]})).iloc[0, 0] == 0
    with pytest.raises(ValueError, match="handle_unknown.*训练结果.*无记录"):
        UniformBinning(max_n_bins=2, handle_unknown=99).fit(X, y)


def test_per_feature_selective_fixed_nodes_survive_optimization_and_bin_cap():
    """字段级布尔掩码中的固定节点不得被舍入、移动或被 max_n_bins 裁掉。"""
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = pd.Series(np.tile([0, 1], 50), name="target")
    fixed = [10.123456, 60.555555]
    binner = OptimalBinning(
        method="best_iv",
        user_splits={"x": [fixed[0], 30.987654, fixed[1]]},
        user_splits_fixed={"x": [True, False, True]},
        max_n_bins=3,
        min_n_bins=2,
        lift_refine=True,
    ).fit(X, y)

    assert binner.n_bins_["x"] == 3
    assert any(np.isclose(binner.splits_["x"], fixed[0], rtol=0, atol=0))
    assert any(np.isclose(binner.splits_["x"], fixed[1], rtol=0, atol=0))


def test_selective_fixed_categorical_groups_only_merge_mutable_neighbors():
    """类别节点级固定不能退化成“任一节点固定即整字段全部固定”。"""
    X = pd.DataFrame({"city": list("AAAABBBBCCCCDDDD")})
    y = pd.Series([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], name="target")
    binner = OptimalBinning(
        method="uniform",
        user_splits={"city": [["A"], ["B"], ["C"], ["D"]]},
        user_splits_fixed={"city": [True, False, False, True]},
        min_n_bins=1,
        max_n_bins=3,
        min_bin_size=1,
        lift_refine=False,
    ).fit(X, y)

    assert binner.export_rules()["city"] == [["A"], ["B", "C"], ["D"]]


def test_prebinning_params_forward_common_method_and_user_rule_parameters():
    """预分箱参数必须到达实际预分箱器，而不是只接受少数字段或复制预分箱结果。"""
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = pd.Series(np.tile([0, 0, 1, 1], 25), name="target")
    method_params = {
        "max_n_bins": 8,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "missing_separate": False,
        "special_codes": [99.0],
    }
    method_binner = OptimalBinning(
        method="uniform",
        prebinning="tree",
        prebinning_params=method_params,
        lift_refine=False,
        n_jobs=1,
    ).fit(X, y)
    rule_binner = OptimalBinning(
        method="uniform",
        prebinning="tree",
        prebinning_params={
            "max_n_bins": 4,
            "user_splits": {"x": [20.5, 50.5]},
            "user_splits_fixed": {"x": [True, False]},
        },
        lift_refine=False,
        n_jobs=1,
    ).fit(X, y)

    actual_prebinner = method_binner._prebinner._binner
    assert isinstance(actual_prebinner, TreeBinning)
    assert actual_prebinner.max_depth == 2
    assert actual_prebinner.min_samples_leaf == 5
    assert actual_prebinner.missing_separate is False
    assert actual_prebinner.special_codes == [99.0]
    assert rule_binner._prebinner.user_splits == {"x": [20.5, 50.5]}
    assert 20.5 in rule_binner._prebinner.splits_["x"]


def test_candidate_search_optimizes_requested_ks_objective_and_monotonic_constraint():
    """KS/单调路径不能再调用固定 IV 目标或忽略单调约束。"""
    x = np.arange(12, dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    candidates = [1.5, 3.5, 5.5, 7.5, 9.5]

    ks_splits = search_candidate_splits(
        x,
        y,
        candidates,
        objective="ks",
        min_n_bins=2,
        max_n_bins=3,
        min_samples=2,
    )
    monotonic_splits = search_candidate_splits(
        x,
        y,
        candidates,
        objective="iv",
        min_n_bins=2,
        max_n_bins=3,
        min_samples=2,
        monotonic="ascending",
    )

    assert ks_splits == [3.5]
    bins = np.digitize(x, monotonic_splits)
    rates = np.asarray([y[bins == index].mean() for index in np.unique(bins)])
    assert np.all(np.diff(rates) >= -1e-12)


def test_cp_sat_ks_paths_never_fall_back_to_iv_builder(monkeypatch):
    """CPSATBinning 和 ORBinning(use_cp_sat=True) 的 KS 路径不得固定构建 IV 目标。"""
    X = pd.DataFrame({"x": np.arange(24, dtype=float)})
    y = pd.Series(np.tile([0, 0, 1, 1, 0, 1], 4), name="target")

    def fail_iv(*args, **kwargs):
        raise AssertionError("不应构建 IV 目标")

    monkeypatch.setattr(CPSATBinning, "_add_iv_objective_cp_sat", fail_iv)
    monkeypatch.setattr(ORBinning, "_add_cp_sat_iv_objective", fail_iv)

    CPSATBinning(objective="ks", max_n_bins=3, min_n_bins=2, time_limit=2).fit(X, y)
    ORBinning(objective="ks", use_cp_sat=True, max_n_bins=3, min_n_bins=2, time_limit=2).fit(X, y)


def test_cp_sat_iv_uses_global_candidate_partition_instead_of_contiguous_suffix():
    """IV 求解必须从任意候选组合中取全局最优，不能把选择变量错误约束为连续后缀。"""
    x = np.arange(20, dtype=float)
    y = np.asarray([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    binner = CPSATBinning(
        objective="iv",
        monotonic=False,
        min_n_bins=3,
        max_n_bins=3,
        min_bin_size=1,
        n_prebins=20,
        max_candidates=30,
        time_limit=2,
        n_jobs=1,
    )
    candidates, _, _ = binner._get_candidate_splits(pd.Series(x), pd.Series(y))
    expected = search_candidate_splits(
        x,
        y,
        candidates,
        objective="iv",
        min_n_bins=3,
        max_n_bins=3,
        min_samples=1,
        time_limit=2,
    )

    assert binner._cp_sat_numerical(pd.Series(x), pd.Series(y)) == expected == [3.5, 10.5]


def test_removed_chi_square_alias_is_not_in_public_signature():
    """已移除参数不能以“传入时报错”的形式继续占据公开签名。"""
    parameters = inspect.signature(ChiMergeBinning.__init__).parameters

    assert "min_chi2" not in parameters
    assert "min_chi2_threshold" in parameters


def test_exported_woe_mapping_uses_actual_bin_numbers_and_load_does_not_mutate_input():
    """缺失箱改变表行号时不能错配 WOE；load 也不能 pop 调用方字典。"""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1], name="target")
    fitted = UniformBinning(
        user_splits={"x": [2.5]},
        user_splits_fixed=True,
        missing_separate=True,
    ).fit(X, y)

    payload = fitted.export()
    expected = {int(row["分箱"]): float(row["分档WOE值"]) for _, row in fitted.bin_tables_["x"].iterrows()}
    for bin_number, woe in expected.items():
        assert payload["_woe_maps_"]["x"][bin_number] == pytest.approx(woe)

    load_payload = {"x": [2.5], "_woe_maps_": {"x": {"0": 0.25, "1": -0.25}}}
    original = {
        "x": list(load_payload["x"]),
        "_woe_maps_": {"x": dict(load_payload["_woe_maps_"]["x"])},
    }
    loaded = UniformBinning().load(load_payload)

    assert load_payload == original
    assert loaded._woe_maps_["x"] == {0: 0.25, 1: -0.25}


def test_optimal_transform_uses_final_woe_map_after_post_fit_split_changes():
    """Optimal 后处理切点后，transform 不能继续使用底层分箱器的旧 WOE 映射。"""
    X = pd.DataFrame({"x": np.repeat([0.0, 1.0, 2.0, 3.0], 30)})
    y = pd.Series(
        ([0] * 27 + [1] * 3) + ([0] * 21 + [1] * 9) + ([0] * 9 + [1] * 21) + ([0] * 3 + [1] * 27),
        name="target",
    )
    binner = OptimalBinning(method="target_bad_rate", max_n_bins=4, n_jobs=1).fit(X, y)

    indices = binner.transform(X, metric="indices")["x"]
    transformed = binner.transform(X, metric="woe")["x"]
    expected_map = {int(row["分箱"]): float(row["分档WOE值"]) for _, row in binner.bin_tables_["x"].iterrows()}

    np.testing.assert_allclose(transformed.to_numpy(), indices.map(expected_map).to_numpy())


def test_stats_report_uses_n_bins_fitted_name():
    """实际分箱数在属性和统计输出中都统一使用 n_bins_."""
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = pd.Series(np.tile([0, 1], 10), name="target")
    binner = OptimalBinning(method="uniform", max_n_bins=3).fit(X, y)

    stats = binner.get_stats("x")

    assert stats["n_bins_"] == binner.n_bins_["x"]
    assert "n_bins" not in stats
