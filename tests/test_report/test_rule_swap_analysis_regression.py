"""规则置入置出流水线业务口径回归测试。"""

import numpy as np
import pandas as pd
import pytest
from inspect import signature

from hscredit.core.rules import Rule
from hscredit.report import rule_swap_analysis


def _swap_case():
    """构造可手算的基础规则、置出和置入样本。"""
    data = pd.DataFrame(
        {
            "score": [10, 20, 30, 40, 50, 60],
            "base": [1, 0, 0, 0, 0, 0],
            "out": [0, 1, 0, 0, 0, 0],
            "x": [0, 0, 1, 2, 3, 4],
            "target": [1, 0, 0, 1, 0, 0],
            "MOB1": [10, 0, 8, 4, 0, 12],
            "amount": [100, 200, np.nan, 400, 500, 600],
            "approved_amount": [np.nan, np.nan, 250, np.nan, np.nan, np.nan],
        }
    )
    table = pd.DataFrame(
        {
            "分箱标签": ["[-inf, 35)", "[35, +inf)"],
            "坏样本率": [0.25, 0.50],
        }
    )
    kwargs = {
        "data": data,
        "score": "score",
        "bin_table": table,
        "rules_base": [Rule("base == 1", name="基础拒绝")],
        "rules_out": [Rule("out == 1", name="本次置出")],
        "rules_in": [Rule("x == 1", name="本次置入")],
        "target": "target",
        "n_jobs": 1,
    }
    return data, table, kwargs


def _row(pipeline, rule_class, rule_name=""):
    """按规则分类和名称提取唯一报告行。"""
    matched = pipeline[
        (pipeline["规则分类"] == rule_class)
        & (pipeline["指标名称"].fillna("") == rule_name)
    ]
    assert len(matched) == 1
    return matched.iloc[0]


def test_rule_swap_uses_total_pass_as_out_in_parent_and_conserves_samples():
    """防止置入规则使用基础规则后母集并导致 ALL-IN 重复计数。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(**kwargs)["swap_pipeline"]

    assert _row(pipeline, "IN-IN通过")["样本总数"] == 3
    assert _row(pipeline, "OUT-IN置入", "合计")["样本总数"] == 1
    assert _row(pipeline, "ALL-IN置换")["样本总数"] == 4


def test_rule_swap_independent_total_deduplicates_and_conserves_samples():
    """防止独立规则合计重复命中，并确保 IN-IN 与 OUT-IN 守恒。"""
    _, _, kwargs = _swap_case()
    kwargs["rules_in"] = [
        Rule("x >= 2", name="规则A"),
        Rule("x >= 3", name="规则B"),
    ]

    pipeline = rule_swap_analysis(rule_analysis_mode="independent", **kwargs)["swap_pipeline"]

    assert _row(pipeline, "OUT-IN置入", "规则A")["样本总数"] == 3
    assert _row(pipeline, "OUT-IN置入", "规则B")["样本总数"] == 2
    assert _row(pipeline, "OUT-IN置入", "合计")["样本总数"] == 3
    assert _row(pipeline, "IN-IN通过")["样本总数"] == 1
    assert _row(pipeline, "ALL-IN置换")["样本总数"] == 4


def test_rule_swap_survival_rate_calibrates_exact_production_funnel():
    """防止通过整数扩样本分母导致有偏通过率校准失真。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(sample_survival_rate=0.7, **kwargs)["swap_pipeline"]

    assert "生产通过率" in pipeline.columns
    assert pipeline.iloc[0]["生产通过率"] == pytest.approx(70.0)
    assert _row(pipeline, "ALL-IN置换")["生产通过率"] == pytest.approx(46.6666666667)


def test_non_out_in_uses_actual_y_and_out_in_uses_predicted_uplift():
    """仅 OUT-IN 使用评分预测风险，其余客群使用实际表现。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(out_in_uplift=2.0, **kwargs)["swap_pipeline"]

    assert _row(pipeline, "IN-IN通过")["原始坏样本数"] == pytest.approx(1.0)
    assert _row(pipeline, "OUT-IN置入", "合计")["原始坏样本数"] == pytest.approx(0.25)
    assert _row(pipeline, "OUT-IN置入", "合计")["调整后坏样本数"] == pytest.approx(0.5)


def test_swap_result_compares_in_in_with_all_in_and_ignores_display_order():
    """置换汇总比较 IN-IN 与 ALL-IN，且不依赖报告展示顺序。"""
    _, _, kwargs = _swap_case()

    normal = rule_swap_analysis(reverse_order=False, **kwargs)["swap_result"]
    reverse = rule_swap_analysis(reverse_order=True, **kwargs)["swap_result"]

    pd.testing.assert_frame_equal(normal, reverse)
    pass_row = normal.set_index("指标").loc["通过率"]
    assert pass_row["变化后"] > pass_row["变化前"]


def test_risk_uplifts_can_adjust_other_atomic_groups():
    """可选上浮映射可作用于 OUT-IN 之外的原子客群。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(risk_uplifts={"in_in": 1.5}, **kwargs)["swap_pipeline"]
    row = _row(pipeline, "IN-IN通过")

    assert row["调整后坏样本数"] == pytest.approx(row["原始坏样本数"] * 1.5)


def test_amount_keeps_order_counts_and_adds_mixed_risk_amounts():
    """金额分析保留订单计数，并将金额指标作为独立列输出。"""
    data, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(amount="amount", **kwargs)["swap_pipeline"]
    full = pipeline.iloc[0]

    assert full["样本总数"] == len(data)
    assert full["样本总额"] == pytest.approx(data["amount"].sum())
    assert full["生产通过率"] <= 100.0


def test_out_in_amount_column_then_fill_has_precedence():
    """OUT-IN 金额优先取指定字段，再取原金额，最后使用填充值。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(
        amount="amount",
        out_in_amount_col="approved_amount",
        out_in_amount_fill=1000.0,
        **kwargs,
    )["swap_pipeline"]
    out_in = _row(pipeline, "OUT-IN置入", "合计")

    assert out_in["样本总额"] == pytest.approx(250.0)


def test_multi_dpd_pipeline_exposes_each_nonzero_target():
    """多个 DPD 标签分别使用对应分箱坏样本率并输出非零风险。"""
    _, _, kwargs = _swap_case()
    kwargs["bin_table"] = pd.DataFrame(
        [
            ["[-inf, 35)", 0.10, 0.20, 0.30],
            ["[35, +inf)", 0.40, 0.50, 0.60],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("分箱详情", "分箱标签"),
                ("MOB1_7+", "坏样本率"),
                ("MOB1_3+", "坏样本率"),
                ("MOB1_0+", "坏样本率"),
            ]
        ),
    )
    kwargs.pop("target")

    pipeline = rule_swap_analysis(overdue="MOB1", dpds=[7, 3, 0], **kwargs)["swap_pipeline"]

    assert isinstance(pipeline.columns, pd.MultiIndex)
    assert {"MOB1_7+", "MOB1_3+", "MOB1_0+"}.issubset(pipeline.columns.get_level_values(0))
    assert pipeline[("MOB1_7+", "调整后坏样本率")].max() > 0


@pytest.mark.parametrize("label_col", ["分箱标签", "分箱"])
def test_single_bin_and_legacy_label_predict_constant_bad_rate(label_col):
    """单箱及旧分箱列名必须映射为箱内坏率，不能静默返回零。"""
    _, _, kwargs = _swap_case()
    kwargs["bin_table"] = pd.DataFrame({label_col: ["[-inf, +inf)"], "坏样本率": [0.25]})

    pipeline = rule_swap_analysis(**kwargs)["swap_pipeline"]

    assert _row(pipeline, "OUT-IN置入", "合计")["原始坏样本率"] == pytest.approx(0.25)


def test_multi_score_bin_keys_and_negative_weights_are_rejected_in_chinese():
    """多评分分箱键和权重必须完整、有限且非负。"""
    data, table, kwargs = _swap_case()
    multi_data = data.assign(s1=data["score"], s2=data["score"])
    common = {
        "data": multi_data,
        "score": {"a": "s1", "b": "s2"},
        "rules_in": kwargs["rules_in"],
        "target": "target",
        "n_jobs": 1,
    }

    with pytest.raises(ValueError, match="评分名必须完全一致"):
        rule_swap_analysis(bin_table={"a": table}, **common)
    with pytest.raises(ValueError, match="有限且非负"):
        rule_swap_analysis(
            bin_table={"a": table, "b": table},
            score_weights={"a": 1.0, "b": -0.5},
            **common,
        )


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.1, np.nan])
def test_sample_survival_rate_must_be_in_unit_interval(rate):
    """输入样本的生产幸存比例必须使用 (0, 1] 小数口径。"""
    _, _, kwargs = _swap_case()

    with pytest.raises(ValueError, match="样本集幸存比例必须位于"):
        rule_swap_analysis(sample_survival_rate=rate, **kwargs)


def test_non_out_in_missing_target_is_rejected_in_chinese():
    """有表现客群缺少实际标签时不得静默当作好样本。"""
    data, _, kwargs = _swap_case()
    kwargs["data"] = data.assign(target=data["target"].mask(data.index == 4))

    with pytest.raises(ValueError, match="非OUT-IN样本缺少实际表现"):
        rule_swap_analysis(**kwargs)


def test_independent_rule_order_does_not_change_swap_result():
    """独立模式的规则展示顺序不应影响去重后的置换结论。"""
    _, _, kwargs = _swap_case()
    rule_a = Rule("x >= 2", name="规则A")
    rule_b = Rule("x >= 3", name="规则B")

    forward = rule_swap_analysis(rules_in=[rule_a, rule_b], **{k: v for k, v in kwargs.items() if k != "rules_in"})[
        "swap_result"
    ]
    backward = rule_swap_analysis(rules_in=[rule_b, rule_a], **{k: v for k, v in kwargs.items() if k != "rules_in"})[
        "swap_result"
    ]

    pd.testing.assert_frame_equal(forward, backward)


@pytest.mark.parametrize("uplift", [-1.0, np.nan, np.inf])
def test_risk_uplifts_must_be_finite_and_nonnegative(uplift):
    """风险上浮参数不得为负数或非有限值。"""
    _, _, kwargs = _swap_case()

    with pytest.raises(ValueError, match="有限且非负"):
        rule_swap_analysis(out_in_uplift=uplift, **kwargs)


def test_total_pass_is_reported_when_only_rules_in_are_configured():
    """没有本次置出规则时，基础拒绝后的样本仍是置入分析的 total 母集。"""
    _, _, kwargs = _swap_case()
    kwargs.pop("rules_out")

    pipeline = rule_swap_analysis(**kwargs)["swap_pipeline"]

    assert _row(pipeline, "total通过样本")["样本总数"] == 5
    assert _row(pipeline, "ALL-IN置换")["样本总数"] == 5


def test_missing_out_in_score_uses_nonmissing_bin_fallback_and_marks_source():
    """OUT-IN 评分缺失时使用非缺失总体坏率，并在风险来源中明确标记。"""
    data, _, kwargs = _swap_case()
    kwargs["data"] = data.assign(score=data["score"].mask(data.index == 2))

    pipeline = rule_swap_analysis(**kwargs)["swap_pipeline"]
    out_in = _row(pipeline, "OUT-IN置入", "合计")

    assert out_in["原始坏样本率"] == pytest.approx(0.4)
    assert "缺失回退" in out_in["风险来源"]


def test_pipeline_exposes_raw_and_adjusted_good_sample_counts():
    """原始和调整后风险同时提供可加总的好坏样本期望数。"""
    _, _, kwargs = _swap_case()

    pipeline = rule_swap_analysis(risk_uplifts={"in_in": 1.5}, **kwargs)["swap_pipeline"]
    in_in = _row(pipeline, "IN-IN通过")

    assert in_in["原始好样本数"] == pytest.approx(2.0)
    assert in_in["调整后好样本数"] == pytest.approx(1.5)


def test_invalid_bin_table_type_is_rejected_instead_of_zero_risk():
    """非法分箱表类型不得退化为空预测并把 OUT-IN 当作好样本。"""
    _, _, kwargs = _swap_case()
    kwargs["bin_table"] = 42

    with pytest.raises(TypeError, match="bin_table 必须"):
        rule_swap_analysis(**kwargs)


def test_sequential_rule_rows_use_previous_remaining_state():
    """串行置入规则的阶段前状态必须承接上一条规则的阶段后状态。"""
    _, _, kwargs = _swap_case()
    kwargs["rules_in"] = [
        Rule("x == 1", name="置入1"),
        Rule("x == 2", name="置入2"),
    ]

    pipeline = rule_swap_analysis(rule_analysis_mode="sequential", **kwargs)["swap_pipeline"]

    first = _row(pipeline, "OUT-IN置入", "置入1")
    second = _row(pipeline, "OUT-IN置入", "置入2")
    assert first["阶段前样本数"] == 2
    assert first["阶段后样本数"] == 3
    assert second["阶段前样本数"] == 3
    assert second["阶段后样本数"] == 4


def test_multi_target_rejects_flat_shared_bad_rate_table():
    """多目标分析必须显式提供每个目标对应的坏样本率列。"""
    _, table, kwargs = _swap_case()
    kwargs.pop("target")

    with pytest.raises(ValueError, match="每个目标"):
        rule_swap_analysis(
            overdue="MOB1",
            dpds=[7, 3, 0],
            bin_table=table,
            **{k: v for k, v in kwargs.items() if k != "bin_table"},
        )


def test_new_risk_uplifts_parameter_does_not_shift_existing_positional_slots():
    """新增参数必须追加，避免破坏既有位置参数调用。"""
    parameter_names = list(signature(rule_swap_analysis).parameters)

    assert parameter_names.index("amount") == parameter_names.index("out_in_uplift") + 1
    assert parameter_names.index("risk_uplifts") > parameter_names.index("parallel_config")


def test_duplicate_dataframe_index_is_supported():
    """分析内部应使用唯一位置索引，不依赖业务索引唯一。"""
    data, _, kwargs = _swap_case()
    kwargs["data"] = data.set_axis([0, 0, 1, 1, 2, 2])

    pipeline = rule_swap_analysis(**kwargs)["swap_pipeline"]

    assert pipeline.iloc[0]["样本总数"] == len(data)


def test_multi_score_requires_keyed_tables_and_complete_dict_weights():
    """多评分必须逐评分提供分箱表，字典权重也必须覆盖全部评分。"""
    data, table, kwargs = _swap_case()
    multi_data = data.assign(s1=data["score"], s2=data["score"])
    common = dict(
        data=multi_data,
        score={"a": "s1", "b": "s2"},
        rules_in=kwargs["rules_in"],
        target="target",
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="逐评分提供"):
        rule_swap_analysis(bin_table=table, **common)
    with pytest.raises(ValueError, match="评分名必须完全一致"):
        rule_swap_analysis(
            bin_table={"a": table, "b": table},
            score_weights={"a": 1.0},
            **common,
        )


def test_swap_result_reports_effective_risk_uplift_and_preserves_single_schema():
    """风险上浮汇总反映前后客群的有效系数，单目标结构保持兼容。"""
    _, _, kwargs = _swap_case()

    swap_result = rule_swap_analysis(risk_uplifts={"in_in": 1.5}, **kwargs)["swap_result"]
    uplift = swap_result.set_index("指标").loc["风险上浮系数"]

    assert "目标标签" not in swap_result.columns
    assert uplift["变化前"] == pytest.approx(1.5)
    assert uplift["变化后"] == pytest.approx(1.6)


@pytest.mark.parametrize(
    "overdue, dpds",
    [([], [7]), (["MOB1"], []), ("MOB1", [])],
)
def test_empty_overdue_or_dpd_configuration_is_rejected_in_chinese(overdue, dpds):
    """空多标签配置必须给出中文参数错误，而不是 StopIteration。"""
    _, _, kwargs = _swap_case()
    kwargs.pop("target")

    with pytest.raises(ValueError, match="不能为空"):
        rule_swap_analysis(overdue=overdue, dpds=dpds, **kwargs)


def test_multi_bin_without_usable_intervals_rejects_nonmissing_scores():
    """总体坏率回退只用于评分缺失，正常评分无法落箱时必须报错。"""
    _, _, kwargs = _swap_case()
    kwargs["bin_table"] = pd.DataFrame(
        {
            "分箱标签": ["箱1", "箱2"],
            "样本总数": [90, 10],
            "坏样本率": [0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="无法映射非缺失评分"):
        rule_swap_analysis(**kwargs)


def test_relative_change_is_undefined_when_zero_baseline_becomes_nonzero():
    """零基线增长不能报告为相对变化 0，应显式输出未定义。"""
    _, _, kwargs = _swap_case()
    kwargs["rules_in"] = [Rule("x >= 0", name="全部置入")]

    result = rule_swap_analysis(**kwargs)
    out_in = _row(result["swap_pipeline"], "OUT-IN置入", "合计")
    summary = result["swap_result"].set_index("指标")

    assert pd.isna(out_in["通过率(相对值)"])
    assert pd.isna(summary.loc["通过率", "相对变化"])
    assert pd.isna(summary.loc["逾期率", "相对变化"])
