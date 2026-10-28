"""通过率以原始行独立核算：规则模式、执行后端和样本范围相互正交。"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from hscredit.core.rules import Rule
from hscredit.report import rule_swap_analysis


def _data():
    return pd.DataFrame(
        {
            "编号": range(12),
            "评分": range(12),
            "标签": [0, 1] * 6,
            "旧通过": [False, False] + [True] * 7 + [False] * 3,
            "金额": np.arange(1, 13) * 100,
        }
    )


def _call(data, **kwargs):
    options = dict(
        score="评分",
        target="标签",
        bin_table=pd.DataFrame({"分箱标签": ["[-inf, +inf)"], "坏样本率": [0.2]}),
        n_jobs=1,
    )
    options.update(kwargs)
    return rule_swap_analysis(data, **options)


def _assert_summary(result, old, new, survival):
    row = result["swap_result"].set_index("指标").loc["通过率"]
    before = survival * np.count_nonzero(old) / len(old) * 100
    after = survival * np.count_nonzero(new) / len(new) * 100
    assert row["变化前"] == pytest.approx(before)
    assert row["变化后"] == pytest.approx(after)
    assert row["绝对变化"] == pytest.approx(after - before)
    relative = (after - before) / before if before else (0 if after == 0 else np.nan)
    if np.isnan(relative):
        assert pd.isna(row["相对变化"])
    else:
        assert row["相对变化"] == pytest.approx(relative)


@pytest.mark.parametrize("survival", [1.0, 0.7, 0.123456789])
@pytest.mark.parametrize("mode", ["independent", "sequential"])
@pytest.mark.parametrize("backend", [None, "threading", "loky"])
@pytest.mark.parametrize("scope", ["full", "survivors", "passed"])
def test_explicit_old_decision_matrix(survival, mode, backend, scope):
    data = _data()
    if scope == "survivors":
        data = data.iloc[2:].copy()
    elif scope == "passed":
        data = data.loc[data["旧通过"]].copy()
    ids = data["编号"].to_numpy()
    old = data["旧通过"].to_numpy()
    # 手工真值表：置出 6/7/8；置入 9/10，11 仍拒绝。
    new = (old & ~np.isin(ids, [6, 7, 8])) | (~old & np.isin(ids, [9, 10]))
    result = _call(
        data,
        original_pass="旧通过",
        rules_out=[Rule("编号 >= 6"), Rule("编号 == 8")],
        rules_in=[Rule("编号 == 9"), Rule("编号 >= 9 and 编号 <= 10")],
        sample_survival_rate=survival,
        rule_analysis_mode=mode,
        n_jobs=1 if backend is None else 2,
        parallel_backend=backend,
        parallel_config={"adaptive": False},
    )
    _assert_summary(result, old, new, survival)
    pipeline = result["swap_pipeline"]
    # 每行通过率以决策人数计算，与命中样本坏率无关。
    for _, row in pipeline.iterrows():
        before = survival * row["阶段前样本数"] / len(data) * 100
        after = survival * row["阶段后样本数"] / len(data) * 100
        assert row["阶段前生产通过率"] == pytest.approx(before)
        for column in ["通过率", "生产通过率", "通过率(绝对值)"]:
            assert row[column] == pytest.approx(after)
        assert row["通过率变化"] == pytest.approx(after - before)
        if before:
            assert row["通过率(相对值)"] == pytest.approx((after - before) / before)
        else:
            assert pd.isna(row["通过率(相对值)"]) if after else row["通过率(相对值)"] == 0
    expected_before = int(old.sum())
    for category, expected_after in [("IN-OUT置出", int((old & new).sum())), ("OUT-IN置入", int(new.sum()))]:
        rows = pipeline[pipeline["规则分类"].eq(category)]
        total = rows.loc[rows["指标名称"].eq("合计")].iloc[0]
        assert total["阶段前样本数"] == expected_before
        assert total["阶段后样本数"] == expected_after
        details = rows[rows["行类型"].eq("规则明细")]
        if mode == "sequential":
            assert details.iloc[0]["阶段前样本数"] == expected_before
            assert details.iloc[-1]["阶段后样本数"] == expected_after
            assert details["通过率变化"].sum() == pytest.approx(total["通过率变化"])
            assert details["决策样本数"].sum() == total["决策样本数"]
        else:
            assert details["阶段前样本数"].eq(expected_before).all()
        expected_before = expected_after


@pytest.mark.parametrize(
    "counts",
    [
        (6, 2, 1, 1),
        (6, 1, 2, 1),
        (6, 2, 2, 0),
        (6, 2, 0, 2),
        (6, 0, 2, 2),
        (10, 0, 0, 0),
        (0, 0, 10, 0),
        (0, 10, 0, 0),
        (0, 0, 0, 10),
    ],
)
@pytest.mark.parametrize("survival", [1.0, 0.7])
@pytest.mark.parametrize("mode", ["independent", "sequential"])
def test_legacy_quadrants_use_old_and_new_pass_not_only_retained(counts, survival, mode):
    group = np.repeat(["II", "IO", "OI", "OO"], counts)
    data = pd.DataFrame({"评分": np.arange(len(group)), "标签": np.arange(len(group)) % 2, "组": group})
    result = _call(
        data,
        rules_base=[Rule('组 == "OO"')],
        rules_out=[Rule('组 == "IO"')],
        rules_in=[Rule('组 == "OI"')],
        sample_survival_rate=survival,
        rule_analysis_mode=mode,
    )
    _assert_summary(result, np.isin(group, ["II", "IO"]), np.isin(group, ["II", "OI"]), survival)


def test_only_out_and_no_rules():
    data = _data()
    _assert_summary(_call(data, rules_out=[Rule("编号 >= 8")]), np.ones(12), np.arange(12) < 8, 1)
    _assert_summary(_call(data), np.ones(12), np.ones(12), 1)


def test_fixed_old_pass_does_not_move_when_in_rule_changes():
    data = _data()
    for value in [9, 10, 11]:
        result = _call(data, original_pass="旧通过", rules_in=[Rule(f"编号 >= 9 and 编号 <= {value}")])
        _assert_summary(result, data["旧通过"], data["旧通过"] | data["编号"].between(9, value), 1)


def test_grey_labels_do_not_change_traffic_and_missing_swap_ins_survive():
    data = _data().assign(标签=[0, 1, 0, 1, -1, 1, 0, 1, 0, np.nan, np.nan, 1])
    results = []
    for grey in [False, True]:
        result = _call(data, original_pass="旧通过", rules_in=[Rule("编号 == 9 or 编号 == 10")], del_grey=grey)
        _assert_summary(result, data["旧通过"], data["旧通过"] | data["编号"].isin([9, 10]), 1)
        pipeline = result["swap_pipeline"]
        row = pipeline.loc[pipeline["规则分类"].eq("OUT-IN置入") & pipeline["指标名称"].eq("合计")].iloc[0]
        assert row["样本总数"] == row["决策样本数"] == 2
        results.append(pipeline[["阶段前样本数", "阶段后样本数", "生产通过率"]])
    pd.testing.assert_frame_equal(*results)


@pytest.mark.parametrize("kind", ["series", "array", "column"])
def test_original_pass_alignment_and_duplicate_index(kind):
    data = _data().set_axis([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    value = {"series": data["旧通过"], "array": data["旧通过"].to_numpy(), "column": "旧通过"}[kind]
    result = _call(data, original_pass=value, rules_out=[Rule("编号 >= 6")])
    _assert_summary(result, data["旧通过"], data["旧通过"] & (data["编号"] < 6), 1)


@pytest.mark.parametrize("value", [[True], [True] * 11 + [None], [2] * 12])
def test_invalid_original_decisions_are_not_silently_coerced(value):
    with pytest.raises(ValueError, match="original_pass"):
        _call(_data(), original_pass=value)


def test_amount_and_display_order_cannot_change_approval():
    data = _data()
    normal = _call(data, original_pass="旧通过", rules_out=[Rule("编号 >= 6")])
    altered = _call(data, original_pass="旧通过", rules_out=[Rule("编号 >= 6")], amount="金额", reverse_order=True)
    pd.testing.assert_frame_equal(normal["swap_result"], altered["swap_result"])


@pytest.mark.parametrize("mode", ["independent", "sequential"])
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_full_outputs_match_serial(mode, backend):
    data = _data()
    options = dict(
        original_pass="旧通过",
        rules_out=[Rule("编号 >= 6"), Rule("编号 == 8")],
        rules_in=[Rule("编号 >= 9"), Rule("编号 == 10")],
        rule_analysis_mode=mode,
        parallel_config={"adaptive": False},
    )
    expected = _call(data, **options)
    actual = _call(data, n_jobs=2, parallel_backend=backend, **options)
    for key in expected:
        pd.testing.assert_frame_equal(expected[key], actual[key])


def test_multilabel_grey_uses_one_approval_denominator():
    data = _data().assign(MOB1=[0, 10, 0, 10, 4, 0, 10, 4, 0, np.nan, np.nan, 0])
    bins = pd.DataFrame(
        [["[-inf, +inf)", 0.2, 0.3]],
        columns=pd.MultiIndex.from_tuples([("分箱详情", "分箱标签"), ("MOB1_7+", "坏样本率"), ("MOB1_3+", "坏样本率")]),
    )
    result = _call(
        data,
        target=None,
        overdue="MOB1",
        dpds=[7, 3],
        del_grey=True,
        bin_table=bins,
        original_pass="旧通过",
        rules_out=[Rule("编号 == 8")],
        rules_in=[Rule("编号 >= 9")],
    )
    summary = result["swap_result"]
    rates = summary[summary["指标"].eq("通过率")]
    np.testing.assert_allclose(rates["变化前"], 7 / 12 * 100)
    np.testing.assert_allclose(rates["变化后"], 9 / 12 * 100)
    pipeline = result["swap_pipeline"]
    assert pipeline.iloc[0][("分箱详情", "决策样本数")] == 12
    assert pipeline.iloc[0][("MOB1_7+", "样本总数")] == 10
    assert pipeline.iloc[0][("MOB1_3+", "样本总数")] == 12


def test_all_grey_still_has_decisions_and_undefined_observed_risk():
    data = _data().assign(标签=-1)
    result = _call(data, original_pass="旧通过", del_grey=True, rules_out=[Rule("编号 >= 6")])
    _assert_summary(result, data["旧通过"], data["旧通过"] & (data["编号"] < 6), 1)
    assert result["swap_pipeline"]["样本总数"].eq(0).all()
    assert result["swap_result"].set_index("指标").loc["逾期率", ["变化前", "变化后"]].isna().all()


def test_original_pass_series_aligns_by_unique_labels_and_base_is_consistent():
    data = _data().set_axis(list(reversed(range(12))))
    result = _call(
        data, original_pass=data["旧通过"].sort_index(), rules_base=[Rule("编号 < 2")], rules_out=[Rule("编号 >= 6")]
    )
    _assert_summary(result, data["旧通过"], data["旧通过"] & (data["编号"] < 6), 1)
    with pytest.raises(ValueError, match="rules_base"):
        _call(data, original_pass="旧通过", rules_base=[Rule("编号 == 2")])


@pytest.mark.parametrize("mode", ["independent", "sequential"])
def test_base_stage_includes_other_rejects_without_breaking_funnel(mode):
    result = _call(
        _data(),
        original_pass="旧通过",
        rules_base=[Rule("编号 == 0"), Rule("编号 < 2")],
        rules_in=[Rule("编号 == 9")],
        rule_analysis_mode=mode,
    )
    p = result["swap_pipeline"]
    rejected = p[p["规则分类"].eq("OUT-OUT拒绝")]
    total = rejected[rejected["指标名称"].eq("合计")].iloc[0]
    assert total["决策样本数"] == 4  # 编号 0、1、10、11；9 被置入。
    assert total["阶段前样本数"] == 12
    assert total["阶段后样本数"] == 8
    assert p.loc[p["规则分类"].eq("剩余样本"), "阶段后样本数"].item() == 8
    if mode == "sequential":
        details = rejected[rejected["行类型"].eq("规则明细")]
        assert details["阶段前样本数"].tolist() == [12, 11, 10]
        assert details["阶段后样本数"].tolist() == [11, 10, 8]
        assert details["通过率变化"].sum() == pytest.approx(total["通过率变化"])


@pytest.mark.parametrize("survival", [1.0, 0.68])
@pytest.mark.parametrize("mode", ["independent", "sequential"])
@pytest.mark.parametrize("backend", [None, "threading", "loky"])
def test_real_workbook_raw_row_approval_matrix(survival, mode, backend):
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少本地演示工作簿 hscredit_yyp.xlsx")
    raw = pd.read_excel(path)
    data = raw.rename(
        columns={
            "衡枢鉴真分老客版": "评分",
            "FPD": "标签",
            "近六个月非银多头机构数": "多头",
            "青云24": "青云",
            "极光欺诈分6v1": "欺诈",
        }
    )
    data["欺诈"] = data["欺诈"].fillna(-1)
    # 直接在原始列上计算，不调用 Rule 或生产掩码函数作为期望值。
    base = (data["欺诈"] >= 0.7) | (data["评分"] >= 0.23)
    out = ~base & ((data["多头"] >= 75) | (data["青云"] < 520))
    ins = (
        ~base & ~out & (((data["青云"] >= 680) & (data["评分"] < 0.1)) | ((data["多头"] <= 50) & (data["评分"] < 0.06)))
    )
    kept = ~(base | out | ins)
    result = _call(
        data,
        rules_base=[Rule("欺诈 >= .7"), Rule("评分 >= .23")],
        rules_out=[Rule("多头 >= 75"), Rule("青云 < 520")],
        rules_in=[Rule("青云 >= 680 and 评分 < .1"), Rule("多头 <= 50 and 评分 < .06")],
        sample_survival_rate=survival,
        rule_analysis_mode=mode,
        n_jobs=1 if backend is None else 2,
        parallel_backend=backend,
        parallel_config={"adaptive": False},
    )
    _assert_summary(result, kept | out, kept | ins, survival)
    old_risk = result["swap_result"].set_index("指标").loc["逾期率", "变化前"]
    assert old_risk == pytest.approx(data.loc[kept | out, "标签"].mean())
