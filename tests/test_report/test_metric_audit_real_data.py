"""真实放款数据逐箱核算；预期直接从原始行聚合，不复用统计内核。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from hscredit.core.binning import OptimalBinning
from hscredit.core.metrics import auc, ks, ks_bucket
from hscredit.core.rules import Rule
from hscredit.report import feature_bin_stats
from hscredit.report.mining import RuleMetrics

FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别"]


@pytest.fixture(scope="module")
def loans():
    path = Path(__file__).resolve().parents[2] / "examples/hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    return pd.read_excel(path)


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("mode", ["FPD", "overdue", "overdue_grey"])
@pytest.mark.parametrize("weighted", [False, True])
def test_real_feature_tables_match_raw_rows(loans, feature, mode, weighted):
    binner = OptimalBinning(method="quantile", max_n_bins=4, n_jobs=1).fit(loans[[feature]], loans.FPD)
    bins = binner.transform(loans[[feature]], metric="indices")[feature].to_numpy()
    labels = binner.bin_tables_[feature].set_index("分箱")["分箱标签"].to_dict()
    amount = loans["放款金额"].to_numpy(dtype=float) if weighted else np.ones(len(loans))
    if mode == "FPD":
        options = dict(target="FPD")
        targets = {None: (loans.FPD.to_numpy(), np.ones(len(loans), dtype=bool))}
    else:
        options = dict(overdue=["MOB1"], dpds=[7, 3, 0], del_grey=mode == "overdue_grey")
        dpd_values = loans.MOB1.to_numpy()
        targets = {
            f"MOB1_{threshold}+": (
                (dpd_values > threshold).astype(int),
                (
                    ~((dpd_values > 0) & (dpd_values <= threshold))
                    if mode == "overdue_grey"
                    else np.ones(len(loans), dtype=bool)
                ),
            )
            for threshold in [7, 3, 0]
        }
    table = feature_bin_stats(
        loans,
        feature,
        binner=binner,
        amount="放款金额" if weighted else None,
        margins=True,
        round_digits=False,
        n_jobs=1,
        **options,
    )
    for target_name, (y, valid) in targets.items():
        if target_name is None:
            frame = table
        else:
            frame = table[target_name].copy()
            for column in ["分箱标签", "样本总数", "样本占比"]:
                if column not in frame:
                    frame[column] = table[("分箱详情", column)]
        frame = frame.set_index("分箱标签")
        total = float(amount[valid].sum())
        total_bad = float(amount[valid & (y == 1)].sum())
        total_good = total - total_bad
        baseline_rate = total_bad / total
        assert frame.loc["合计", "样本总数"] == pytest.approx(total)
        assert frame.loc["合计", "坏样本数"] == pytest.approx(total_bad)
        assert frame.loc["合计", "坏样本率"] == pytest.approx(baseline_rate)
        for code in np.unique(bins):
            selected = valid & (bins == code)
            weight = float(amount[selected].sum())
            bad = float(amount[selected & (y == 1)].sum())
            row = frame.loc[labels[code]]
            assert row["样本总数"] == pytest.approx(weight)
            assert row["好样本数"] == pytest.approx(weight - bad)
            assert row["坏样本数"] == pytest.approx(bad)
            assert row["坏样本率"] == pytest.approx(bad / weight if weight else 0, abs=5.1e-7)
            assert row["好样本占比"] == pytest.approx((weight - bad) / total_good, abs=5.1e-7)
            assert row["坏样本占比"] == pytest.approx(bad / total_bad, abs=5.1e-7)
            remaining_rate = (total_bad - bad) / (total - weight) if total > weight else 0
            improvement = (baseline_rate - remaining_rate) / baseline_rate
            efficiency = improvement / (weight / total) if weight else 0
            assert row["坏账改善"] == pytest.approx(improvement, abs=5.1e-5)
            assert row["风险拒绝比"] == pytest.approx(efficiency, abs=5.1e-5)


@pytest.mark.parametrize("feature", FEATURES[:3])
def test_real_ranking_and_ks_table_consistency(loans, feature):
    y, score = loans.FPD.to_numpy(), loans[feature].to_numpy()
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    assert auc(y, score, score_direction="higher_risk") == pytest.approx(roc_auc_score(y, score))
    assert ks(y, score) == pytest.approx(np.max(abs(tpr - fpr)))
    table = ks_bucket(y, score, method="quantile", max_n_bins=5, n_jobs=1)
    assert table["样本数"].sum() == len(loans)
    np.testing.assert_allclose(table["KS贡献"], abs(table["累积坏样本率"] - table["累积好样本率"]), atol=1e-10)
    assert table["累积坏样本率"].max() == pytest.approx(1)
    assert table["累积好样本率"].max() == pytest.approx(1)


def test_real_rule_amount_metrics_match_raw_subset(loans):
    rule = Rule("`衡枢鉴真分老客版` < 0.08")
    mask = loans["衡枢鉴真分老客版"].to_numpy() < 0.08
    amount = loans["放款金额"].to_numpy(dtype=float)
    y = loans.FPD.to_numpy()
    result = RuleMetrics(n_jobs=1).evaluate_rule(rule, loans[FEATURES], loans.FPD, amount_train=loans["放款金额"])
    selected = amount[mask].sum()
    loss = amount[mask & (y == 1)].sum() / selected
    assert result["训练_命中样本数"] == int(mask.sum())
    assert result["训练_命中金额"] == pytest.approx(selected)
    assert result["训练_命中损失率"] == pytest.approx(loss)
    assert result["训练_命中金额LIFT值"] == pytest.approx(loss / (amount[y == 1].sum() / amount.sum()))


def test_real_rule_train_test_bad_rate_difference_uses_count_population(loans):
    train, test = loans.iloc[:600], loans.iloc[600:]
    result = RuleMetrics(n_jobs=1).evaluate_rule(
        Rule("`衡枢鉴真分老客版` < 0.08"),
        train[FEATURES],
        train.FPD,
        test[FEATURES],
        test.FPD,
        amount_train=train["放款金额"],
        amount_test=test["放款金额"],
    )
    expected = train.loc[train[FEATURES[0]] < 0.08, "FPD"].mean() - test.loc[test[FEATURES[0]] < 0.08, "FPD"].mean()
    assert result["badrate_diff"] == pytest.approx(expected)
