"""提交前复核的类别规则与遗传分箱指标独立回归验证。"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hscredit.core.binning import GeneticBinning, OptimalBinning


def _groups(values, bad_counts, count=50, dtype=None):
    X = pd.DataFrame({"枚举": pd.Series(np.repeat(values, count), dtype=dtype)})
    y = pd.Series(np.concatenate([np.r_[np.ones(b), np.zeros(count - b)] for b in bad_counts]).astype(int))
    return X, y


def _binner(method="quantile", **kwargs):
    options = dict(
        method=method,
        max_n_bins=3,
        min_n_bins=2,
        min_bin_size=0.05,
        max_bin_size=None,
        monotonic=False,
        n_jobs=1,
        random_state=42,
    )
    if method in ("or_tools", "cp_sat"):
        options.update(n_prebins=8, time_limit=5)
    if method == "genetic":
        options.update(population_size=12, generations=6)
    if method == "kernel_density":
        options.update(n_grid_points=128)
    return OptimalBinning(**{**options, **kwargs})


def test_nullable_integer_categories_preserve_observed_values_and_missing():
    X, y = _groups([10, 20, 30], [40, 5, 25], dtype="Int64")
    X.loc[::5, "枚举"] = pd.NA
    X.index = y.index = np.repeat(np.arange(75), 2)
    model = _binner(cat_cutoff=3).fit(X, y)
    known = model.transform(X)["枚举"].to_numpy()
    assert np.all(known[X["枚举"].notna()] >= 0)
    assert np.all(known[X["枚举"].isna()] == -1)
    prediction = pd.DataFrame({"枚举": pd.Series([10, 40, pd.NA], dtype="Int64")})
    actual = model.transform(prediction)["枚举"].to_numpy()
    expected = model.transform(prediction.astype(object))["枚举"].to_numpy()
    assert_array_equal(actual, expected)
    assert actual[0] >= 0 and actual[1] == -3 and actual[2] == -1


@pytest.mark.parametrize("method", OptimalBinning.VALID_METHODS)
@pytest.mark.parametrize("special", [0, 1])
def test_absent_numeric_special_code_cannot_change_categorical_learning(method, special):
    X, y = _groups(list("ABCD"), [5, 20, 35, 45])
    baseline = _binner(method).fit(X, y)
    actual = _binner(method, special_codes=[special]).fit(X, y)
    assert actual.export_rules() == baseline.export_rules()
    assert_array_equal(actual.transform(X), baseline.transform(X))


@pytest.mark.parametrize("claim_integer", [False, True])
def test_categorical_special_codes_respect_types_and_user_owned_groups(claim_integer):
    groups = [[1.0], [True], ["1"], ["A"]]
    if claim_integer:
        groups.insert(0, [1])
    model = _binner(special_codes=[np.int64(1)]).import_rules({"枚举": groups})
    X = pd.DataFrame({"枚举": pd.Series([1, 1.0, True, "1", "A", None], dtype=object)})
    expected = [0, 1, 2, 3, 4, -1] if claim_integer else [-2, 0, 1, 2, 3, -1]
    assert_array_equal(model.transform(X)["枚举"], expected)


def test_imported_category_rules_learn_woe_but_explicit_woe_is_preserved():
    X, y = _groups([10, 20, 30], [40, 5, 25], dtype="category")
    rules = {"枚举": [[10], [20], [30]]}
    model = _binner().import_rules(rules).fit(X, y)
    bad = np.array([40, 5, 25])
    good = 50 - bad
    expected = np.log((bad / bad.sum()) / (good / good.sum()))
    assert_allclose(model.transform(X.iloc[[0, 50, 100]], metric="woe")["枚举"], expected)
    explicit = {**rules, "_woe_maps_": {"枚举": {0: 0.3, 1: -0.4, 2: 0.2}}}
    loaded = _binner().load(explicit).fit(X, y)
    assert_allclose(loaded.transform(X.iloc[[0, 50, 100]], metric="woe")["枚举"], [0.3, -0.4, 0.2])


@pytest.mark.parametrize("method", ["quantile", "tree", "or_tools"])
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_mixed_features_preserve_numeric_special_codes_and_parallel_state(method, backend):
    X, y = _groups(list("ABCD"), [5, 20, 35, 45])
    X["数值"] = np.arange(len(X), dtype=float)
    X.loc[::10, "数值"] = 0
    model = _binner(method, special_codes=[0], n_jobs=2, parallel_backend=backend).fit(X, y)
    baseline = _binner(method).fit(X[["枚举"]], y)
    assert model.splits_["枚举"] == baseline.splits_["枚举"]
    assert model.special_codes == [0]
    result = model.transform(X)
    assert_array_equal(result["数值"].eq(-2), X["数值"].eq(0))
    assert np.all(result["枚举"] >= 0)


def test_real_product_categories_with_nullable_codes_and_absent_special_value():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    data = pd.read_excel(path)
    codes = pd.Series(pd.factorize(data["商品类别"])[0] + 10, dtype="Int64")
    codes.iloc[::9] = pd.NA
    X = codes.to_frame("枚举")
    model = _binner(cat_cutoff=int(codes.nunique()), special_codes=[0]).fit(X, data.FPD)
    bins = model.transform(X)["枚举"]
    assert not bins.eq(-3).any()
    raw = pd.DataFrame({"箱号": bins, "标签": data.FPD}).groupby("箱号")["标签"].agg(["size", "sum"])
    table = model.get_bin_table("枚举").set_index("分箱").loc[raw.index]
    assert_array_equal(table["样本总数"], raw["size"])
    assert_array_equal(table["坏样本数"], raw["sum"])


@pytest.mark.parametrize(
    "bins,y",
    [([0, 0, 1, 1], [0, 0, 1, 1]), ([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 1, 1, 1, 2, 2], [1, 1, 0, 1, 0, 0, 1])],
)
def test_genetic_native_objectives_match_raw_row_definitions(bins, y):
    bins, y = np.array(bins), np.array(y)
    model = GeneticBinning()
    expected_ks = max(abs(np.mean(bins[y == 0] <= edge) - np.mean(bins[y == 1] <= edge)) for edge in np.unique(bins))
    rates = np.array([np.mean(y[bins == group]) for group in bins])
    pairwise = rates[y == 1, None] - rates[y == 0]
    expected_gini = 2 * np.mean((pairwise > 0) + 0.5 * (pairwise == 0)) - 1
    good = np.array([np.sum((bins == group) & (y == 0)) for group in np.unique(bins)], dtype=float)
    bad = np.array([np.sum((bins == group) & (y == 1)) for group in np.unique(bins)], dtype=float)
    good[good == 0] = bad[bad == 0] = 1e-10
    gd, bd = good / good.sum(), bad / bad.sum()
    expected_iv = np.sum((bd - gd) * np.log(bd / gd))
    assert_allclose(model._calculate_ks(bins, y), expected_ks, atol=1e-12)
    assert_allclose(model._calculate_gini(bins, y), expected_gini, atol=1e-12)
    assert_allclose(model._calculate_iv(bins, y), expected_iv, atol=1e-12, rtol=1e-12)
