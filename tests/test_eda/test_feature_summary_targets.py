"""特征摘要的标签口径、共享统计与多级表头。"""

import operator

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from hscredit import feature_summary


@pytest.fixture
def orders():
    rng = np.random.default_rng(700)
    days = [-1, 0, 1, 29, 30, 31, 60, np.nan]
    frame = pd.DataFrame({"x": rng.normal(size=160), "MOB3": rng.choice(days, 160), "MOB6": rng.choice(days, 160)})
    frame.loc[::7, "x"] = np.nan
    return frame


@pytest.mark.parametrize("op,compare", [(">", operator.gt), (">=", operator.ge), ("<", operator.lt), ("<=", operator.le)])
@pytest.mark.parametrize("missing", ["drop", "positive", "negative"])
def test_each_overdue_block_uses_its_own_population(orders, op, compare, missing):
    orders.index = np.arange(len(orders)) // 2
    options = dict(n_jobs=1, fillna=-99999, val_df=orders.iloc[::2])
    table = orders.summary(overdue=["MOB3", "MOB6"], dpds=30, overdue_operator=op, del_grey=True, handle_missing_target=missing, **options)
    common = feature_summary(orders[["x"]], **options)
    names = [col + ("@" if op == ">" else op) + "30" for col in ["MOB3", "MOB6"]]
    assert table.columns.get_level_values(0).unique().tolist() == ["统计详情", *names, "分布详情"]
    for group in ["统计详情", "分布详情"]:
        assert_frame_equal(table[group], common[table[group].columns])
    for col, name in zip(["MOB3", "MOB6"], names):
        days = orders[col]
        y = compare(days, 30).astype(float).mask(days.isna(), {"drop": np.nan, "positive": 1.0, "negative": 0.0}[missing])
        if op == ">":
            y = y.mask((days > 0) & (days <= 30))
        elif op == ">=":
            y = y.mask((days > 0) & (days < 30))
        valid = y.notna()
        expected = feature_summary(orders.loc[valid, ["x"]], y=y[valid], **options)
        assert_frame_equal(table[name], expected[["IV", "KS", "AUC", "趋势"]])


def test_explicit_targets_align_once_and_match_column_names(orders):
    orders["三期"] = orders.MOB3.map({0: 0, 1: 2, 30: 1})
    orders["六期"] = orders.MOB6.map({0: 0, 1: 1, 30: 2})
    calls = []

    def six(frame):
        calls.append(len(frame))
        return frame["六期"].iloc[::-1]

    explicit = {"三期": orders["三期"].iloc[::-1], "六期": six}
    via_dict = feature_summary(orders, features="x", target=explicit, bin_params={"del_grey": True, "handle_missing_target": "positive"}, n_jobs=1)
    assert calls == [len(orders)]
    feature_frame = orders[["x", "三期", "六期"]]
    by_name = feature_frame.summary(target=["三期", "六期"], del_grey=True, handle_missing_target="positive", n_jobs=1)
    assert_frame_equal(by_name, via_dict)
    assert by_name["统计详情", "特征名"].tolist() == ["x"]
    for name in explicit:
        y = orders[name].fillna(1)
        valid = y.isin([0, 1])
        expected = orders.loc[valid, ["x"]].summary(y=y[valid], n_jobs=1)
        assert_frame_equal(by_name[name], expected[by_name[name].columns])
    single = feature_frame.summary(features="x", target=["三期"], handle_missing_target="negative", del_grey=False, n_jobs=1)
    assert single.columns.nlevels == 1
    assert_frame_equal(single, orders[["x"]].summary(y=orders["三期"].eq(1), n_jobs=1))
    assert_series_equal(orders["三期"], orders.MOB3.map({0: 0, 1: 2, 30: 1}), check_names=False)


def test_external_days_series_dataframe_and_threshold_order(orders):
    days = orders[["MOB6", "MOB3"]].iloc[::-1]
    options = dict(dpds=[30, 0, 30], n_jobs=1)
    expected = orders.summary(overdue=["MOB6", "MOB3"], **options)
    assert_frame_equal(orders.x.summary(overdue=days, **options), expected)
    parallel = orders[["x"]].assign(x2=orders.x).summary(overdue=days, n_jobs=2, parallel_backend="loky", dpds=[30, 0])
    assert_frame_equal(parallel.iloc[:1].reset_index(drop=True), expected)
    assert expected.columns.get_level_values(0).unique().tolist() == ["统计详情", "MOB6@30", "MOB6@0", "MOB3@30", "MOB3@0", "分布详情"]
    one = orders.x.summary(overdue=days.MOB3, dpds=30, n_jobs=1)
    assert one.columns.nlevels == 1
    assert_frame_equal(one, orders.summary(features="x", overdue="MOB3", dpds=[30, 30], n_jobs=1))
    records = orders.x.summary(overdue=days, dpds=30, n_jobs=1, return_type="dict")
    assert records[0]["MOB3@30", "KS"] == one.at[0, "KS"]


def test_unobserved_label_keeps_other_labels_and_shared_stats(orders):
    table = orders.x.summary(target={"未表现": [np.nan] * len(orders), "已表现": orders.MOB6.gt(30)}, n_jobs=1)
    assert table["统计详情", "样本数"].iat[0] == len(orders)
    assert table["未表现"][["IV", "KS", "AUC"]].isna().all().all()
    assert table["未表现", "趋势"].iat[0] == "unknown"
    assert table["已表现", "AUC"].notna().all()


def test_failed_model_still_has_the_same_columns_as_other_labels(orders):
    target = {"未表现": [np.nan] * len(orders), "已表现": orders.MOB6.gt(30)}
    with pytest.warns(UserWarning, match="未表现"):
        report = feature_summary(orders, features="x", target=target, model_type="randomforest", model_params={"n_estimators": 3}, return_result=True, n_jobs=1)
    table = report["特征摘要"]
    assert not report.complete
    assert table["未表现"].columns.equals(table["已表现"].columns)
    assert pd.isna(table["未表现", "randomforest重要性"].iat[0])
    assert table["已表现", "randomforest重要性"].notna().all()


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(target="MOB3", overdue="MOB3", dpds=30), "不能同时"),
        (dict(y="MOB3", target="MOB6"), "不能同时"),
        (dict(overdue="MOB6"), "同时传入"),
        (dict(dpds=30), "同时传入"),
        (dict(overdue="MOB3", dpds=30, overdue_operator="=="), "overdue_operator"),
        (dict(target="MOB3", handle_missing_target="raise", del_grey=True), "缺失"),
        (dict(overdue="MOB6", dpds=30, handle_missing_target="raise"), "缺失"),
        (dict(target="MOB3", handle_missing_target="invalid"), "handle_missing_target"),
        (dict(target={"统计详情": lambda d: d.MOB3, "A": lambda d: d.MOB6}), "保留"),
    ],
)
def test_invalid_labels_fail_before_summary(orders, options, message):
    with pytest.raises(ValueError, match=message):
        orders.summary(features="x", n_jobs=1, **options)


def test_multilabel_importance_fits_each_valid_population(orders, monkeypatch):
    from hscredit.core import models

    fitted = []

    class Model:
        def __init__(self, **params):
            pass

        def fit(self, X, y):
            assert X.index.equals(y.index)
            fitted.append(y.copy())

        def get_feature_importances(self):
            return pd.Series({"x": 0.75})

    monkeypatch.setattr(models, "RandomForest", Model)
    report = feature_summary(orders, overdue=["MOB3", "MOB6"], dpds=30, model_type="randomforest", models={"外部模型": Model()}, del_grey=True, mode="strict", return_result=True, n_jobs=1)
    table = report["特征摘要"]
    assert report.complete and len(fitted) == 2
    assert table["统计详情", "外部模型重要性"].iat[0] == 0.75
    for col, y in zip(["MOB3", "MOB6"], fitted):
        days = orders[col].dropna()
        expected = days[(days <= 0) | (days > 30)].gt(30).astype(float)
        assert_series_equal(y, expected, check_names=False)
        assert table[col + "@30", "randomforest重要性"].iat[0] == 0.75
