"""日期频率的新旧别名、pandas 分组边界与分析入口回归。"""

from datetime import timedelta
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from hscredit.utils.dates import date_groups, normalize_freq

ALIASES = [
    ("M", "ME"),
    ("BM", "BME"),
    ("CBM", "CBME"),
    ("SM-15", "SME-15"),
    ("Q", "QE"),
    ("2Q-MAR", "2QE-MAR"),
    ("BQ-JUN", "BQE-JUN"),
    ("A", "YE"),
    ("Y-JUN", "YE-JUN"),
    ("BA-JUN", "BYE-JUN"),
    ("BY", "BYE"),
    ("AS-JUL", "YS-JUL"),
    ("BAS-JAN", "BYS-JAN"),
    ("BH", "bh"),
    ("CBH", "cbh"),
    ("H", "h"),
    ("T", "min"),
    ("S", "s"),
    ("L", "ms"),
    ("U", "us"),
    ("N", "ns"),
    ("2H30T", "2h30min"),
    ("1D2H", "1D2h"),
    ("1.5H", "1.5h"),
]


@pytest.mark.parametrize("old,new", ALIASES)
def test_old_and_new_aliases_resolve_without_deprecation_warnings(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert normalize_freq(old) == normalize_freq(new)


@pytest.mark.parametrize("freq", ["MS", "QS-APR", "BMS", "CBMS", "SMS-15", "BQS", "W-MON", "WOM-2MON"])
def test_unchanged_pandas_frequencies_are_not_rewritten(freq):
    assert normalize_freq(freq) == pd.tseries.frequencies.to_offset(freq)


@pytest.mark.parametrize(
    "freq", [pd.offsets.MonthBegin(2), pd.offsets.Week(weekday=2), pd.Timedelta(minutes=15), timedelta(hours=3)]
)
def test_offset_and_timedelta_inputs(freq):
    assert normalize_freq(freq) == pd.tseries.frequencies.to_offset(freq)


@pytest.mark.parametrize(
    "freq", ["MS", "2ME", "QS-APR", "YE-JUN", "YS-JUL", "BME", "BMS", "SME-15", "WOM-2MON", "2W-MON", "B"]
)
def test_calendar_groups_match_pandas_with_unsorted_duplicate_indexes(freq):
    dates = pd.Series(
        pd.to_datetime(["2024-04-01", "2024-01-01", "2024-01-31", "2024-02-01", None, "2024-01-31"]),
        index=[1, 1, 3, 0, 0, 2],
    )
    actual = date_groups(dates, freq, timestamps=True)
    frame = pd.DataFrame({"date": dates.to_numpy(), "position": np.arange(len(dates))})
    expected = pd.Series(np.nan, index=dates.index, name=dates.name, dtype=object)
    for label, group in frame.groupby(pd.Grouper(key="date", freq=normalize_freq(freq))):
        expected.iloc[group.position.to_numpy()] = label
    assert_series_equal(actual, expected)


@pytest.mark.parametrize("freq", ["7h", "17min", "1.5h", "2D", "1D2H30T"])
@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai", "America/New_York"])
def test_fixed_groups_match_native_origin_and_dst(freq, timezone):
    dates = pd.Series(pd.date_range("2024-03-09 21:13", periods=60, freq="h", tz=timezone))[::-1]
    actual = date_groups(dates, freq, timestamps=True)
    frame = pd.DataFrame({"date": dates.to_numpy(), "position": np.arange(len(dates))})
    expected = pd.Series(np.nan, index=dates.index, dtype=object)
    for label, group in frame.groupby(pd.Grouper(key="date", freq=normalize_freq(freq))):
        expected.iloc[group.position.to_numpy()] = label
    assert_series_equal(actual, expected)


def test_sparse_nanosecond_groups_do_not_expand_empty_intervals():
    dates = pd.Series(pd.to_datetime(["2024-01-01 00:00:00.000000001", "2025-01-01 00:00:00.000000002"]))
    labels = date_groups(dates, "ns")
    assert labels.iloc[0].endswith("000000001")
    assert labels.iloc[1].endswith("000000002")
    assert len(labels) == 2


def test_custom_calendar_is_kept_and_missing_dates_stay_missing():
    offset = pd.offsets.CustomBusinessDay(weekmask="Mon Tue Wed Thu", holidays=["2024-01-01"])
    assert normalize_freq(offset) is offset
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", None]))
    groups = date_groups(dates, offset, timestamps=True)
    assert groups.iloc[0] == pd.Timestamp("2023-12-28")
    assert groups.iloc[1] == pd.Timestamp("2024-01-02")
    assert pd.isna(groups.iloc[2])
    assert date_groups(pd.Series([pd.NaT]), "MS").isna().all()


def test_fall_back_dst_hours_keep_distinct_timezone_labels():
    dates = pd.Series(pd.to_datetime(["2024-11-03 05:30Z", "2024-11-03 06:30Z"]).tz_convert("America/New_York"))
    labels = date_groups(dates, "H")
    assert labels.is_unique
    assert labels.iloc[0].endswith("-04:00")
    assert labels.iloc[1].endswith("-05:00")


@pytest.mark.parametrize(
    "freq,date,label",
    [
        ("M", "2024-03-31", "2024-03"),
        ("Q-MAR", "2024-03-31", "2024Q4"),
        ("Y-JUN", "2024-06-30", "2024"),
        ("W-MON", "2024-01-08", "2024-01-02/2024-01-08"),
    ],
)
def test_calendar_display_labels_are_stable_across_pandas_versions(freq, date, label):
    from hscredit.utils.dates import format_time_labels

    assert format_time_labels([pd.Timestamp(date)], freq) == [label]


def test_midnight_business_hour_keeps_time_and_timezone():
    from hscredit.utils.dates import format_time_labels

    offset = pd.offsets.BusinessHour(start="00:00", end="04:00")
    labels = format_time_labels([pd.Timestamp("2024-01-01", tz="UTC")], offset)
    assert labels == ["2024-01-01 00:00:00+00:00"]


@pytest.mark.parametrize("freq", ["", "invalid", None, 3, "0h", "-1D"])
def test_invalid_or_nonpositive_grouping_frequencies_have_chinese_errors(freq):
    with pytest.raises(ValueError, match="freq.*频率"):
        normalize_freq(freq)


def _data():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-02-01 00:00"]),
            "score": [4.0, 1.0, 2.0, 3.0],
            "target": [0, 0, 1, 1],
        },
        index=[2, 1, 1, 0],
    )


@pytest.mark.parametrize("freq", ["MS", "QS-APR", "YS-JUL", "BME", "2ME", "7h"])
def test_eda_and_financial_metrics_use_pandas_group_sizes(freq):
    from hscredit.core.eda.target import bad_rate_trend
    from hscredit.core.metrics.finance import badrate_trend

    data = _data()
    expected = data.groupby(pd.Grouper(key="date", freq=normalize_freq(freq))).target.agg(["count", "sum"])
    expected = expected.loc[expected["count"] > 0]
    for actual in [
        bad_rate_trend(data, date="date", target="target", freq=freq),
        badrate_trend(data.target, data.date, freq=freq),
    ]:
        assert actual["样本数"].tolist() == expected["count"].tolist()
        assert actual["坏样本数"].tolist() == expected["sum"].tolist()


@pytest.mark.parametrize("old,new", [("H", "h"), ("T", "min"), ("S", "s"), ("L", "ms"), ("U", "us"), ("N", "ns")])
def test_subday_distribution_keeps_unique_labels_and_does_not_multiply_rows(old, new):
    from hscredit.core.viz import distribution_plot

    dates = pd.date_range("2024-01-01", periods=4, freq=normalize_freq(new))
    data = pd.DataFrame({"date": dates, "y": [0, 1, 0, 1], "z": [1, 1, 0, 0]})
    try:
        actual = distribution_plot(data, date="date", target={"甲": data.y, "乙": data.z}, freq=old, result=True)
        expected = distribution_plot(data, date="date", target={"甲": data.y, "乙": data.z}, freq=new, result=True)
        assert_frame_equal(actual, expected)
        assert len(actual) == 4
        assert actual["日期"].is_unique
        assert actual["样本总数"].tolist() == [1, 1, 1, 1]
    finally:
        plt.close("all")


def test_month_start_rule_flow_and_plots():
    from hscredit.core.rules import Rule, RuleFlow
    from hscredit.core.viz import bad_rate_trend_plot, approval_rate_trend_plot, feature_trend_by_time

    data = _data().reset_index(drop=True)
    flow = RuleFlow([Rule("score > 2")])
    report = flow.report(data, date="date", freq="MS")
    assert "统计周期" in report.columns
    try:
        bad_rate_trend_plot(data, date="date", target="target", freq="MS")
        approval_rate_trend_plot(data, date="date", decision_col="target", freq="MS")
        feature_trend_by_time(data, "score", date="date", freq="MS")
    finally:
        plt.close("all")
