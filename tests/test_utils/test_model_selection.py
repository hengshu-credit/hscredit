"""时间、OOT 和 Group 切分工具测试."""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.model_selection import (
    group_train_test_split,
    oot_split,
    time_train_test_split,
)


@pytest.fixture
def dated_data():
    return pd.DataFrame(
        {
            "申请日期": pd.date_range("2024-01-01", periods=10, freq="D"),
            "客户编号": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "特征": np.arange(10),
            "目标": [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        },
        index=np.arange(100, 110),
    )


def test_time_split_by_ratio_has_no_time_leakage(dated_data):
    shuffled = dated_data.sample(frac=1, random_state=7)
    train, test = time_train_test_split(shuffled, "申请日期", test_size=0.3)

    assert len(train) == 7
    assert len(test) == 3
    assert train["申请日期"].max() < test["申请日期"].min()
    assert test["特征"].tolist() == [7, 8, 9]


def test_time_split_with_cutoff_and_gap(dated_data):
    train, test = time_train_test_split(
        dated_data,
        "申请日期",
        cutoff="2024-01-07",
        gap=2,
    )

    assert train["特征"].tolist() == [0, 1, 2, 3]
    assert test["特征"].tolist() == [6, 7, 8, 9]


def test_oot_split_respects_start_and_end(dated_data):
    development, oot = oot_split(
        dated_data,
        "申请日期",
        oot_start="2024-01-07",
        oot_end="2024-01-09",
    )

    assert development["特征"].tolist() == [0, 1, 2, 3, 4, 5]
    assert oot["特征"].tolist() == [6, 7, 8]


def test_group_split_has_no_group_overlap(dated_data):
    X = dated_data[["客户编号", "特征"]]
    y = dated_data["目标"]
    X_train, X_test, y_train, y_test = group_train_test_split(
        X,
        y,
        group_col="客户编号",
        test_size=0.4,
        random_state=42,
    )

    assert set(X_train["客户编号"]).isdisjoint(set(X_test["客户编号"]))
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)


def test_group_split_requires_group_information(dated_data):
    with pytest.raises(ValueError, match="必须提供 groups 或 group_col"):
        group_train_test_split(dated_data, test_size=0.2)


def test_split_functions_are_top_level_exports():
    import hscredit

    assert hscredit.time_train_test_split is time_train_test_split
    assert hscredit.oot_split is oot_split
    assert hscredit.group_train_test_split is group_train_test_split
