"""技能返回的标签摘要记录实际方向、剔灰配置和标签组合。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.skills_runtime.operations.binning import _overdue_summary


@pytest.mark.parametrize("container", [list, tuple, np.array, pd.Index, pd.Series])
def test_overdue_summary_matches_deduplicated_combinations(container):
    result = _overdue_summary({"overdue": container(["MOB1", "MOB1", "MOB2"]), "dpds": [3.0, "3", 0]})
    assert result["label_combinations"] == [
        {"overdue": "MOB1", "dpd": 3},
        {"overdue": "MOB1", "dpd": 0},
        {"overdue": "MOB2", "dpd": 3},
        {"overdue": "MOB2", "dpd": 0},
    ]


@pytest.mark.parametrize("explicit", [True, False, None])
@pytest.mark.parametrize("target_mode", [True, False])
def test_target_summary_preserves_grey_precedence(explicit, target_mode):
    labels = {"target": {"自定义": [0, -1, 1]}} if target_mode else {"overdue": "MOB1", "dpds": 3}
    result = _overdue_summary(
        {
            **labels,
            "del_grey": explicit,
            "overdue_operator": None,
            "bin_params": {"del_grey": True, "overdue_operator": ">="},
        }
    )
    assert result["del_grey"] is (True if explicit is None else explicit)
    if target_mode:
        assert result["target_names"] == ["自定义"]
    else:
        assert result["overdue_operator"] == ">="


def test_paired_summary_keeps_matching_field_threshold_pairs():
    result = _overdue_summary(
        {"overdue": pd.Index(["MOB1", "MOB2", "MOB1"]), "dpds": [3, 3, 7]},
        paired=True,
        default_operator=">=",
    )
    assert result["label_combinations"] == [
        {"overdue": "MOB1", "dpd": 3},
        {"overdue": "MOB2", "dpd": 3},
        {"overdue": "MOB1", "dpd": 7},
    ]
    assert result["overdue_operator"] == ">="
