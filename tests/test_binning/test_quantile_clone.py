"""QuantileBinning sklearn clone 契约回归测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import QuantileBinning
from hscredit.report import feature_bin_stats


@pytest.mark.parametrize(
    "quantiles",
    [None, [0.0, 0.2, 0.8, 1.0], [0.2, 0.8]],
    ids=["默认分位点", "完整首尾", "自动补齐首尾"],
)
def test_quantile_binning_clone_keeps_constructor_quantiles_and_fitted_rules(quantiles):
    """构造器保留公开参数，克隆后的分箱器仍可拟合并导出规则。"""
    data = pd.DataFrame({"评分": np.arange(20, dtype=float)})
    target = pd.Series([0, 1] * 10, name="FPD")

    binner = QuantileBinning(quantiles=quantiles)

    cloned = clone(binner)

    assert binner.quantiles is quantiles
    cloned.fit(data, target)

    assert cloned._is_fitted
    assert "评分" in cloned.export_rules()
    if quantiles is not None:
        assert cloned.quantiles == quantiles
        assert cloned.quantiles_ == [0.0, 0.2, 0.8, 1.0]


def test_feature_bin_stats_quantile_accepts_quantiles_without_endpoints():
    """公开报告接口可使用缺少首尾的自定义分位点完成等频分箱。"""
    data = pd.DataFrame(
        {
            "评分": np.arange(20, dtype=float),
            "FPD": [0, 1] * 10,
        }
    )

    table, rules = feature_bin_stats(
        data,
        feature="评分",
        target="FPD",
        method="quantile",
        quantiles=[0.2, 0.8],
        return_rules=True,
    )

    assert not table.empty
    assert "评分" in rules
    assert rules["评分"] == pytest.approx([3.8, 15.2])
