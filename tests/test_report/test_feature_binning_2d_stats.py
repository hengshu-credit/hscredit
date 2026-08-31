"""二维特征分箱统计报告回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning2D
from hscredit.report import feature_bin_stats_2d


def test_feature_bin_stats_2d_removes_grey_per_target_and_returns_binner():
    """二维分箱复用单一灰样本掩码或公共总数时，本测试必须失败。"""
    data = pd.DataFrame(
        {
            "特征1": np.arange(12, dtype=float),
            "特征2": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "MOB1": [0, 2, 0, 5, 0, 2, 5, 8, 0, 4, 2, 8],
        }
    )

    table, binner = feature_bin_stats_2d(
        data,
        features=["特征1", "特征2"],
        overdue="MOB1",
        dpds=[1, 3],
        del_grey=True,
        margins=True,
        return_binner=True,
        method="quantile",
        max_n_bins=2,
        min_bin_size=0.01,
        n_jobs=1,
    )

    assert isinstance(binner, OptimalBinning2D)
    assert isinstance(table.columns, pd.MultiIndex)
    total = table.loc[table[("分箱详情", "分箱标签")] == "合计"].iloc[0]
    assert total[("MOB1_1+", "样本总数")] == 12
    assert total[("MOB1_1+", "好样本数")] == 4
    assert total[("MOB1_1+", "坏样本数")] == 8
    assert total[("MOB1_1+", "坏样本率")] == pytest.approx(8 / 12)
    assert total[("MOB1_3+", "样本总数")] == 9
    assert total[("MOB1_3+", "好样本数")] == 4
    assert total[("MOB1_3+", "坏样本数")] == 5
    assert total[("MOB1_3+", "坏样本率")] == pytest.approx(5 / 9)
