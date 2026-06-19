"""分箱基类绘图接口测试."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from hscredit.core.binning import QuantileBinning
from hscredit.exceptions import FeatureNotFoundError, NotFittedError


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    x = rng.normal(size=200)
    y = (x + rng.normal(scale=0.8, size=200) > 0).astype(int)
    return pd.DataFrame({"评分": x}), pd.Series(y)


def test_plot_returns_figure(sample_data):
    X, y = sample_data
    binner = QuantileBinning(max_n_bins=5).fit(X, y)

    figure = binner.plot("评分")

    assert figure.axes


def test_plot_requires_fitted_binner():
    with pytest.raises(NotFittedError, match="尚未拟合"):
        QuantileBinning().plot("评分")


def test_plot_rejects_unknown_feature(sample_data):
    X, y = sample_data
    binner = QuantileBinning(max_n_bins=5).fit(X, y)

    with pytest.raises(FeatureNotFoundError, match="不存在"):
        binner.plot("未知特征")
