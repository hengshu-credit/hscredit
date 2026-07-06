"""IVSelector 分箱筛选测试。"""

import numpy as np
import pandas as pd

from hscredit.core.selectors import IVSelector


def test_iv_selector_uses_default_optimal_binning_when_configured():
    """传入空 binning_params 时，应使用默认 MDLP 参数先分箱再计算 IV。"""
    x = np.linspace(-4.0, 4.0, 400)
    X = pd.DataFrame({
        'signal': x,
        'noise': np.sin(x * 7.0),
    })
    y = pd.Series((x > 0).astype(int))

    selector = IVSelector(threshold=0.02, binning_params={})
    selector.fit(X, y)

    assert selector.binner_.method == 'mdlp'
    assert selector.binner_.max_n_bins == 10
    assert selector.binner_.min_bin_size == 0.01
    assert selector.scores_['signal'] > 0.02
    assert 'signal' in selector.selected_features_


def test_iv_selector_binning_params_override_defaults():
    """用户配置应覆盖默认 OptimalBinning 参数。"""
    x = np.linspace(-4.0, 4.0, 400)
    X = pd.DataFrame({'signal': x})
    y = pd.Series((x > 0).astype(int))

    selector = IVSelector(
        threshold=0.02,
        binning_params={
            'method': 'quantile',
            'max_n_bins': 4,
            'min_bin_size': 0.05,
        },
    )
    selector.fit(X, y)

    assert selector.binner_.method == 'quantile'
    assert selector.binner_.max_n_bins == 4
    assert selector.binner_.min_bin_size == 0.05
