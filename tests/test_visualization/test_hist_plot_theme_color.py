import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from hscredit.core.viz.binning_plots import hist_plot
from hscredit.core.viz.utils import DEFAULT_COLORS


def test_hist_plot_uses_theme_color_without_y_true():
    score = np.linspace(0, 1, 50)

    fig = hist_plot(score, bins=5, kde=False, title='demo')
    ax = fig.axes[0]

    assert ax.collections, 'hist_plot 未生成直方图图元'
    assert to_hex(ax.collections[0].get_facecolor()[0], keep_alpha=False).lower() == DEFAULT_COLORS[0].lower()

    plt.close(fig)


def test_hist_plot_handles_infinite_values_without_mutating_order():
    score = pd.Series([0.1, np.inf, 0.4, -np.inf, 0.8], index=[5, 4, 3, 2, 1])
    original = score.copy(deep=True)

    fig = hist_plot(score, bins=3, kde=False)

    pd.testing.assert_series_equal(score, original)
    assert fig.axes[0].collections
    vertices = np.concatenate(
        [
            path.vertices
            for collection in fig.axes[0].collections
            for path in collection.get_paths()
        ]
    )
    assert not np.isinf(vertices).any()

    plt.close(fig)
