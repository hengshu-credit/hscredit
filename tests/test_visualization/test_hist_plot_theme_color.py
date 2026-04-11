import matplotlib
matplotlib.use('Agg')

import numpy as np
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
