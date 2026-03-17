# -*- coding: utf-8 -*-
"""
可视化模块 (viz).

提供评分卡开发过程中常用的可视化功能，包括：
- 特征分箱图 (bin_plot)
- 特征相关性热力图 (corr_plot)  
- KS/ROC曲线图 (ks_plot)
- 特征分布图 (hist_plot)
- PSI稳定性分析图 (psi_plot)
- CSI特征稳定性图 (csi_plot)
- DataFrame表格图 (dataframe_plot)
- 时间分布图 (distribution_plot)

辅助函数已移至 utils 模块：
- init_setting -> hscredit.utils.init_setting
- feature_describe -> hscredit.utils.feature_describe
- round_float -> hscredit.utils.round_float
- feature_bins -> hscredit.utils.feature_bins

参考 scorecardpipeline 实现优化而来。
"""

from .binning_plots import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
)

__all__ = [
    # 画图方法
    "bin_plot",
    "corr_plot", 
    "ks_plot",
    "hist_plot",
    "psi_plot",
    "dataframe_plot",
    "distribution_plot",
]
