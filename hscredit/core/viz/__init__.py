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
- 逻辑回归系数误差图 (plot_weights)

金融风控专用图表 (risk_plots)：
- ROC曲线图 (roc_plot)
- PR曲线图 (pr_plot)
- Lift提升图 (lift_plot)
- Gain增益图 (gain_plot)
- 混淆矩阵图 (confusion_matrix_plot)
- 校准曲线图 (calibration_plot)
- 评分分布对比图 (score_dist_plot)
- 评分分箱效果图 (score_bin_plot)
- 决策阈值分析图 (threshold_analysis_plot)
- 策略效果对比图 (strategy_compare_plot)
- Vintage账龄曲线图 (vintage_plot)
- 特征重要性图 (feature_importance_plot)
- 审批通过率趋势图 (approval_rate_trend_plot)
- 逾期率趋势图 (bad_rate_trend_plot)

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
    bin_trend_plot,
    batch_bin_trend_plot,
    overdues_bin_plot,
)

from .model_plots import plot_weights

# 金融风控专用图表
from .risk_plots import (
    # 模型评估
    roc_plot,
    pr_plot,
    lift_plot,
    gain_plot,
    confusion_matrix_plot,
    calibration_plot,
    # 评分卡
    score_dist_plot,
    score_bin_plot,
    # 风控策略
    threshold_analysis_plot,
    strategy_compare_plot,
    vintage_plot,
    feature_importance_plot,
    approval_rate_trend_plot,
    bad_rate_trend_plot,
)

# 导出工具函数供外部使用
from .utils import (
    setup_axis_style,
    save_figure,
    get_or_create_ax,
    create_legend,
    format_bin_label,
    DEFAULT_COLORS,
)

__all__ = [
    # 特征分箱相关
    "bin_plot",
    "corr_plot",
    "ks_plot",
    "hist_plot",
    "psi_plot",
    "dataframe_plot",
    "distribution_plot",
    # 特征趋势分析
    "bin_trend_plot",
    "batch_bin_trend_plot",
    "overdues_bin_plot",
    # 模型相关
    "plot_weights",
    # 模型评估
    "roc_plot",
    "pr_plot",
    "lift_plot",
    "gain_plot",
    "confusion_matrix_plot",
    "calibration_plot",
    # 评分卡
    "score_dist_plot",
    "score_bin_plot",
    # 风控策略
    "threshold_analysis_plot",
    "strategy_compare_plot",
    "vintage_plot",
    "feature_importance_plot",
    "approval_rate_trend_plot",
    "bad_rate_trend_plot",
    # 工具函数
    "setup_axis_style",
    "save_figure",
    "get_or_create_ax",
    "create_legend",
    "format_bin_label",
    "DEFAULT_COLORS",
]
