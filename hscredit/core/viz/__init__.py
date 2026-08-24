# -*- coding: utf-8 -*-
"""
可视化模块 (viz).

推荐的公开绘图入口：
- 特征分箱图 (bin_plot)
- KS/ROC 曲线图 (ks_plot)
- 特征分布图 (hist_plot)
- 特征相关性热力图 (corr_plot)
- PSI稳定性分析图 (psi_plot)
- Lift 提升图 (lift_plot)
- DataFrame表格图 (dataframe_plot)
- 时间分布图 (distribution_plot)
- 模型特征重要性图 (plot_model_feature_importance)
- 逻辑回归系数误差图 (plot_weights)

``score_*``、``feature_importance_plot`` 等同类函数为兼容或特定报告场景入口；
新代码优先使用以上推荐方法。全局图片样式由 :func:`hscredit.init_setting`
初始化，``set_style`` / ``reset_style`` 仅作为可选主题覆盖层。

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
- 坏样本率趋势图 (bad_rate_trend_plot)

辅助函数已移至 utils 模块：
- init_setting -> hscredit.utils.init_setting
- feature_describe -> hscredit.utils.feature_describe
- round_float -> hscredit.utils.round_float
- feature_bins -> hscredit.utils.feature_bins

参考 scorecardpipeline 实现优化而来。
"""

from .binning_plots import (
    bin_plot,
    bin_2d_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    psi_plot,
    dataframe_plot,
    distribution_plot,
    bin_trend_plot,
    batch_bin_trend_plot,
    bin_overdues_plot,
)

from .model_plots import plot_model_feature_importance, plot_model_sample_shap, plot_weights

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

# 新增：变量分析图表
from .variable_plots import (
    metric_comparison_plot,
    variable_iv_plot,
    variable_woe_trend_plot,
    variable_psi_heatmap,
    variable_importance_grouped_plot,
    variable_missing_badrate_plot,
)

# 新增：评分分析图表
from .score_plots import (
    score_ks_plot,
    score_distribution_comparison_plot,
    score_badrate_bin_plot,
    score_lift_plot,
    score_approval_badrate_curve,
)

# 新增：策略分析图表
from .strategy_plots import (
    rule_swap_plot,
    strategy_simulation_plot,
    feature_trend_by_time,
    feature_drift_comparison,
    feature_effectiveness_by_segment,
    feature_cross_heatmap,
    population_drift_monitor,
    segment_scorecard_comparison,
)

# 新增：决策树可视化（AntV G6 风格，支持 matplotlib / pyecharts / graphviz）
from .tree_plots import (
    DecisionTreeViz,
    plot_tree,
    plot_tree_matplotlib,
    plot_tree_pyecharts,
    plot_tree_graphviz,
    tree_leaf_comparison_plot,
)

# 统一样式系统
from .style import (
    set_style,
    reset_style,
    get_current_theme,
    get_palette,
    get_font_sizes,
    get_defaults,
    PRIMARY_COLORS,
    EXTENDED_COLORS,
    SEMANTIC_COLORS,
    GRADIENT_PALETTES,
)

# 导出工具函数供外部使用
from .utils import (
    setup_axis_style,
    save_figure,
    get_or_create_ax,
    create_legend,
    format_bin_label,
    DEFAULT_COLORS,
    BAD_RATE_COLOR,
    REFERENCE_COLOR,
    STABLE_COLOR,
    CHANGING_COLOR,
    UNSTABLE_COLOR,
    NEUTRAL_COLOR,
    get_series_colors,
    get_psi_color,
    make_colormap,
    make_risk_cmap,
    make_diverging_cmap,
)

__all__ = [
    # 特征分箱相关
    "bin_plot",
    "bin_2d_plot",
    "corr_plot",
    "ks_plot",
    "hist_plot",
    "psi_plot",
    "dataframe_plot",
    "distribution_plot",
    # 特征趋势分析
    "bin_trend_plot",
    "batch_bin_trend_plot",
    "bin_overdues_plot",
    # 模型相关
    "plot_weights",
    "plot_model_feature_importance",
    "plot_model_sample_shap",
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
    # 新增：变量分析图表
    "metric_comparison_plot",
    "variable_iv_plot",
    "variable_woe_trend_plot",
    "variable_psi_heatmap",
    "variable_importance_grouped_plot",
    "variable_missing_badrate_plot",
    # 新增：评分分析图表
    "score_ks_plot",
    "score_distribution_comparison_plot",
    "score_badrate_bin_plot",
    "score_lift_plot",
    "score_approval_badrate_curve",
    # 新增：策略分析图表
    "rule_swap_plot",
    "strategy_simulation_plot",
    "feature_trend_by_time",
    "feature_drift_comparison",
    "feature_effectiveness_by_segment",
    "feature_cross_heatmap",
    "population_drift_monitor",
    "segment_scorecard_comparison",
    # 决策树可视化
    "DecisionTreeViz",
    "plot_tree",
    "plot_tree_matplotlib",
    "plot_tree_pyecharts",
    "plot_tree_graphviz",
    "tree_leaf_comparison_plot",
    # 工具函数
    "setup_axis_style",
    "save_figure",
    "get_or_create_ax",
    "create_legend",
    "format_bin_label",
    "DEFAULT_COLORS",
    "BAD_RATE_COLOR",
    "REFERENCE_COLOR",
    "STABLE_COLOR",
    "CHANGING_COLOR",
    "UNSTABLE_COLOR",
    "NEUTRAL_COLOR",
    "get_series_colors",
    "get_psi_color",
    "make_colormap",
    "make_risk_cmap",
    "make_diverging_cmap",
    # 统一样式系统
    "set_style",
    "reset_style",
    "get_current_theme",
    "get_palette",
    "get_font_sizes",
    "get_defaults",
    "PRIMARY_COLORS",
    "EXTENDED_COLORS",
    "SEMANTIC_COLORS",
    "GRADIENT_PALETTES",
]
