"""EDA (Exploratory Data Analysis) 模块 - 金融信贷数据探索性分析.

提供金融风控场景下的完整数据探索功能，采用函数式API设计：
- 数据概览与质量评估
- 目标变量分析 (逾期率、分布)
- 特征分析 (数值/类别特征分布)
- 特征与标签关系 (IV、WOE、逾期率分箱)
- 特征稳定性分析 (PSI、CSI、时间稳定性)
- 相关性分析
- Vintage分析
- 综合报告生成

所有函数统一返回DataFrame格式，列名使用中文，便于理解和报告生成。

示例:
    >>> import hscredit.core.eda as eda
    >>> 
    >>> # 数据概览
    >>> summary = eda.data_info(df)
    >>> 
    >>> # 批量IV分析
    >>> iv_result = eda.batch_iv_analysis(df, features=['age', 'income'], target='fpd15')
    >>> 
    >>> # 逾期率趋势
    >>> trend = eda.bad_rate_trend(df, target_col='fpd15', date_col='apply_month')
    >>> 
    >>> # Vintage分析
    >>> vintage = eda.vintage_analysis(df, vintage_col='issue_month', mob_col='mob', target_col='ever_dpd30')
"""

# 数据概览
from .overview import (
    data_info,
    missing_analysis,
    feature_summary,
    numeric_summary,
    category_summary,
    data_quality_report,
    feature_group_analysis,
    population_stability_monitor,
)

# 目标变量分析
from .target import (
    target_distribution,
    bad_rate_overall,
    bad_rate_by_dimension,
    bad_rate_trend,
    bad_rate_by_bins,
    sample_distribution,
    # 辅助函数
    _build_overdue_labels,
    _create_binary_target,
)

# 特征分析
from .feature import (
    feature_type_inference,
    numeric_distribution,
    categorical_distribution,
    outlier_detection,
    rare_category_detection,
    concentration_analysis,
    feature_stability_over_time,
)

# 特征标签关系
from .relationship import (
    iv_analysis,
    batch_iv_analysis,
    woe_analysis,
    binning_bad_rate,
    monotonicity_check,
    univariate_auc,
    feature_importance_ranking,
)

# 相关性分析
from .correlation import (
    correlation_matrix,
    high_correlation_pairs,
    correlation_filter,
    vif_analysis,
)

# 稳定性分析
from .stability import (
    psi_analysis,
    batch_psi_analysis,
    csi_analysis,
    time_psi_tracking,
    stability_report,
    psi_cross_analysis,
    feature_drift_report,
    score_drift_report,
)

# 客群分析与偏移监控
from .population import (
    population_profile,
    population_shift_analysis,
    population_monitoring_report,
    segment_drift_analysis,
    feature_cross_segment_effectiveness,
)

# 策略分析
from .strategy import (
    approval_badrate_tradeoff,
    score_strategy_simulation,
    vintage_performance_summary,
    roll_rate_matrix,
    label_leakage_check,
    multi_label_correlation,
)

# Vintage分析
from .vintage import (
    vintage_analysis,
    vintage_summary,
    roll_rate_analysis,
)

# 综合报告
from .report import (
    eda_summary,
    generate_report,
    export_report_to_excel,
)

__all__ = [
    # 数据概览
    'data_info',
    'missing_analysis',
    'feature_summary',
    'numeric_summary',
    'category_summary',
    'data_quality_report',
    'feature_group_analysis',
    'population_stability_monitor',
    
    # 目标变量分析
    'target_distribution',
    'bad_rate_overall',
    'bad_rate_by_dimension',
    'bad_rate_trend',
    'bad_rate_by_bins',
    'sample_distribution',
    # 辅助函数
    '_build_overdue_labels',
    '_create_binary_target',
    
    # 特征分析
    'feature_type_inference',
    'numeric_distribution',
    'categorical_distribution',
    'outlier_detection',
    'rare_category_detection',
    'concentration_analysis',
    'feature_stability_over_time',
    
    # 特征标签关系
    'iv_analysis',
    'batch_iv_analysis',
    'woe_analysis',
    'binning_bad_rate',
    'monotonicity_check',
    'univariate_auc',
    'feature_importance_ranking',
    
    # 相关性分析
    'correlation_matrix',
    'high_correlation_pairs',
    'correlation_filter',
    'vif_analysis',
    
    # 稳定性分析
    'psi_analysis',
    'batch_psi_analysis',
    'csi_analysis',
    'time_psi_tracking',
    'stability_report',
    'psi_cross_analysis',
    'feature_drift_report',
    'score_drift_report',

    # 客群分析与偏移监控
    'population_profile',
    'population_shift_analysis',
    'population_monitoring_report',
    'segment_drift_analysis',
    'feature_cross_segment_effectiveness',

    # 策略分析
    'approval_badrate_tradeoff',
    'score_strategy_simulation',
    'vintage_performance_summary',
    'roll_rate_matrix',
    'label_leakage_check',
    'multi_label_correlation',

    # Vintage分析
    'vintage_analysis',
    'vintage_summary',
    'roll_rate_analysis',
    
    # 综合报告
    'eda_summary',
    'generate_report',
    'export_report_to_excel',
]
