"""报告模块.

提供专业的模型报告生成功能。

子模块:
- feature_analyzer: 特征分箱统计与自动分析
- rule_analysis: 规则集与多标签规则分析
- swap_analysis: 规则置换风险分析
- overdue_estimator: 逾期数据预估
"""
from ..excel import ExcelWriter, dataframe2excel
from .feature_analyzer import feature_bin_stats, auto_feature_analysis
from .rule_analysis import ruleset_analysis, multi_label_rule_analysis
from .swap_analysis import (
    ReferenceDataProvider,
    SwapAnalyzer,
    SwapAnalysisResult,
    SwapRiskConfig,
    create_swap_dataset,
    create_swap_dataset_from_rules,
    swap_analysis,
    SwapType,
)

from .overdue_predictor import OverduePredictor, overdue_prediction_report
from .model_report import QuickModelReport, auto_model_report, compare_models
from .mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    MultiLabelRuleMiner,
    TreeRuleExtractor,
    RuleMetrics,
    calculate_rule_metrics,
    TreeVisualizer,
    plot_decision_tree,
)

try:
    from .population_drift import population_drift
except ImportError:
    population_drift = None

__all__ = [
    "ExcelWriter",
    "dataframe2excel",
    "feature_bin_stats",
    "auto_feature_analysis",
    "ruleset_analysis",
    # swap分析
    "ReferenceDataProvider",
    "SwapAnalyzer",
    "SwapAnalysisResult",
    "SwapRiskConfig",
    "create_swap_dataset",
    "create_swap_dataset_from_rules",
    "swap_analysis",
    "SwapType",
    # 逾期预测
    "OverduePredictor",
    "overdue_prediction_report",
    # 规则挖掘（迁移自 core.rules.mining）
    "SingleFeatureRuleMiner",
    "MultiFeatureRuleMiner",
    "MultiLabelRuleMiner",
    "TreeRuleExtractor",
    "RuleMetrics",
    "calculate_rule_metrics",
    "TreeVisualizer",
    "plot_decision_tree",
    # 模型报告
    "QuickModelReport",
    "auto_model_report",
    "compare_models",
    "multi_label_rule_analysis",
    "population_drift",
]
