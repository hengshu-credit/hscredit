"""报告模块.

提供专业的模型报告生成功能。

子模块:
- excel: Excel报告生成
- feature_report: 特征分析报告（三方数据评估）
- feature_analyzer: 特征分箱统计分析
- ruleset_report: 规则集综合评估报告
- swap_analysis_report: 规则置换风险分析报告
- overdue_estimator: 逾期数据预估
"""

from .excel import ExcelWriter, dataframe2excel
from .feature_analyzer import feature_bin_stats, FeatureAnalyzer
from .ruleset_report import ruleset_report
from .swap_analysis_report import (
    ReferenceDataProvider,
    SwapAnalyzer,
    SwapAnalysisResult,
    SwapRiskConfig,
    create_swap_dataset,
    create_swap_dataset_from_rules,
    swap_analysis_report,
    SwapType,
)

from .overdue_predictor import OverduePredictor, overdue_prediction_report
from .mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    TreeRuleExtractor,
    RuleMetrics,
    calculate_rule_metrics,
    TreeVisualizer,
    plot_decision_tree,
)

try:
    from .feature_report import auto_feature_analysis_report
except ImportError:
    auto_feature_analysis_report = None

__all__ = [
    "ExcelWriter",
    "dataframe2excel",
    "feature_bin_stats",
    "FeatureAnalyzer",
    "ruleset_report",
    # swap分析
    "ReferenceDataProvider",
    "SwapAnalyzer",
    "SwapAnalysisResult",
    "SwapRiskConfig",
    "create_swap_dataset",
    "create_swap_dataset_from_rules",
    "swap_analysis_report",
    "SwapType",
    # 逾期预测
    "OverduePredictor",
    "overdue_prediction_report",
    # 规则挖掘（迁移自 core.rules.mining）
    "SingleFeatureRuleMiner",
    "MultiFeatureRuleMiner",
    "TreeRuleExtractor",
    "RuleMetrics",
    "calculate_rule_metrics",
    "TreeVisualizer",
    "plot_decision_tree",
    "auto_feature_analysis_report",
]
