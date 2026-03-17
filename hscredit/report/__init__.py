"""报告模块.

提供专业的模型报告生成功能。

子模块:
- excel: Excel报告生成
- feature_report: 特征分析报告（三方数据评估）
- feature_analyzer: 特征分箱统计分析
- ruleset_report: 规则集综合评估报告
- swapin_prediction: 规则置换风险预估
"""

from .excel import ExcelWriter, dataframe2excel
from .feature_analyzer import feature_bin_stats, FeatureAnalyzer
from .ruleset_report import ruleset_report
from .swapin_prediction import (
    swapin_risk_prediction,
    SwapinRiskAnalyzer,
    swapin_summary_report,
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
    "swapin_risk_prediction",
    "SwapinRiskAnalyzer",
    "swapin_summary_report",
    "auto_feature_analysis_report",
]
