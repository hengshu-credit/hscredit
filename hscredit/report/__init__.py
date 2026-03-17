"""报告模块.

提供专业的模型报告生成功能。

子模块:
- excel: Excel报告生成
- feature_report: 特征分析报告（三方数据评估）
- feature_analyzer: 特征分箱统计分析
- ruleset_report: 规则集综合评估报告
"""

from .excel import ExcelWriter, dataframe2excel
from .feature_analyzer import feature_bin_stats, FeatureAnalyzer
from .ruleset_report import ruleset_report

try:
    from .feature_report import auto_feature_analysis_report
    __all__ = [
        "ExcelWriter",
        "dataframe2excel",
        "auto_feature_analysis_report",
        "feature_bin_stats",
        "FeatureAnalyzer",
        "ruleset_report",
    ]
except ImportError:
    __all__ = [
        "ExcelWriter",
        "dataframe2excel",
        "feature_bin_stats",
        "FeatureAnalyzer",
        "ruleset_report",
    ]
