"""报告模块.

提供专业的模型报告生成功能。

子模块:
- excel: Excel报告生成
- auto_report: 自动数据测试报告（三方数据评估）
"""

from .excel import ExcelWriter, dataframe2excel

try:
    from .auto_report import auto_feature_analysis_report
    __all__ = [
        "ExcelWriter",
        "dataframe2excel",
        "auto_feature_analysis_report",
    ]
except ImportError:
    __all__ = [
        "ExcelWriter",
        "dataframe2excel",
    ]
