"""Excel报告生成模块.

提供专业的Excel报告生成功能，支持丰富的样式和格式化选项。

核心功能:
- ExcelWriter: Excel写入器，支持DataFrame、图片、超链接等
- dataframe2excel: 快速将DataFrame写入Excel的便捷函数
"""

from .writer import ExcelWriter, dataframe2excel

__all__ = [
    "ExcelWriter",
    "dataframe2excel",
]
