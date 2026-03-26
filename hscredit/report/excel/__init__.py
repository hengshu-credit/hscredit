"""Excel报告生成模块.

提供专业的Excel报告生成功能，支持丰富的样式和格式化选项。

核心功能:
- ExcelWriter: Excel写入器，支持DataFrame、图片、超链接等
- dataframe2excel: 快速将DataFrame写入Excel的便捷函数
- DataFrame.save(): pandas DataFrame的save方法扩展（在utils.pandas_extensions中统一注册）
- Series.save(): pandas Series的save方法扩展（在utils.pandas_extensions中统一注册）

使用示例:
    >>> import pandas as pd
    >>> from hscredit.report.excel import ExcelWriter
    >>> 
    >>> # DataFrame直接保存
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df.save("report.xlsx", sheet_name="数据", title="统计表")
    >>> 
    >>> # Series直接保存（自动转为DataFrame）
    >>> s = pd.Series([1, 2, 3], name='数值')
    >>> s.save("series_report.xlsx", title="序列数据")
    >>> 
    >>> # 写入已有的ExcelWriter
    >>> writer = ExcelWriter()
    >>> worksheet = writer.get_sheet_by_name("Sheet1")
    >>> df.save(writer, worksheet=worksheet)
    >>> writer.save("report.xlsx")
"""

from .writer import ExcelWriter, dataframe2excel

# 注意：pandas扩展方法（df.save(), df.show(), df.summary()等）
# 现在在 utils.pandas_extensions 模块中统一注册
# 导入hscredit时会自动注册所有扩展方法

__all__ = [
    "ExcelWriter",
    "dataframe2excel",
]
