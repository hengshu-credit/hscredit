"""
Pandas扩展 - 为DataFrame和Series添加save方法

注意：此功能已迁移到 utils.pandas_extensions 模块统一实现。
保留此文件是为了向后兼容。

使用方式：
    import pandas as pd
    import hscredit  # 自动安装save方法
    
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.save("report.xlsx", sheet_name="Sheet1", title="数据表")
    
    # Series也会自动转为DataFrame保存
    s = pd.Series([1, 2, 3], name='数值')
    s.save("series_report.xlsx", title="序列数据")
"""

# 从utils.pandas_extensions导入，确保向后兼容
from ...utils.pandas_extensions import (
    _dataframe_save,
    _series_save,
    register_extensions,
)

# 自动安装（当导入此模块时）- 已由utils.pandas_extensions处理
# 这里调用是为了保持向后兼容
register_extensions()
