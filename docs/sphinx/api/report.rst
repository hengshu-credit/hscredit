报告模块 API 参考
==================

Excel报告生成
-------------

.. automodule:: hscredit.report.excel
   :members:
   :undoc-members:
   :show-inheritance:

ExcelWriter 类
~~~~~~~~~~~~~~

.. autoclass:: hscredit.report.excel.ExcelWriter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: 示例

   .. code-block:: python

      from hscredit.report.excel import ExcelWriter
      import pandas as pd

      # 创建写入器
      writer = ExcelWriter(theme_color='3f1dba')

      # 获取工作表
      ws = writer.get_sheet_by_name("Sheet")

      # 写入数据
      df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
      writer.insert_df2sheet(ws, df, (1, 1), fill=True)

      # 保存
      writer.save("output.xlsx")

便捷函数
~~~~~~~~

.. autofunction:: hscredit.report.excel.dataframe2excel

   .. rubric:: 示例

   .. code-block:: python

      from hscredit.report.excel import dataframe2excel
      import pandas as pd

      df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
      dataframe2excel(df, "quick_export.xlsx", sheet_name="数据")
