Excel ``hscredit.excel``
========================

``ExcelWriter`` 上下文管理器与 ``dataframe2excel`` 等工具，支持写入数据表、插入图片/
超链接、条件格式、单元格样式、数字格式、冻结窗格、列宽调整与模板化输出。

大数据集保样式快速写入
----------------------

``dataframe2excel`` 的 ``speed`` 默认是 ``"auto"``，会根据 DataFrame 的大小和维数
自动选择写入路径。满足以下任一条件时使用快速路径：数据行数不少于 500、有效列数
（包含写出的索引层级）不少于 50，或有效单元格数不少于 10000。其余情况使用普通路径。

无论选择哪条路径，都会保留标题、表头、内容填充、边框、合并、数字格式、条件格式、
图片、筛选和自动列宽，不会生成无样式 Excel。通常直接使用默认设置即可：

.. code-block:: python

   from hscredit.excel import dataframe2excel

   dataframe2excel(
       df,
       "特征明细.xlsx",
       sheet_name="特征明细",
       auto_width=True,
   )

需要固定写入路径时，可显式指定 ``speed="normal"`` 或 ``speed="fast"``，显式设置会
覆盖自动判断：

.. code-block:: python

   dataframe2excel(df, "兼容模式.xlsx", speed="normal")
   dataframe2excel(df, "快速模式.xlsx", speed="fast")

浮点值精度使用与 ``ScoreCard`` 一致的 ``decimal`` 参数。默认 ``decimal=4`` 保持历史
行为；传入 ``decimal=None`` 时不主动舍入 DataFrame 浮点值：

.. code-block:: python

   dataframe2excel(df, "原始精度.xlsx", speed="fast", decimal=None)

对于包含多个工作表或多张表格的报告，应复用同一个 ``ExcelWriter``，全部写入完成后只
保存一次。大表的自动列宽会在数据写完后统一处理，不再逐单元格重复扫描。

.. automodule:: hscredit.excel
   :members:
   :imported-members:
   :show-inheritance:
