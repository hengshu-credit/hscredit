Excel ``hscredit.excel``
========================

``ExcelWriter`` 上下文管理器与 ``dataframe2excel`` 等工具，支持写入数据表、插入图片/
超链接、条件格式、单元格样式、数字格式、冻结窗格、列宽调整与模板化输出。

大数据集保样式快速写入
----------------------

``dataframe2excel`` 可通过 ``fast=True`` 启用保样式快速写入。快速模式仍保留标题、
表头、内容填充、边框、合并、数字格式、条件格式、图片、筛选和自动列宽，不会降级为
无样式 Excel：

.. code-block:: python

   from hscredit.excel import dataframe2excel

   dataframe2excel(
       df,
       "特征明细.xlsx",
       sheet_name="特征明细",
       fast=True,
       auto_width=True,
   )

浮点值精度使用与 ``ScoreCard`` 一致的 ``decimal`` 参数。默认 ``decimal=4`` 保持历史
行为；传入 ``decimal=None`` 时不主动舍入 DataFrame 浮点值：

.. code-block:: python

   dataframe2excel(df, "原始精度.xlsx", fast=True, decimal=None)

对于包含多个工作表或多张表格的报告，应复用同一个 ``ExcelWriter``，全部写入完成后只
保存一次。大表的自动列宽会在数据写完后统一处理，不再逐单元格重复扫描。

.. automodule:: hscredit.excel
   :members:
   :imported-members:
   :show-inheritance:
