工具 ``hscredit.utils``
=======================

随机种子、pickle IO、数据描述、Pandas 扩展（``df.summary()`` / ``df.save()`` /
``df.show()``）、日志等通用工具。``import hscredit`` 时自动注册 Pandas 扩展方法。

DataFrame.show 的参数与用法
---------------------------

在 Notebook 中输入 ``table.show?``，或在 Python 中执行 ``help(table.show)``，
可以查看每个参数的默认值和传值示例。调用时用关键字指定参数。

.. code-block:: python

   import pandas as pd
   import hscredit

   table = pd.DataFrame({"样本总数": [80, 20], "坏样本率": [0.125, 0.25]})
   table.show(
       percent_cols=["坏样本率"],       # 按百分比显示
       precision={"坏样本率": 1},      # 0.125 显示为 12.5%
       condition_cols=["样本总数"],    # 数据条长度表示样本数
       color_cols=["坏样本率"],        # 背景色深浅表示坏样本率
   )

完整参数见 :meth:`hscredit.utils.BinTableDisplay.show`。

工具 API
--------

.. automodule:: hscredit.utils
   :members:
   :imported-members:
   :show-inheritance:
