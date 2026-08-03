分箱 ``hscredit.core.binning``
==============================

提供 17 种分箱算法与二维交互分箱，全部继承统一基类 ``BaseBinning``，并由工厂类
``OptimalBinning`` 作为统一入口。所有分箱指标统一通过 ``core.metrics`` 计算。

类别变量分箱
------------

类别变量会先转换为有序编码，再交给所选 ``method`` 的原生数值分箱算法决定边界。
因此等宽、等频、树、卡方、KS、IV、MDLP 等方法仍分别使用自己的切分和合并标准，
不会在 ``OptimalBinning`` 中被替换为同一种类别合并结果。

默认顺序按训练集坏样本率升序生成；坏样本率相同的类别保持首次出现顺序。也可以通过
``category_order`` 为指定字段提供完整顺序。下面使用 ``hscredit_hsk.xlsx`` 的 ``工资``
字段主动指定顺序：

.. code-block:: python

   import pandas as pd
   from hscredit.core.binning import OptimalBinning

   df = pd.read_excel("examples/hscredit_hsk.xlsx")
   X = df.drop(columns="target")
   y = df["target"]
   wage_order = X["工资"].dropna().drop_duplicates().tolist()

   binner = OptimalBinning(
       method="best_iv",
       max_n_bins=5,
       cat_cutoff=10,
       category_order={"工资": wage_order},
   ).fit(X, y)

   assert binner._category_orders_["工资"] == wage_order

``category_order`` 也可传入 ``callable(feature, x, y)``。函数应返回当前字段的完整非缺失
类别序列；缺少类别、重复类别或包含训练集以外的类别都会抛出中文 ``ValueError``。
数值编码的状态字段不会默认当作类别变量，需要通过 ``cat_cutoff`` 显式启用。例如
``cat_cutoff=10`` 表示唯一值数量不超过 10 的数值字段按类别处理。

自定义类别分箱
--------------

自定义类别分箱使用 ``List[List]``，必须完整覆盖训练类别，且类别和缺失标记不能重复。
缺失值既可以单独成箱，也可以与普通类别放在同一箱：

.. code-block:: python

   import numpy as np

   missing_alone = {
       "工资": [wage_order[:4], wage_order[4:8], wage_order[8:], [np.nan]],
   }
   missing_mixed = {
       "工资": [wage_order[:4], wage_order[4:8], [*wage_order[8:], np.nan]],
   }

   strict = OptimalBinning(
       user_splits=missing_alone,
       strict_user_splits=True,
   ).fit(X[["工资"]], y)

   mixed = OptimalBinning(
       user_splits=missing_mixed,
       strict_user_splits=True,
       missing_separate=False,
   ).fit(X[["工资"]], y)

``strict_user_splits=True`` 会完整保留用户分组；非严格模式把每个用户组视为不可拆分的
原子单位，再用当前 ``method`` 决定是否合并相邻组。未知预测期类别默认转换为索引 ``-3``、
标签 ``unknown`` 和中性 WOE ``0.0``；设置 ``handle_unknown="error"`` 可改为直接报错。

``max_n_bins``、``min_bin_size``、``max_bin_size``、``min_bad_rate`` 和明确的单调方向同样
适用于类别路径。如果单个类别或用户原子组本身已使约束不可满足，分箱器会指出字段、参数和
实际值，而不是静默忽略限制。

.. automodule:: hscredit.core.binning
   :members:
   :imported-members:
   :show-inheritance:
