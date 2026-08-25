分箱 ``hscredit.core.binning``
==============================

提供 17 种分箱算法与二维交互分箱，全部继承统一基类 ``BaseBinning``，并由工厂类
``OptimalBinning`` 作为统一入口。所有分箱指标统一通过 ``core.metrics`` 计算。

数值变量的箱数与样本约束
--------------------------

``max_n_bins`` 表示普通箱（不含 missing/special）的最大目标箱数，``min_n_bins`` 表示
数据和其他约束可满足时的最小目标箱数。唯一值不足、等频边界重复或其他硬约束冲突时，
实际普通箱数可以少于目标值。

``min_bin_size`` 默认是 ``0.01``。取值 ``0 < value < 1`` 时按有效训练样本占比解释，
取值 ``value >= 1`` 时按绝对样本数解释；显式传入 ``None`` 表示关闭最小样本数限制。
缺失值和特殊值不参与普通箱最小样本数的计算。

``uniform`` 会先按 ``max_n_bins`` 生成完整等距切点，再由 ``OptimalBinning`` 应用
``min_bin_size`` 等通用约束并合并相邻箱，因此约束收口后的区间允许宽度不等。
当 ``min_bin_size=None`` 时保留原始等距切点，也允许规则空间中存在训练样本数为 0 的箱。

``quantile`` 按分位数生成等频边界；多个分位点落在相同重复值上时会像
``qcut(duplicates='drop')`` 一样合并重复边界，所以实际箱数会正常减少且不会人为拆分相同值。
自定义 ``quantiles`` 直接决定候选分位点，不再受 ``min_n_bins``、``max_n_bins`` 或
``min_bin_size`` 二次裁剪。

``max_bin_size``、``min_bad_rate`` 和 ``monotonic`` 是否参与收口取决于具体算法；
固定用户切点/类别组优先保持用户规则，约束不可满足时会抛出包含字段和参数的中文错误。

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
       user_splits_fixed=True,
   ).fit(X[["工资"]], y)

   mixed = OptimalBinning(
       user_splits=missing_mixed,
       user_splits_fixed=True,
       missing_separate=False,
   ).fit(X[["工资"]], y)

``user_splits_fixed=True`` 会完整保留用户分组；非严格模式把每个用户组视为不可拆分的
原子单位，再用当前 ``method`` 决定是否合并相邻组。未知预测期类别默认转换为索引 ``-3``、
标签 ``unknown`` 和中性 WOE ``0.0``。``handle_unknown='value'`` 与默认 ``-3`` 等价；
``handle_unknown='raise'`` 会在 transform 遇到训练期未知类别时直接报错；也可指定任意已记录的整数箱号。

``max_n_bins``、``min_bin_size``、``max_bin_size``、``min_bad_rate`` 和明确的单调方向同样
适用于类别路径。如果单个类别或用户原子组本身已使约束不可满足，分箱器会指出字段、参数和
实际值，而不是静默忽略限制。

.. automodule:: hscredit.core.binning
   :members:
   :imported-members:
   :show-inheritance:
