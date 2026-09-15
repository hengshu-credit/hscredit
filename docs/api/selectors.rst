特征筛选 ``hscredit.core.selectors``
====================================

24 种特征筛选器，覆盖缺失率、众数率、方差、相关性、VIF、IV、KS、Lift、PSI、模型重要性、
逐步回归、Boruta、组合筛选等维度。``SelectionReportCollector`` 可手动聚合多个已拟合
筛选器的报告，生成统一中文摘要。

原始字段 KS 筛选
---------------

``KSSelector`` 支持直接传入包含目标列的原始数据，保留 ``KS >= threshold`` 的字段：

.. code-block:: python

   import pandas as pd
   from hscredit import KSSelector

   df = pd.read_excel("examples/hscredit_yyp.xlsx")
   columns = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别", "FPD"]
   selector = KSSelector(target="FPD", threshold=0.1)
   selected_df = selector.fit_transform(df[columns])
   scores = selector.scores_                       # 各字段 KS 值
   report = selector.get_selection_report()        # 统一中文报告
   dropped = selector.get_dropped_df()             # 剔除原因及 KS 值

数值字段使用原始取值计算双向 KS，重复值整体累计；非数值字段按训练样本的类别坏样本率
排序后计算，属于样本内区分度，高基数类别建议先合并或分箱。缺失值按字段排除；全缺失、
常量或有效样本仅含一种标签时记为 0。目标必须无缺失且同时包含 0 和 1。

也支持 ``fit(X, y)``；同时传入外部 ``y`` 与含目标列的数据框时，以外部 ``y`` 为准，
目标列不参与筛选。``transform`` 保留原始字段值，并在输入包含目标列时透传目标列。
``include``、``exclude``、``force_drop``、并行配置、Pipeline 和组合筛选沿用统一接口。
可通过 ``binner`` 或 ``binning_params`` 启用前置分箱，分箱 KS 统一使用
``compute_bin_stats`` 的最大分档 KS，缺失箱与特殊值箱遵循其排序约定。

.. automodule:: hscredit.core.selectors
   :members:
   :imported-members:
   :show-inheritance:
