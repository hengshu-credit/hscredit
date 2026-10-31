特征筛选 ``hscredit.core.selectors``
====================================

24 种特征筛选器，覆盖缺失率、众数率、方差、相关性、VIF、IV、KS、Lift、PSI、模型重要性、
逐步回归、Boruta、组合筛选等维度。``SelectionReportCollector`` 可手动聚合多个已拟合
筛选器的报告，生成统一中文摘要。

.. rubric:: 统一筛选报告接口

全部单筛选器、``CompositeFeatureSelector`` 和 ``ScorecardFeatureSelection``
均实现 ``summary, details = selector.report()``。报告能力由筛选器核心层提供，
只读取拟合时保存的真实决策，不重新筛选、不重算指标、不生成文件。

.. code-block:: python

   selector.fit(X, y)
   summary, details = selector.report()
   feature_decisions = details.loc[details["记录类型"] == "特征决策"]

   composite.fit(X, y)
   combined_summary, combined_details = composite.report()

两张表的列顺序和 dtype 对所有筛选器一致；汇总使用 ``SUMMARY_COLUMNS``，
明细使用 ``PROCESS_COLUMNS``（定义在 ``hscredit.core.selectors.reporting``）。
汇总包含阶段、父阶段、输入/选中/剔除特征数和保留率。明细用“记录类型”区分
特征决策、指标明细和迭代事件，保留算法特有的实际判定条件：

* 缺失、众数、方差、IV、KS 等保留实际指标、比较符和阈值；类型/正则条件不伪造数值阈值。
* 相关性、VIF 保留冲突字段、决策时指标及淘汰轮次。
* 卡方/F 检验、多条件稳定性筛选保留各项条件，不简化成一个得分阈值。
* 重要性、RFE、逐步选择和 Boruta 保留排名、候选评分、显著性或已记录的迭代事件。
* 强制保留/剔除、组合根决策和未执行步骤均保留原有语义；缺失或未计算的值留空。

组合报告包含根节点与子阶段，不能将其计数重复相加。
返回值是隔离副本，修改表和 attrs 不影响筛选器；修改构造参数也不会改写已有拟合快照。
默认未拟合时报错；``strict=False`` 可读取诊断，完整性与诊断存放在两表的 attrs。
原 ``get_selection_report()`` 字典接口仍可使用。

多个筛选器或已拟合 Pipeline 使用 ``hscredit.report.selection_report(...)`` 聚合，
与各筛选器的 ``report()`` 共用核心层的输出结构，参见 :doc:`report`。

.. rubric:: 原始字段 KS 筛选

``KSSelector`` 支持直接传入包含目标列的原始数据，保留 ``KS >= threshold`` 的字段：

.. code-block:: python

   import pandas as pd
   from hscredit import KSSelector

   df = pd.read_excel("examples/hscredit_yyp.xlsx")
   columns = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别", "FPD"]
   selector = KSSelector(target="FPD", threshold=0.1)
   selected_df = selector.fit_transform(df[columns])
   scores = selector.scores_                       # 各字段 KS 值
   summary, details = selector.report()            # 统一中文双表报告
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
