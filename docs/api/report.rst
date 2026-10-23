报告 ``hscredit.report``
========================

报告生成与规则挖掘：特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比，
以及格式化 Excel 输出工具。

.. automodule:: hscredit.report
   :members:
   :imported-members:
   :show-inheritance:
   :exclude-members: ModelReport, ExcelWriter, dataframe2excel, SingleFeatureRuleMiner, MultiFeatureRuleMiner, MultiLabelRuleMiner, TreeRuleExtractor, RuleMetrics, calculate_rule_metrics

.. rubric:: 模型报告的配置、评估与导出

.. autoclass:: hscredit.report.ModelReport
   :members:
   :show-inheritance:
   :class-doc-from: both

.. automethod:: hscredit.report.ModelReport.get_training_process

.. note::

   ``ExcelWriter`` / ``dataframe2excel`` 见 :doc:`excel`；规则挖掘器在下方
   ``hscredit.report.mining`` 小节统一说明。

.. rubric:: 规则挖掘 ``hscredit.report.mining``

单特征 / 多特征交叉 / 多标签规则挖掘器与树规则提取器。

.. automodule:: hscredit.report.mining
   :members:
   :imported-members:
   :show-inheritance:
