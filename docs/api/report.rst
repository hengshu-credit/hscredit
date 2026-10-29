报告 ``hscredit.report``
========================

报告生成与规则挖掘：特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比，
以及格式化 Excel 输出工具。

.. automodule:: hscredit.report
   :members:
   :imported-members:
   :show-inheritance:
   :exclude-members: ModelReport, ExcelWriter, dataframe2excel, SingleFeatureRuleMiner, MultiFeatureRuleMiner, MultiLabelRuleMiner, TreeRuleExtractor, MultiwayTreeExtractor, RuleMetrics, calculate_rule_metrics

.. rubric:: 模型报告的配置、评估与导出

分类、回归及所有模型的完整链路验收见 :doc:`model_workflow_validation`。
``prediction_method=None`` 按任务自动选择概率或连续预测。报告的
``accuracy/precision/recall/f1`` 与模型的决策语义一致，规则模型保留 AND/OR 逻辑。

训练过程与 SHAP 重要性
----------------------------------------

训练过程页展示训练概览、预处理/筛选、逐试验参数和指标、最终训练记录及学习曲线。
不再递归展开 Python 对象、默认参数和依赖版本。完整模型参数在参数页；原始过程表仍可通过
``get_training_process()`` 获取，逐轮数值也仍完整保留。

``training_sheet_index=None`` 按实际启用章节生成编号：没有完整模型解释页时为
``7-训练与调参过程``，有 ``7-模型解释`` 时为 ``8-训练与调参过程``。
``to_excel`` 和 ``auto_model_report`` 均可指定 ``training_sheet_index=10`` 等未占用编号，
目录链接同步更新；编号冲突会明确报错。
正文标题固定从 B 列开始，与其他页面一致：主标题使用“七、训练与调参过程”，
小节从“1、训练概览”独立编号；最终合并宽度覆盖该节表格和图表。
参数页逐项保留嵌套模型、规则、列表和配置字典，并区分公开配置、实际训练参数以及
与构造配置不同的已拟合组件。未设置的参数显示为 ``None``，不显示成缺失的空白单元格。
配置路径使用原始参数和属性名，例如 ``model.C``、``model_.C``、``native_params_.max_depth``，
不翻译参数名，也不在路径中插入中文分组名。
模型报告不生成“报告执行状态”工作表或目录项；执行结果仍由 ``return_result=True`` 返回，
严格模式仍拒绝发布失败报告。
基本信息页的样本分布只合并“数据集”列；“数据分组”逐行保留月份或类别，
同名月份和类别不会跨数据集合并单元格。

重要性使用 SHAP 的平均绝对贡献，另保留平均 SHAP 值的正负方向。
树模型同时显示原生重要性与 SHAP；net 不编造 gain。两种数值的尺度不混合相加。
配置解释数据和背景后显式计算，避免普通报告在调用方不知情时进行大量额外预测：

.. code-block:: python

   report = model.model_report(
       datasets={"训练集": train, "选参验证集": valid},
       explain_config={
           "enabled": True,
           "scope": "importance",  # 仅全局重要性，不生成独立的完整解释页
           "dataset": "选参验证集",
           "max_samples": 100,
           "max_background": 32,
           "random_state": 42,
       },
       verbose=False,
   )
   importance = report.get_shap_importance()
   report.to_excel("模型报告.xlsx", training_sheet_index=None)

背景默认来自训练数据，解释数据由 dataset 或 data 指定，两者不能同时提供。
``scope='full'`` 保留完整分类解释页；回归使用 ``scope='importance'``，解释连续预测值。
分类解释坏样本概率。输入是 Pipeline 时解释完整预测链路的原始字段。
树后端可用时使用 Tree SHAP；通用模型、Pipeline 或不支持该输出尺度的类别输入使用排列式 SHAP，
预测前还原原始字段类型，并核对基准值与各特征贡献的和等于模型输出。
排列式 SHAP 估计 Shapley 贡献，不是打乱特征后观察 AUC 下降的性能置换重要性。
报告记录解释样本量、背景样本量、算法和输出尺度；重要性只描述当前模型在所选样本上的依赖。
TabPFN 的混合精度推理可能产生批量相关的数值差异；加总校验失败时不会用放宽断言掩盖。
可在独立模型中显式设置 inference_precision=torch.float32 后重新建立推理上下文，
并使用同一精度生成预测和解释。此配置可能改变预测值，不能把结果标作原混合精度模型的解释。

定义见 `SHAP PermutationExplainer <https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html>`_
和 `SHAP 全局条形图 <https://shap.readthedocs.io/en/latest/generated/shap.plots.bar.html>`_。

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
