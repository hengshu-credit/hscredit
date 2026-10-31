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
``prediction_method=None`` 默认使用分类模型的 ``predict_proba``，纯评分转换器的
``transform``，回归模型的 ``predict``。ScoreCard 等具有 ``predict_proba`` 的分类模型
仍默认输出概率。显式 ``prediction_method`` 优先，可指定模型已实现的方法名或 callable；
``prediction_kwargs`` 同时支持两种形式，数据集只缓存该方法的返回结果。
纯评分转换器以概率列作为输入，例如 ``features=["原始概率"]``，不把业务目标列传入转换器。

逾期标签在所有模型报告表中统一展示为 ``字段@阈值``，例如 ``MOB1@3``；比较符不会混入标签名。
显式 target 中形如 ``MOB1大于3``、``MOB1>3`` 的名称也按此规则展示，但名称不会用于推断其计算逻辑。
归一化后重名的标签会明确报错。基本信息的“标签定义与剔灰规则”表按数据集/标签说明
来源、操作符、坏样本定义、剔灰开关、灰区、缺失值处理及实际有效/剔除样本数。
``>`` 的灰区为 ``(0, dpd]``，``>=`` 为 ``(0, dpd)``；``<`` 和 ``<=`` 不设置灰区。
显式 target 开启剔灰时剔除非0/1及缺失值，否则非1值按0；外部 y 优先，并明确标出实际采用的口径。

采用概率转评分输出时，模型参数页单独展示转换器、实际参数及转换公式。
支持标准、线性、分位数和 Box-Cox；分位数的全部分位点单独列表，Box-Cox 展示已拟合 lambda 和值域。
``transform`` 直接返回转换值；显式使用 ``predict`` 时还执行转换器的截断及四舍五入，报告说明与实际方法一致。
普通分类器默认概率报告不会因为模型持有评分转换器就自动改为评分。

.. code-block:: python

   # 默认使用模型的坏样本概率，按每个标签独立剔灰。
   report = model.model_report(
       datasets={"训练集": train, "验证集": valid},
       overdue=["MOB1"], dpds=[7, 3, 0], overdue_operator=">", del_grey=True,
   )
   # 直接报告已拟合的评分转换器，默认调用 transform；显式参数优先。
   score_report = ModelReport(
       converter, datasets={"验证集": probability_frame},
       features=["原始概率"], target="FPD", prediction_method="transform",
   )

多标签报告默认单图使用逾期标签的“任一为1”联合视图，显式 target 字典使用首标签。
基本信息会说明该口径；联合视图只剩一个类别时，KS/LIFT 图按不适用处理，保留其他统计与图表。

报告的
``accuracy/precision/recall/f1`` 与模型的决策语义一致，规则模型保留 AND/OR 逻辑。

.. rubric:: 训练过程与 SHAP 重要性

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
排序性与变量分析的图组在表格列宽确定后统一定位，避免首组图与后续图组错位。
学习曲线、SHAP 重要性和回归图复用 hscredit 坐标轴样式、主题色、字号和 240 DPI 保存规范。
模型报告的 SHAP 与重要性图统一使用 ``plot_model_feature_importance``，左侧展示
重要性柱条与 SHAP 蜂群分布，最多 TOP32；右侧展示 TOP10 依赖图。
依赖图有对应标签时按标签着色，显式无标签解释数据不伪造标签颜色。
图直接复用报告已经校验的 SHAP 样本、背景和贡献值，不重复解释模型。
有原生重要性时按原生值排序，否则使用平均绝对 SHAP 值；回归不按每个连续目标值分组。
完整变量明细表不截断。启用 SHAP 后不再另绘独立的特征贡献图；未启用时仍使用统一入口展示
TOP32 原生重要性，不隐式增加 SHAP 计算。``feature_contribution_label_max_features``
保留调用兼容性，不控制新的整合图。

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

.. rubric:: 特征筛选报告

``summary, process = selection_report(selector=...)`` 是模型报告中的数据生成环节，
固定返回筛选汇总和具体过程两个 DataFrame，不接受 Excel 输出参数，也不写入文件。
双表整理与固定表头统一在 ``hscredit.core.selectors.base`` 实现；单个筛选器直接使用 ``selector.report()``，
组合筛选器同样使用 ``composite.report()``。本模块导出的 ``selection_report`` 就是 base 中的同一函数，
负责列表、收集器、Pipeline 等来源的收集与汇总。
输入为用户已配置并拟合的单筛选器、组合筛选器、Pipeline 或这些对象组成的列表。
列表元素可用 ``(名称, 筛选器)`` 命名，也可直接传入已保存的 ``SelectionReport`` 快照。

.. code-block:: python

   from hscredit.report import selection_report

   summary, process = fitted_selector.report()
   summary, process = fitted_composite.report()
   summary, process = selection_report(selector=fitted_pipeline)

   # 实际串行拟合过的多个单筛选器，字段必须前后衔接。
   summary, process = selection_report(
       [("缺失", fitted_null_selector), ("相关性", fitted_corr_selector)],
       relation="sequential",
   )

   # 独立的单筛选器、组合筛选器和 Pipeline，分别展示各自的完整筛选过程。
   summary, process = selection_report([
       ("单项", fitted_selector),
       ("组合", fitted_composite),
       ("流水线", fitted_pipeline),
   ])
   model_sections = {"筛选汇总": summary, "筛选过程": process}
   feature_decisions = process.loc[process["记录类型"] == "特征决策"]

汇总表按阶段保留路径、父路径、输入/选中/剔除特征数和保留率。流程根节点与
子步骤分别呈现，不能将它们的计数再次相加。列表默认独立比较，不编造最终特征数；
Pipeline 按实际步骤串行收集，列表串行或交并集关系通过 ``relation`` 显式指定。

过程表按汇总中的阶段顺序排列，用“记录类型”区分特征决策、指标明细和迭代事件，
保留原始字段、指标精度、阈值、原因和轮次；同一类型内保留原始记录顺序。
采用逐行合并，避免多指标与多轮历史相互连接造成重复计数。不重新拟合或计算指标，
也不补造拟合时未保存的历史。两张表均为隔离副本，空表保留固定结构。

默认 ``strict=True`` 拒绝不完整来源。诊断时可传 ``strict=False``，完整性及诊断保留在
``summary.attrs`` 和 ``process.attrs``。原 ``auto_selection_report``、``source``、
``excel_writer``、``return_result``、``writer_params`` 等参数不再属于该接口。

.. automethod:: hscredit.report.ModelReport.get_training_process

.. note::

   ``ExcelWriter`` / ``dataframe2excel`` 见 :doc:`excel`；规则挖掘器见子包 :doc:`mining`。

.. toctree::
   :maxdepth: 2

   mining

.. raw:: html

   <script>
   if (window.location.hash.startsWith('#hscredit.report.mining.')) {
       window.location.replace('mining.html' + window.location.hash);
   }
   </script>
