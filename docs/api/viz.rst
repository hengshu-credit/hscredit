可视化 ``hscredit.core.viz``
============================

46+ 种风控可视化图表：分箱趋势、KS / ROC / PR / Lift / Gain、评分分布、策略阈值、
Vintage、变量稳定性、客群漂移、决策树图等。

推荐入口
--------

新代码优先使用以下四个公开方法；其他同类入口主要用于兼容既有代码或特定报告布局。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 方法
     - 推荐场景
   * - ``bin_plot``
     - 原始特征或分箱统计表的样本结构、坏率和分箱指标
   * - ``ks_plot``
     - KS / ROC 评估；用 ``pos_label`` 明确正类，用 ``score_direction`` 明确评分方向
   * - ``lift_plot``
     - 按预测坏概率排序的 Lift 分箱效果
   * - ``plot_model_feature_importance``
     - 模型原生重要性、SHAP 分布和单特征依赖关系

样式与调用契约
--------------

``import hscredit`` 会自动调用 ``hscredit.init_setting()``，统一配置内置中文字体、
Matplotlib 基础样式和负号显示。``set_style`` / ``reset_style`` 仅用于在该基线上
临时覆盖主题，不是独立的初始化入口。

推荐方法独立创建画布时返回 Matplotlib ``Figure``。``bin_plot`` 和 ``lift_plot``
通过 ``ax`` 嵌入已有画布；``ks_plot(curve='ks'/'roc')`` 通过 ``ax`` 嵌入单图，
默认双图模式则传入 ``axes=[ks_ax, roc_ax]``。嵌入时 ``bin_plot`` / ``ks_plot``
返回所用 ``Axes``，``lift_plot`` 返回该轴所属 ``Figure``。
``bin_plot(..., ax=ax, return_frame=True)`` 返回 ``(ax, 分箱统计表)``。
``save`` 路径的文件格式由后缀推断，例如 ``.png``、``.svg`` 或 ``.pdf``。

.. automodule:: hscredit.core.viz
   :members:
   :imported-members:
   :show-inheritance:
