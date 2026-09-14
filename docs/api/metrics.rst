指标 ``hscredit.core.metrics``
==============================

43 种指标：分类指标（KS / AUC / Gini / 精确率 / 召回率 …）、稳定性指标（PSI / CSI）、
特征指标（IV / WOE / 分箱统计）与金融指标，建模、评估、策略与监控共用同一套口径。

``auc(y_true, score)``、``roc_curve(y_true, score)``、``ks_plot(score, y_true)``
和 ``roc_plot(y_true, score)`` 共用同一 ROC/AUC 计算入口，默认自动识别分数方向，
原始 AUC 小于 0.5 时反向，返回 0.5～1 的区分度。
固定方向的模型概率评估可传 ``score_direction='higher_risk'`` 保留原始 AUC；
高分代表安全的评分可传 ``score_direction='higher_safe'``。
``gini`` 接受相同参数，始终按 ``2 * AUC - 1`` 计算。
模型 ``evaluate``、内置 AUC 调参目标、逐步筛选和 Gini 回调也使用公共 AUC。
上述 ROC/AUC 入口接受 ``sample_weight``，同一权重同时用于方向判断、曲线和面积计算；
零权重样本不参与计算，缺失标签或分数仍按位置成对删除。
图中 AUC 统一显示四位小数，指标函数保留完整精度。

``ks`` 与 ``ks_plot``、``score_ks_plot`` 共用不同分数阈值处的累计曲线，
同分样本整体累计，KS 不受同分行顺序或分数方向影响。
``auc``、``gini``、``ks``、``roc_curve`` 与两种 ROC 图支持 ``pos_label`` 指定正样本，
并按位置配对删除标签或分数缺失的样本。
分箱统计表中的 KS 只在分箱边界计算，可能低于原始分数的 KS。

.. automodule:: hscredit.core.metrics
   :members:
   :imported-members:
   :show-inheritance:
