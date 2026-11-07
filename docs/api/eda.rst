数据探索 ``hscredit.core.eda``
==============================

57 种数据探索与策略分析能力：数据质量、目标分布、坏率趋势、客群画像与迁移、
Vintage、Roll Rate、策略仿真、稳定性（PSI/CSI）等。

.. rubric:: 特征摘要的标签与缺失值口径

一个标签返回单层表头。多个标签共用统计详情、分布详情和 PSI，各标签独立计算 IV、KS、AUC 与趋势。
可直接提供 ``target``，或用 ``overdue``、``dpds`` 和 ``overdue_operator`` 生成标签。
原始标签缺失由 ``handle_missing_target`` 处理，``del_grey=True`` 逐标签剔除灰样本。
完整示例见 :ref:`feature-summary-labels`。

``DataFrame.summary``、``Series.summary`` 与 ``feature_summary`` 均支持
``fillna``。默认情况下，数值特征的 KS 和 AUC 使用特征与标签都有值的样本。
指定有限数值后，先用它填补特征，再与有效标签配对计算 KS 和 AUC。
AUC 沿用 ``hscredit.core.metrics.auc`` 的自动方向口径。
IV、趋势、PSI、缺失率和分布统计始终使用原始特征。

.. code-block:: python

   observed = X.summary(y=y)
   filled = X.summary(y=y, fillna=-999)

.. automodule:: hscredit.core.eda
   :members:
   :imported-members:
   :show-inheritance:
