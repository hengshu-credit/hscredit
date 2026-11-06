数据探索 ``hscredit.core.eda``
==============================

57 种数据探索与策略分析能力：数据质量、目标分布、坏率趋势、客群画像与迁移、
Vintage、Roll Rate、策略仿真、稳定性（PSI/CSI）等。

.. rubric:: 特征摘要的缺失值口径

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
