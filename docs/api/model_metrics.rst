评估指标
========

``hscredit.core.models`` 提供面向风控任务的模型评估指标。

推荐从 ``hscredit`` 统一导入 ``get_metric`` / ``list_metrics``。
同一指标对象用于模型 ``eval_metric``、调参 ``metric`` 和离线 ``evaluate``。
损失派生指标使用 ``loss.metric(kind='loss'/'business')`` 或 ``get_metric(loss, kind=...)``。

.. autofunction:: hscredit.core.models.get_metric

.. autofunction:: hscredit.core.models.list_metrics

.. autoclass:: hscredit.core.models.BaseMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.KSMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.GiniMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.PSIMetric
   :members:
   :show-inheritance:
