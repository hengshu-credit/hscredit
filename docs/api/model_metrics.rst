评估指标 ``hscredit.core.models.losses.metrics``
================================================

指标实现位于 ``hscredit/core/models/losses/metrics.py``，基类位于同包的
``base_metric.py``，工厂与注册函数位于 ``registry.py``。这些对象由模型包重导出。

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
