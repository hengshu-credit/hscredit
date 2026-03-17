模型模块 API 参考
==================

自定义损失函数
--------------

.. automodule:: hscredit.model.losses
   :members:
   :undoc-members:
   :show-inheritance:

损失函数基类
~~~~~~~~~~~~

.. autoclass:: hscredit.model.losses.base.BaseLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

损失函数列表
~~~~~~~~~~~~

Focal Loss
^^^^^^^^^^

.. autoclass:: hscredit.model.losses.focal_loss.FocalLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: 示例

   .. code-block:: python

      from hscredit.core.models import FocalLoss
      import numpy as np

      # 创建损失函数
      loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

      # 计算
      y_true = np.array([0, 1, 1, 0, 1])
      y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
      loss = loss_fn(y_true, y_pred)

加权二元交叉熵
^^^^^^^^^^^^^^

.. autoclass:: hscredit.model.losses.weighted_loss.WeightedBCELoss
   :members:
   :undoc-members:
   :show-inheritance:

成本敏感损失
^^^^^^^^^^^^

.. autoclass:: hscredit.model.losses.risk_loss.CostSensitiveLoss
   :members:
   :undoc-members:
   :show-inheritance:

坏账损失
^^^^^^^^

.. autoclass:: hscredit.model.losses.risk_loss.BadDebtLoss
   :members:
   :undoc-members:
   :show-inheritance:

评估指标
--------

.. autoclass:: hscredit.model.losses.custom_metrics.KSMetric
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: hscredit.model.losses.custom_metrics.GiniMetric
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: hscredit.model.losses.custom_metrics.PSIMetric
   :members:
   :undoc-members:
   :show-inheritance:

框架适配器
----------

.. automodule:: hscredit.model.losses.adapters
   :members:
   :undoc-members:
   :show-inheritance:

XGBoost适配器
~~~~~~~~~~~~~

.. autoclass:: hscredit.model.losses.adapters.XGBoostLossAdapter
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: 示例

   .. code-block:: python

      from hscredit.core.models import FocalLoss, XGBoostLossAdapter
      import xgboost as xgb

      # 创建损失函数和适配器
      focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
      adapter = XGBoostLossAdapter(focal_loss)

      # 训练模型
      model = xgb.XGBClassifier(
          objective=adapter.objective,
          n_estimators=100
      )
      model.fit(X_train, y_train)

LightGBM适配器
~~~~~~~~~~~~~~

.. autoclass:: hscredit.model.losses.adapters.LightGBMLossAdapter
   :members:
   :undoc-members:
   :show-inheritance:

CatBoost适配器
~~~~~~~~~~~~~~

.. autoclass:: hscredit.model.losses.adapters.CatBoostLossAdapter
   :members:
   :undoc-members:
   :show-inheritance:
