损失函数
========

自定义风控损失函数（继承 ``BaseLoss``）通过框架适配器接入 XGBoost / LightGBM /
CatBoost / TabNet。

.. autoclass:: hscredit.core.models.BaseLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.FocalLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.AsymmetricFocalLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.WeightedBCELoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.CostSensitiveLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.BadDebtLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ApprovalRateLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ProfitMaxLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.OrdinalRankLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.LiftFocusedLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.XGBoostLossAdapter
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.LightGBMLossAdapter
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.CatBoostLossAdapter
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.TabNetLossAdapter
   :members:
   :show-inheritance:
