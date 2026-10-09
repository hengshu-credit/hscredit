损失函数与评估指标
==================

每一种损失都通过 ``loss.metric()`` 提供同定义评估指标，通过
``loss.business_metric()`` 提供对应业务指标，通过 ``metric.to_scorer()``
接入 sklearn 交叉验证。HSCredit 模型可直接接受 ``objective=loss`` 与
``eval_metric=metric`` 对象。

完整用法、全部损失对应表、框架训练、早停、CV、调参、金额对齐与迁移说明见
:doc:`../articles/losses-guide`。

.. code-block:: python

   from hscredit import LightGBM
   from hscredit.core.models.losses import FocalLoss

   loss = FocalLoss(alpha=0.75)
   model = LightGBM(objective=loss, n_estimators=100, n_jobs=1)
   model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   value = loss.metric().evaluate(y_valid, model.predict_proba(X_valid))

统一接口
--------

``loss.metric()`` 越小越好；真实业务指标的方向以 ``metric.direction`` 为准。
仅 sklearn scorer 对最小化指标自动取负。所有概率均为坏样本概率。

.. autoclass:: hscredit.core.models.BaseLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.BaseMetric
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: hscredit.core.models.losses.LossMetric
   :members:
   :show-inheritance:

自定义评估函数
--------------

只需要新评价口径时，优先用 ``make_metric`` 包装普通概率评估函数；
需要状态和额外方法时继承 ``BaseMetric``；训练目标才需要实现 ``BaseLoss``。

.. autofunction:: hscredit.core.models.losses.make_metric

.. autoclass:: hscredit.core.models.losses.CallableMetric
   :members:
   :show-inheritance:

分类与类别不平衡损失
--------------------

.. autoclass:: hscredit.core.models.FocalLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.AsymmetricFocalLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.BalancedFocalLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.WeightedBCELoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.CostSensitiveLoss
   :members:
   :show-inheritance:

审批、利润与金额损失
--------------------

.. autoclass:: hscredit.core.models.BadDebtLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ApprovalRateLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ProfitMaxLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.ExpectedProfitLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.AmountWeightedLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.ExpectedValueLoss
   :members:
   :show-inheritance:

排序与头部效果损失
------------------

.. autoclass:: hscredit.core.models.OrdinalRankLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.LiftFocusedLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.RankingAUCProxyLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.KSFocusedLoss
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.TopKBadCaptureLoss
   :members:
   :show-inheritance:

独立业务评估指标
----------------

这些指标可单独创建并评价任意二分类模型。``PSIMetric`` 用于参考分布与当前分布
的稳定性比较，其输入用途与其他监督指标不同。

.. autoclass:: hscredit.core.models.losses.AUCMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.KSMetric
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: hscredit.core.models.losses.GiniMetric
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: hscredit.core.models.losses.TopKCaptureMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.TopKLiftMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.BadDebtMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.ApprovalRateMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.ClassificationCostMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.ProfitMetric
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.losses.PSIMetric
   :members:
   :show-inheritance:
   :no-index:

框架适配器
----------

一般优先使用 ``loss.to_xgboost(api=...)``、``loss.to_lightgbm(api=...)``、
``loss.to_catboost()`` 或 ``loss.ngboost_params()``。
CatBoost、NGBoost 与 TabNet 的损失必须可安全逐样本或分批计算；具体限制见使用指南。

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

.. autoclass:: hscredit.core.models.losses.NGBoostLossAdapter
   :members:
   :show-inheritance:
