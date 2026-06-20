模型 ``hscredit.core.models``
=============================

涵盖经典模型、评分卡、规则模型、自定义风控损失函数与框架适配器、模型评估等。
Boosting 模型与调参器为懒加载，分别见 :doc:`boosting` 与 :doc:`tuning`。

经典模型与评分卡
----------------

.. autoclass:: hscredit.core.models.LogisticRegression
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ScoreCard
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.BaseRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.RandomForestRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ExtraTreesRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.GradientBoostingRiskModel
   :members:
   :show-inheritance:

规则模型
--------

.. autoclass:: hscredit.core.models.RuleSet
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.RulesClassifier
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.RuleResult
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.LogicOperator
   :members:
   :show-inheritance:

.. autofunction:: hscredit.core.models.create_and_ruleset

.. autofunction:: hscredit.core.models.create_or_ruleset

.. autofunction:: hscredit.core.models.combine_rules

损失函数与指标
--------------

自定义风控损失函数（继承 ``BaseLoss``）通过各框架适配器接入 XGBoost / LightGBM /
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

模型报告
--------

.. autoclass:: hscredit.core.models.ModelReport
   :members:
   :show-inheritance:
