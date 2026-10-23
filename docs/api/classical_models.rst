经典模型
========

``hscredit.core.models`` 提供统一接口的经典风控模型。

统一链路接口
------------

搜索返回最终单模型，金融指标可在评估与报告复用。以下过程与完整报告接口也适用于
boosting、评分卡、规则和概率校准模型；不存在逐轮过程的算法返回空曲线表。
完整示例见 :doc:`../articles/model-workflow`。

.. automethod:: hscredit.core.models.BaseRiskModel.model_report

.. automethod:: hscredit.core.models.BaseRiskModel.get_training_history

.. automethod:: hscredit.core.models.BaseRiskModel.get_learning_curve

.. automethod:: hscredit.core.models.BaseRiskModel.get_eval_results

模型参数与方法
--------------

.. autoclass:: hscredit.core.models.LogisticRegression
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.BaseRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.RandomForest
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.ExtraTrees
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.GradientBoosting
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.SVM
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.DecisionTreeClassifier
   :members:
   :show-inheritance:
