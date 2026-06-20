超参数调优 ``hscredit.core.models.tuning``
==========================================

基于 Optuna 的超参数调优组件（``pip install hscredit[tune]``）。可从顶层
``hscredit`` 懒加载导入 ``ModelTuner`` / ``AutoTuner`` / ``TuningObjective``。

.. autoclass:: hscredit.core.models.tuning.tuning.ModelTuner
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.tuning.tuning.AutoTuner
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.tuning.tuning.TuningObjective
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.tuning.tuning.Metric
   :members:
   :show-inheritance:
