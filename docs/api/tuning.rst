超参数调优
==========

``hscredit.core.models.tuning`` 提供基于 Optuna 的超参数调优组件
（``pip install hscredit[tune]``）。可从顶层
``hscredit`` 懒加载导入 ``ModelTuner`` / ``AutoTuner`` / ``TuningObjective``。

快速上手与自定义目标
--------------------

完整教程见 :doc:`../articles/tuning-guide`，包含可执行 Notebook、普通函数指标、
``BaseMetric``、训练 ``BaseLoss``、金额权重、时间 CV、多目标及保存续跑。

已有模型优先使用 ``best = model.tune(X, y, metric=...)``；独立 ``ModelTuner.fit``
返回最佳参数字典，随后用 ``get_best_model()`` 获取已重训模型。
``loss`` 指定训练目标，``metric`` 指定搜索评价；普通函数可用
``make_metric(fn, name="业务误差", greater_is_better=False)`` 声明名称与方向。

训练后的完整搜索过程
--------------------

两种入口都保留原始 Optuna Study 和完整绘图能力::

   best = model.tune(X, y, n_trials=20)
   best.tuner.visualization.plot_timeline().show()

   tuner = ModelTuner(model, metric="auc", cv=3)
   tuner.fit(X, y, n_trials=20)
   tuner.visualization.plot_intermediate_values().show()
   tuner.visualization.matplotlib.plot_optimization_history()

   import optuna
   optuna.visualization.plot_rank(tuner.get_study()).show()

``visualization`` 动态提供当前 Optuna 版本的全部可视化函数；可使用
``dir(tuner.visualization)`` 查看，参数、绘图条件和返回值遵循原生接口。
原有 ``tuner.plot_*`` 便捷接口继续保留。
完整样例见 :doc:`../articles/model-workflow`。

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

.. autofunction:: hscredit.core.models.losses.make_metric
   :no-index:

.. autoclass:: hscredit.core.models.losses.CallableMetric
   :members:
   :show-inheritance:
   :no-index:
