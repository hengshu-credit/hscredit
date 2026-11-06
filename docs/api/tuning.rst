参数搜索 ``hscredit.core.models.tuning``
========================================================================

``hscredit.core.models.tuning`` 提供基于 Optuna 的超参数调优组件
（``pip install hscredit[tune]``）。可从顶层
``hscredit`` 懒加载导入 ``ModelTuner`` / ``AutoTuner`` / ``TuningObjective``。

.. rubric:: 快速上手与自定义目标

完整教程见 :doc:`../articles/tuning-guide`，包含可执行 Notebook、普通函数指标、
``BaseMetric``、训练 ``BaseLoss``、金额权重、时间 CV、多目标及保存续跑。

已有模型优先使用 ``best = model.tune(X, y, metric=...)``；独立 ``ModelTuner.fit``
返回最佳参数字典，随后用 ``get_best_model()`` 获取已重训模型。
``loss`` 指定训练目标，``metric`` 指定搜索评价；普通函数可用
``make_metric(fn, name="业务误差", greater_is_better=False)`` 声明名称与方向。

net 网络的自定义损失、优化器配置、条件搜索、回归指标及全量重训见 :doc:`net`。
回归自定义指标直接使用连续预测函数并指定 direction，分类概率 BaseMetric 不用于回归。

.. rubric:: 训练后的完整搜索过程

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
交互图使用 hscredit 的字体、配色和坐标轴样式。多目标搜索完成后，运行
``tuner.plot_pareto_front().show()`` 即可查看帕累托前沿。蓝色为普通试验，红色为前沿试验，
颜色深浅表示各组内的试验编号。悬停可查看该点的试验编号、目标值和超参数，
缩放和三维旋转沿用 Plotly 交互。使用 ``figure.write_html(...)`` 可导出交互图。
完整样例见 :doc:`../articles/model-workflow`。

.. autoclass:: hscredit.core.models.tuning.tuning.ModelTuner
   :members:
   :show-inheritance:
   :class-doc-from: both

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
