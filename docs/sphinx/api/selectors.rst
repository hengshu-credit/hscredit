特征筛选模块 API 参考
=====================

本模块提供多种特征筛选方法，从多个维度评估和筛选特征。

核心特性
--------

- 过滤法: 方差、相关性、VIF、缺失率、单一值率、基数、IV、Lift、PSI
- 包装法: 穷举搜索、逐步回归、Boruta
- 嵌入法: 特征重要性、Lasso、树模型重要性、Permutation Importance

所有筛选器支持：

- 独立使用: fit/transform接口
- Pipeline集成
- 统一的中文筛选报告

筛选报告包含：

- 选择的特征列表
- 被剔除的特征及原因
- 筛选得分统计

基类
--------

BaseFeatureSelector
~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.BaseFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

SelectionReportCollector
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.SelectionReportCollector
   :members:
   :undoc-members:
   :show-inheritance:

CompositeFeatureSelector
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.CompositeFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

过滤法 - 基础筛选
--------

TypeSelector
~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.TypeSelector
   :members:
   :undoc-members:
   :show-inheritance:

RegexSelector
~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.RegexSelector
   :members:
   :undoc-members:
   :show-inheritance:

NullSelector
~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.NullSelector
   :members:
   :undoc-members:
   :show-inheritance:

ModeSelector
~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.ModeSelector
   :members:
   :undoc-members:
   :show-inheritance:

CardinalitySelector
~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.CardinalitySelector
   :members:
   :undoc-members:
   :show-inheritance:

VarianceSelector
~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.VarianceSelector
   :members:
   :undoc-members:
   :show-inheritance:

过滤法 - 相关性筛选
--------

CorrSelector
~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.CorrSelector
   :members:
   :undoc-members:
   :show-inheritance:

VIFSelector
~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.VIFSelector
   :members:
   :undoc-members:
   :show-inheritance:

过滤法 - 目标导向筛选
--------

InformationValueSelector
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.InformationValueSelector
   :members:
   :undoc-members:
   :show-inheritance:

LiftSelector
~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.LiftSelector
   :members:
   :undoc-members:
   :show-inheritance:

PSISelector
~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.PSISelector
   :members:
   :undoc-members:
   :show-inheritance:

包装法
--------

RFESelector
~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.RFESelector
   :members:
   :undoc-members:
   :show-inheritance:

SequentialFeatureSelector
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.SequentialFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

StepwiseSelector
~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.StepwiseSelector
   :members:
   :undoc-members:
   :show-inheritance:

StepwiseFeatureSelector
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.StepwiseFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

BorutaSelector
~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.BorutaSelector
   :members:
   :undoc-members:
   :show-inheritance:

嵌入法 - 特征重要性
--------

FeatureImportanceSelector
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.FeatureImportanceSelector
   :members:
   :undoc-members:
   :show-inheritance:

NullImportanceSelector
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.NullImportanceSelector
   :members:
   :undoc-members:
   :show-inheritance:

统计检验
--------

MutualInfoSelector
~~~~~~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.MutualInfoSelector
   :members:
   :undoc-members:
   :show-inheritance:

Chi2Selector
~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.Chi2Selector
   :members:
   :undoc-members:
   :show-inheritance:

FTestSelector
~~~~~~~~~~~~~

.. autoclass:: hscredit.core.selectors.FTestSelector
   :members:
   :undoc-members:
   :show-inheritance:

使用示例
--------

基本使用
~~~~~~~~

.. code-block:: python

    import pandas as pd
    import numpy as np
    from hscredit.core.selectors import StepwiseSelector

    # 准备数据
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    y = (X['feature1'] * 0.5 + X['feature2'] * 0.3 + np.random.randn(n) * 0.5 > 0).astype(int)

    # 创建筛选器
    selector = StepwiseSelector(
        estimator='logit',
        direction='both',
        criterion='aic',
        p_enter=0.05,
        p_remove=0.05
    )

    # 拟合并筛选
    selector.fit(X, y)

    # 获取选中特征
    print(selector.select_columns_)

    # 获取筛选报告
    report = selector.get_selection_report()
    print(report)

与Pipeline集成
~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from hscredit.core.selectors import (
        VarianceSelector,
        CorrSelector,
        StepwiseSelector
    )

    pipeline = Pipeline([
        ('variance', VarianceSelector(threshold=0.1)),
        ('correlation', CorrSelector(threshold=0.8)),
        ('stepwise', StepwiseSelector(direction='both', criterion='aic')),
    ])

    pipeline.fit(X, y)
    selected = pipeline.transform(X)

报告收集器
~~~~~~~~~~

.. code-block:: python

    from hscredit.core.selectors import (
        SelectionReportCollector,
        VarianceSelector,
        CorrSelector,
        StepwiseSelector
    )

    collector = SelectionReportCollector(name="特征筛选流程")

    # 添加筛选器
    v selector = VarianceSelector(threshold=0.1)
    selector.fit(X)
    collector.add_report(selector, '粗筛')

    c_selector = CorrSelector(threshold=0.8)
    c_selector.fit(X, y)
    collector.add_report(c_selector, '相关性筛选')

    s_selector = StepwiseSelector(direction='both')
    s_selector.fit(X, y)
    collector.add_report(s_selector, '逐步回归筛选')

    # 获取汇总报告
    summary = collector.get_summary()
    print(summary)

    # 导出为Excel
    collector.to_excel('selection_report.xlsx')
