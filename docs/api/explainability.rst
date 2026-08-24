模型可解释性
============

``hscredit.core.models.explainability`` 统一结构化 SHAP 结果、全局和局部解释、中文绘图、业务原因码与
受约束反事实建议。反事实仅表示模型条件下的非因果候选变化，不构成授信承诺或审批依据。

.. autoclass:: hscredit.core.models.explainability.ExplanationResult
   :members:

.. autoclass:: hscredit.core.models.explainability.ModelExplainer
   :members:

.. autoclass:: hscredit.core.models.explainability.CounterfactualExplainer
   :members:

.. autofunction:: hscredit.core.models.explainability.model_explain_report

.. autofunction:: hscredit.core.models.explainability.build_reason_codes

.. autofunction:: hscredit.core.models.explainability.plot_feature_importance

.. autofunction:: hscredit.core.models.explainability.plot_shap_importance

.. autofunction:: hscredit.core.models.explainability.plot_importance_comparison
