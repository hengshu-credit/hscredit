Boosting 模型 ``hscredit.core.models.boosting``
===============================================

基于梯度提升框架的风控模型，统一继承 ``BaseRiskModel``，需要安装对应可选依赖
（``pip install hscredit[boost]``）。可直接从顶层 ``hscredit`` 懒加载导入，例如
``from hscredit import XGBoostRiskModel``。

.. note::

   ``NGBoostRiskModel`` 依赖 ``ngboost``；构建本文档时若未安装该依赖，相关签名以
   mock 形式呈现，不影响接口说明。

.. autoclass:: hscredit.core.models.boosting.xgboost_model.XGBoostRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.boosting.lightgbm_model.LightGBMRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.boosting.catboost_model.CatBoostRiskModel
   :members:
   :show-inheritance:

.. autoclass:: hscredit.core.models.boosting.ngboost_model.NGBoostRiskModel
   :members:
   :show-inheritance:
