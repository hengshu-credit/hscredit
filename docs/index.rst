hscredit 文档
=============

.. raw:: html

   <div class="hs-home-hero">
     <p class="hs-home-hero__eyebrow">HENGSHU CREDIT · OPEN SOURCE RISK ENGINEERING</p>
     <div class="hs-home-hero__lockup">
       <span class="hs-home-hero__seal" aria-hidden="true">衡</span>
       <div>
         <p class="hs-home-hero__motto">鉴真伪 · 斟信用 · 衡风险 · 枢定策</p>
         <p class="hs-home-hero__title">让信用风险建模有据可循</p>
         <p class="hs-home-hero__lead">从真实信贷数据出发，把分箱、评分卡、规则与报告交付收进同一套中文工具链。</p>
       </div>
     </div>
     <div class="hs-home-hero__actions" aria-label="衡枢项目入口">
       <a class="hs-home-hero__action hs-home-hero__action--primary"
          href="https://hengshucredit.com" target="_blank" rel="noopener noreferrer">
         访问衡枢官网 <span aria-hidden="true">↗</span>
       </a>
       <a class="hs-home-hero__action"
          href="https://github.com/hengshu-credit/hscredit" target="_blank" rel="noopener noreferrer">
         查看 GitHub 仓库 <span aria-hidden="true">↗</span>
       </a>
     </div>
     <div class="hs-home-hero__track" aria-label="hscredit 核心能力">
       <span>数据探索</span><span>变量工程</span><span>风险建模</span><span>策略规则</span><span>报告交付</span>
     </div>
   </div>

**hscredit（衡枢真信）** 是面向金融信贷行业的量化分析工具箱，为风控策略分析人员与
模型人员提供覆盖 **数据探索、变量评估、分箱编码、特征筛选、评分卡建模、机器学习风控
模型、策略规则分析、模型监控与报告交付** 的全流程能力。

所有组件均遵循 scikit-learn 的 ``BaseEstimator`` / ``TransformerMixin`` 接口范式，
可直接嵌入 ``Pipeline``；所有面向用户的输出（列名、报告、图表、错误信息）均为 **中文**。

.. note::

   当前版本 |release|。基础安装保持轻量，Boosting、深度学习、调参、解释、PMML
   等能力按需安装，详见 :doc:`installation`。

能力一览
--------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - 风控工作
     - 能力规模
     - 说明
   * - 数据探索分析
     - 57 种 EDA
     - 数据质量、目标分布、坏率趋势、客群画像/迁移、Vintage、Roll Rate、策略仿真
   * - 变量分箱
     - 18 种分箱器
     - 等频/等宽/卡方/树/CART/Best IV/Best KS/Best Lift/MDLP/单调/遗传/二维分箱
   * - 特征编码
     - 9 种编码器
     - WOE、目标编码、频数编码、独热、序数、分位数、CatBoost/GBM 编码
   * - 特征筛选
     - 23 种筛选器
     - 缺失率、众数率、方差、相关性、VIF、IV、Lift、PSI、模型重要性、逐步回归、Boruta
   * - 风控指标
     - 43 种指标
     - KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、分箱统计、分类与回归指标
   * - 模型训练
     - 36 个建模组件
     - 逻辑回归、评分卡、随机森林、GBDT、XGBoost、LightGBM、CatBoost、NGBoost、规则模型
   * - 可视化分析
     - 46 种图表
     - 分箱趋势、KS/ROC/PR/Lift/Gain、评分分布、策略阈值、Vintage、漂移、树图
   * - 报告交付
     - 28 种报告工具
     - 特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比、Excel 输出

快速开始
--------

.. code-block:: bash

   pip install hscredit

.. code-block:: python

   import hscredit
   from hscredit.core.binning import OptimalBinning
   from hscredit.core.models import ScoreCard

   binner = OptimalBinning(method="best_iv", max_n_bins=5)
   binner.fit(X_train, y_train)
   X_woe = binner.transform(X_train, metric="woe")

   scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750, binner=binner)
   scorecard.fit(X_woe, y_train)
   scores = scorecard.predict(X_test)

更多端到端示例见 :doc:`quickstart`。

.. toctree::
   :maxdepth: 2
   :caption: 入门

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/index

.. toctree::
   :maxdepth: 1
   :caption: 项目

   ROADMAP

索引与检索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
