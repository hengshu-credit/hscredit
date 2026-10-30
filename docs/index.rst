hscredit 文档
=============

.. raw:: html

   <div class="hs-home-hero">
     <p class="hs-home-hero__eyebrow">HENGSHU CREDIT · OPEN SOURCE RISK ENGINEERING</p>
     <div class="hs-home-hero__lockup">
       <span class="hs-home-hero__seal" aria-hidden="true">
         <img src="_static/hengshucredit_logo.svg" alt="" />
       </span>
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

分箱、编码、筛选和学习模型遵循 scikit-learn 接口范式，
可嵌入 ``Pipeline``；所有面向用户的输出（列名、报告、图表、错误信息）均为 **中文**。

.. note::

   当前版本 |release|。基础安装已含 SHAP、OR-Tools 与 PyArrow；Boosting、深度学习、
   调参、PMML 与数据库驱动按需安装，详见 :doc:`installation`。

能力一览
--------

* **变量工程**：17 种一维分箱方法、1 个二维分箱器、9 种编码器、24 个筛选器。
* **探索与评估**：58 个 EDA 函数、43 个指标函数、52 个绘图函数与 1 个树图组件。
* **模型训练**：6 个经典模型、4 个 Boosting 模型、4 个神经网络模型、3 个评分卡模型。
* **规则与交付**：6 个规则挖掘器、9 类数据库适配器、10 个金融函数，以及变量、模型、策略与 Excel 报告。

统计口径与模块入口见 :doc:`overview`。

快速开始
--------

.. code-block:: bash

   pip install hscredit

.. code-block:: python

   from hscredit.utils.datasets import germancredit
   from hscredit.core.binning import OptimalBinning
   from hscredit.core.models import ScoreCard

   df = germancredit()
   X_train = df[["duration_in_month", "credit_amount", "age_in_years"]]
   y_train = df["class"]
   X_test = X_train.iloc[:20]

   binner = OptimalBinning(method="best_iv", max_n_bins=5)
   binner.fit(X_train, y_train)
   X_woe = binner.transform(X_train, metric="woe")

   scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750, binner=binner)
   scorecard.fit(X_woe, y_train)
   scores = scorecard.predict(X_test)

更多端到端示例见 :doc:`quickstart`。

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 入门

   功能总览 <overview>
   安装配置 <installation>
   快速开始 <quickstart>
   标签处理 <overdue_labels>
   规则置换 <rule_swap_approval>
   分析参数 <analysis_parameters>
   接口约定 <api/conventions>
   并行计算 <parallelism>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 建模实践

   模型流程 <articles/model-workflow>
   迁移续训 <api/training_lifecycle>
   损失指标 <articles/losses-guide>
   参数搜索 <articles/tuning-guide>
   手工树 <articles/manual_tree_rule_mining>
   多叉树 <multiway_tree>
   模型解释 <articles/model-interpretability>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: 数据接入

   database

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/index

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: 智能技能

   智能技能 <skills>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 项目

   版本规划 <ROADMAP>
   项目记录 <project_notes>

索引与检索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
