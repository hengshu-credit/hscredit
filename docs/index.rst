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

* **数据准备**：读取内置现金贷数据或接入业务数据库，定义多期逾期标签，按客户或时间切分样本。
* **变量分析**：检查质量与稳定性，完成分箱、编码、特征衍生和组合筛选。
* **模型研发**：训练经典模型、Boosting、表格神经网络和评分卡，完成调参、概率校准、解释与部署。
* **策略与交付**：挖掘规则树、回测策略置换，生成特征、模型、规则和策略报告，可通过 Agent Skills 调用。

按任务查找 Python 入口和指南，见 :doc:`overview`。

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
   :caption: 入门

   安装指南 <installation>
   快速开始 <quickstart>
   功能总览 <overview>
   功能与示例 <showcase>

.. toctree::
   :maxdepth: 2
   :caption: 数据集

   现金贷数据集 <datasets>
   特征摘要报告 <dataset-summary>

.. toctree::
   :maxdepth: 1
   :caption: 风控分析

   标签与灰样本 <overdue_labels>
   分析参数 <analysis_parameters>
   规则置换评估 <rule_swap_approval>
   手工树挖掘 <articles/manual_tree_rule_mining>
   多叉树编辑 <multiway_tree>

.. toctree::
   :maxdepth: 1
   :caption: 建模实践

   模型训练与保存 <articles/model-workflow>
   预训练与续训 <api/training_lifecycle>
   损失与评估 <articles/losses-guide>
   超参数搜索 <articles/tuning-guide>
   模型解释 <articles/model-interpretability>
   报告与部署 <articles/report-resources-deployment>

.. toctree::
   :maxdepth: 2
   :caption: 工具与集成

   数据库接入 <database>
   并行计算 <parallelism>
   Agent Skills <skills>

.. toctree::
   :maxdepth: 2
   :caption: 开发参考

   接口约定 <api/conventions>
   api/index

索引与检索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
