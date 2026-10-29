可编辑的多叉决策树
==================

``MultiwayTreeExtractor`` 在一个节点中直接产生多个分支，可处理数值区间、类别分组、
二分类和多分类标签。训练、编辑、预测和导出的 ``Rule`` 共用 ``BinSpec`` 分支语义。
绘图继续使用现有 ``plot_tree`` 的卡片风格和 matplotlib、pyecharts、graphviz 后端。

完整分步演示见仓库 ``examples/26_multiway_tree.ipynb``，使用 ``examples/hscredit_yyp.xlsx``。

训练与编辑
----------

.. code-block:: python

   from hscredit import MultiwayTreeExtractor, plot_tree

   tree = MultiwayTreeExtractor(
       target="FPD", features=["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别"],
       max_branches=4, max_depth=3, criterion="chi2",
   ).fit(df_train)
   candidates = tree.get_split_candidates(node=0)
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[0.12, 0.18, 0.25], node=0)
   tree.set_thresholds(0, [0.10, 0.20, 0.30])
   child = tree.nodes_[0].children[1]
   tree.grow(node=child)
   tree.prune(node=child)
   plot_tree(tree)
   effects = tree.evaluate({"训练集": df_train, "测试集": df_test})
   table = tree.report(df_test, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额")
   rules = tree.get_rules()
   tree.save("多叉树.pkl")
   restored = MultiwayTreeExtractor.load("多叉树.pkl")

数值分支为 ``(-inf, t1]``、``(t1, t2]``、……、``(tk, +inf)``，因此等于阈值时走左侧区间。
``thresholds`` 必须严格递增。类别用 ``category_groups`` 指定互不重复、完整覆盖节点训练类别的分组。
``missing_branch`` 是从 0 开始的分支序号；自动搜索会比较缺失值归属，没有缺失时选择样本最多的分支。
新类别与缺失走同一条已保存的分支，规则表达式也包含该兜底条件。

调整阈值或调用 ``manual_split`` 会替换所选节点的后代。节点自身和其他子树编号不变，
新后代取得新编号；``grow=True`` 可在替换后继续自动生长。``prune`` 和 ``delete_node``
将所选节点变为叶子，``grow`` 从现有叶子继续训练。失败的编辑不会改变原模型。
编辑使用拟合时保存的数据；若需要改训练样本，请重新 ``fit``。

人工阈值允许产生空分支，其样本计数仍为 0，预测概率继承最近的非空祖先。
自动分裂始终执行 ``min_samples_split``、``min_samples_leaf``、``max_depth`` 和 ``max_branches`` 约束。
``get_rules()`` 返回互斥且完整覆盖输入的叶子规则；``to_ruleset(nodes=[...])`` 可把选定节点合成
风险策略。默认全叶 OR 规则集会命中所有样本，不能直接当作拒绝策略。

分裂搜索与 SAS 的关系
----------------------------------------

参考 SAS Enterprise Miner 的多叉分支和节点编辑操作，采用其公开的 Pearson 卡方、Gini、Entropy
准则思想。这里是独立实现，不声明与 SAS 输出相同：

* ``chi2`` 使用分支与类别的列联表，按 ``-log10(调整后 P 值)`` 排序，``alpha`` 控制自动生长。
  默认按该特征实际搜索方案数作 Bonferroni 调整；不实现 SAS 的 Kass 或深度调整。
* ``gini`` 与 ``entropy`` 最大化父节点不纯度减去按子节点样本量加权的不纯度。
* 数值候选池复用 ``hscredit.core.binning.QuantileBinning`` 的无监督切点计算。
  候选组合不超过 ``max_candidates`` 时联合穷举多切点；超过时逐步增加切点，不能保证全局最优。
* 类别从各类别独立成组开始，按最大 Pearson P 值逐次合并相似组，比较不超过 ``max_branches``
  的候选分组。这是启发式搜索，不强加类别顺序，也不声称实现完整 CHAID。

模型效果中二分类提供 AUC、KS、准确率和交叉熵，多分类提供 OVR AUC、准确率和交叉熵。
节点规则表使用 ``Rule.report``；多分类默认按各类别对其余类别输出指标。
二分类测试集只有一类时 AUC 为 NaN、KS 为 0；多分类测试集未覆盖全部训练类别时 OVR AUC 为 NaN。

参考：`SAS 分裂准则 <https://support.sas.com/documentation/cdl/en/stathpug/68163/HTML/default/stathpug_hpsplit_details02.htm>`_，
`SAS 交互式训练与剪枝 <https://support.sas.com/documentation/cdl/en/emgsj/64144/HTML/default/n1vdha504xxy5tn10t9azejt0n7t.htm>`_。

API
---

完整参数和方法见 :class:`hscredit.report.mining.MultiwayTreeExtractor`。
