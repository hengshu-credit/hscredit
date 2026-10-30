可编辑的多叉决策树
==================

``MultiwayTreeExtractor`` 在一个节点中直接产生多个分支，可处理数值区间、类别分组、
二分类和多分类标签。训练、编辑、预测和导出的 ``Rule`` 共用 ``BinSpec`` 分支语义。
绘图继续使用现有 ``plot_tree`` 的卡片风格和 matplotlib、pyecharts、graphviz 后端。

完整分步演示见仓库 ``examples/26_multiway_tree.ipynb``，使用 ``examples/hscredit_yyp.xlsx``。

方法速查
--------

.. currentmodule:: hscredit.report.mining

构造参数与所有方法的参数类型、返回值和参考样例见 :class:`MultiwayTreeExtractor`。
下表按操作流程列出入口；调用示例中的 ``tree`` 为已训练实例，``df_train`` / ``df_test`` 为含所需字段的数据表。

.. list-table:: 训练与节点编辑
   :header-rows: 1
   :widths: 28 42 30

   * - 方法
     - 用途
     - 返回值与编辑范围
   * - :meth:`MultiwayTreeExtractor.fit`
     - 使用 DataFrame 中的目标列或显式 y 重新训练
     - 当前实例；成功后替换整棵树，编号从 0 开始
   * - :meth:`MultiwayTreeExtractor.get_split_candidates`
     - 查看指定节点每个特征的最佳候选及统计量
     - DataFrame；不修改模型，也不保证自动生长会采用候选
   * - :meth:`MultiwayTreeExtractor.manual_split`
     - 指定数值切点、类别分组，或选择最佳候选
     - 当前实例；替换指定节点的全部后代
   * - :meth:`MultiwayTreeExtractor.set_thresholds`
     - 修改已有数值分裂节点的切点
     - 当前实例；保留分裂字段，重建后代
   * - :meth:`MultiwayTreeExtractor.grow`
     - 按当前参数继续自动分裂叶节点
     - 当前实例；保留已有分裂，递归处理所选子树
   * - :meth:`MultiwayTreeExtractor.prune` / :meth:`MultiwayTreeExtractor.delete_node`
     - 删除后代，将指定节点变为叶子
     - 当前实例；保留节点自身和其他子树编号

.. list-table:: 预测、规则与模型效果
   :header-rows: 1
   :widths: 28 42 30

   * - 方法
     - 用途
     - 返回结构
   * - :meth:`MultiwayTreeExtractor.get_leaf_node_ids`
     - 获取当前叶节点编号，含空分支
     - ``list[int]``，与叶节点规则列表顺序一致
   * - :meth:`MultiwayTreeExtractor.apply`
     - 返回每个样本到达的叶节点
     - ``(样本数,)`` 整数数组
   * - :meth:`MultiwayTreeExtractor.predict_proba`
     - 返回各类别概率
     - ``(样本数, 类别数)`` 数组，列顺序为 ``classes_``
   * - :meth:`MultiwayTreeExtractor.predict`
     - 返回最大概率类别
     - ``(样本数,)`` 标签数组
   * - :meth:`MultiwayTreeExtractor.get_rules`
     - 导出叶节点或全部节点的路径规则
     - ``list[Rule]``；仅叶节点规则互斥且完整覆盖
   * - :meth:`MultiwayTreeExtractor.to_ruleset`
     - 将选定叶节点或内部节点组合成风险策略
     - OR ``RuleSet``；默认全叶会命中全部样本
   * - :meth:`MultiwayTreeExtractor.get_node_table`
     - 查看结构、切点、分支与各训练类别的计数
     - 每节点一行的中文 DataFrame
   * - :meth:`MultiwayTreeExtractor.get_rule_table`
     - 便捷生成节点规则效果表，默认使用训练数据
     - 同 ``report``；默认包含全部节点
   * - :meth:`MultiwayTreeExtractor.report`
     - 在指定数据集评估节点命中效果，支持多标签和金额
     - DataFrame、表格列表或表格字典；多标签使用多层列
   * - :meth:`MultiwayTreeExtractor.evaluate`
     - 评估整棵树的准确率、交叉熵、AUC、KS
     - 每个数据集一行的中文 DataFrame

.. list-table:: 展示与持久化
   :header-rows: 1
   :widths: 28 42 30

   * - 方法
     - 用途
     - 返回值
   * - :meth:`MultiwayTreeExtractor.plot`
     - 使用 matplotlib、pyecharts 或 graphviz 绘图
     - 对应后端的图形对象；可通过 save 保存
   * - :meth:`MultiwayTreeExtractor.display`
     - 在 Notebook 展示树图和训练集全部节点规则表
     - 当前实例；图表作为展示输出
   * - :meth:`MultiwayTreeExtractor.save`
     - 保存完整模型与编辑所需的训练数据
     - 保存路径字符串
   * - :meth:`MultiwayTreeExtractor.load`
     - 从可信文件恢复模型，保留原有路由
     - 新恢复的模型实例

``get_params`` / ``set_params`` 沿用 sklearn 估计器接口；``set_params`` 只更新配置，
已有分裂不会自动重建。需要整体应用新参数时重新 ``fit``，仅继续生长时调用 ``grow``。
继承的 ``score(X, y)`` 返回准确率；完整模型效果使用 ``evaluate``。
``save_artifact`` / ``load_artifact`` 提供额外的序列化引擎、压缩和元数据校验选项。

训练与编辑
----------

.. code-block:: python

   import numpy as np
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
``thresholds`` 中的有限切点必须严格递增。类别用 ``category_groups`` 指定互不重复、完整覆盖节点训练类别的分组。
``missing_branch`` 是从 0 开始的分支序号。未声明缺失策略时，节点训练样本有缺失则自动比较合并方向；
该节点分裂字段没有缺失时，生产中新出现的缺失默认走最左分支（0 号），与各分支样本量无关。
此规则按每个节点实际收到的训练样本判断，适用于自动训练、手工分裂和继续生长；
显式 ``missing_branch``、缺失标记或独立缺失配置在训练无缺失时仍然生效。

缺失值声明与 binner 一致
----------------------------------------

数值规则复用 binner 的解析器：``np.nan``、``None``、``pd.NA`` 都可以作为缺失标记，最多出现一次。
标记的位置表示合并目标普通分支，不能把它当作一个数值阈值或独立分支：

.. code-block:: python

   # 合并进普通分支 0：(-inf, 0.12]
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[np.nan, 0.12, 0.18, 0.25])
   # 合并进普通分支 1：(0.12, 0.18]
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[0.12, np.nan, 0.18, 0.25])
   # 显式 missing_branch 优先，因此此处仍是四叉，缺失走 0 号分支
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[0.12, 0.18, 0.25, np.nan], missing_branch=0)

因此 ``[0.12, np.nan, 0.18, 0.25]`` 等价于有限切点配合 ``missing_branch=1``，
不等价于 ``missing_branch=0``。切点与标记分开生成，便于调整目标分支后同步验证：

.. code-block:: python

   cuts = [0.12, 0.18, 0.25]
   merge_branch = 1  # 可改为 0、1、2、3，分别对应从左到右的四个普通分支
   marked_cuts = cuts.copy()
   marked_cuts.insert(merge_branch, np.nan)
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=marked_cuts)

独立缺失分支由 ``missing_separate=True`` 指定，数值型追加在普通分支之后：

.. code-block:: python

   # 三个阈值对应四个普通分支，另加缺失分支，共五叉
   tree.set_params(max_branches=5)
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[0.12, 0.18, 0.25], missing_separate=True)
   # 没有普通切点时，可以只分成“非缺失”和“缺失”两支
   tree.manual_split(feature="衡枢鉴真分老客版", thresholds=[], missing_separate=True)

类别规则中 ``[np.nan]`` 表示独立缺失分支，``[类别, np.nan]`` 表示合并。
独立组可位于任意位置，分支顺序保持声明顺序；若没有缺失标记，``missing_separate=True`` 追加独立组。

.. code-block:: python

   # 假设训练类别为 A、B、C
   tree.manual_split(feature="商品类别", category_groups=[["A"], [np.nan], ["B", "C"]])
   tree.manual_split(feature="商品类别", category_groups=[["A", np.nan], ["B", "C"]])

优先级为：显式 ``missing_branch`` > 规则中的合并标记 > ``missing_separate`` 配置。
独立类别组 ``[np.nan]`` 与指向其他分支的 ``missing_branch`` 冲突时会报错，避免留下语义不明的空组。
构造参数 ``missing_separate=False`` 保持原来的自动合并行为；设为 True 可在自动训练、``grow`` 中使用独立分支。
方法参数 ``missing_separate=None`` 沿用构造配置。

缺失分支计入 ``max_branches``，独立缺失分支不受普通分支 ``min_samples_leaf`` 限制。
在实际分裂节点上，即使训练时没有缺失也会保留该分支，计数为 0，未来缺失样本使用最近非空祖先的概率。
类别的未知值在合并模式下与缺失同路，在独立模式下走样本最多的普通类别分支，不会进入纯缺失分支。
``apply``、``predict_proba``、``Rule``、节点统计及三个绘图后端共用这套分支定义。

调整阈值或调用 ``manual_split`` 会替换所选节点的后代。节点自身和其他子树编号不变，
新后代取得新编号；``grow=True`` 可在替换后继续自动生长。``prune`` 和 ``delete_node``
将所选节点变为叶子，``grow`` 从现有叶子继续训练。失败的编辑不会改变原模型。
``set_thresholds`` 未显式声明新缺失策略或缺失标记时保留原节点的路由方式。
若减少普通分支后原来的合并序号越界，需明确提供新的 ``missing_branch``。
编辑使用拟合时保存的数据；若需要改训练样本，请重新 ``fit``。

节点编号与分支位置
------------------

``node`` 参数使用 ``get_node_table()`` 或 ``nodes_`` 中的节点编号。
``missing_branch`` 使用当前节点 ``children`` 列表中的位置：

.. code-block:: python

   root = tree.nodes_[0]
   missing_child_id = root.children[root.missing_branch]
   child_to_edit = root.children[1]  # 根节点的第二个分支对应的节点编号

二者不可以直接互换。编辑后新子节点使用新编号，编号可能不连续；应重新获取所需节点，
不要保存旧的后代编号继续编辑。数值和类别分支均按 ``children`` 顺序从左到右绘制，
``get_leaf_node_ids()`` 则按节点创建顺序返回，局部编辑后不保证等于图上的左右顺序。

人工阈值允许产生空分支，其样本计数仍为 0，预测概率继承最近的非空祖先。
已保存模型保留训练时的分支语义；更新库后需重新训练或重新分裂节点，才会使用新的默认路由。
自动分裂执行 ``min_samples_split``、普通叶节点的 ``min_samples_leaf``、``max_depth`` 和 ``max_branches`` 约束。
``get_rules()`` 返回互斥且完整覆盖输入的叶子规则；``to_ruleset(nodes=[...])`` 可把选定节点合成
风险策略。默认全叶 OR 规则集会命中所有样本，不能直接当作拒绝策略。

报告与模型效果的输入输出
------------------------

``get_node_table()`` 展示训练时的节点结构和计数；``report`` 在给定数据上重新计算每个节点
规则的命中效果；``evaluate`` 评估整棵树的预测。三者对应不同层次的结果：

.. code-block:: python

   structure = tree.get_node_table()
   train_rules = tree.get_rule_table(leaf_only=True)
   test_rules = tree.report(df_test, leaf_only=True)
   reports = tree.report({"训练集": df_train, "测试集": df_test}, leaf_only=True)
   train_report = reports["训练集"]  # report 保留输入字典的结构
   effects = tree.evaluate({"训练集": df_train, "测试集": df_test})  # evaluate 汇总为一张表
   external_y_effects = tree.evaluate(df_test[tree.features_], y=df_test["FPD"])

``evaluate`` 的显式标签是第二个参数，不支持 ``evaluate((X, y))``；使用数据集字典时，
每张表都应含构造参数 ``target`` 对应的列，外层 y 不参与评估。
``report`` 的单标签结果使用单层中文列，多标签结果使用 MultiIndex 列；
``report`` 和 ``get_rule_table`` 默认包含全部节点，父子节点命中存在重叠，不能直接相加计数。
只需互斥叶节点结果时传 ``leaf_only=True``。

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
