可编辑的多叉决策树
==================

``MultiwayTreeExtractor`` 在一个节点中直接产生多个分支，可处理数值区间、类别分组、
二分类和多分类标签。训练、编辑、预测和导出的 ``Rule`` 共用 ``BinSpec`` 分支语义。
绘图继续使用现有 ``plot_tree`` 的卡片风格和 matplotlib、pyecharts、graphviz 后端。

业务流程演示见仓库 ``examples/26_multiway_tree.ipynb``，使用 ``examples/hscredit_yyp.xlsx``。
Notebook 依次演示自动建树、节点剪枝、自动重选特征、指定特征自动分裂、指定特征与阈值，
展示最终树并导出全部节点规则。在训练集、验证集、测试集上复核 FPD 后，
直接调用 ``report(..., overdue=['MOB1'], dpds=[7, 3, 0], save=...)`` 输出 Excel 挖掘报告。
Notebook 不再自行组装工作簿；金额口径可通过同一接口的 ``amount`` 参数选择。
输出文件为 ``examples/model_report/multiway_tree/多叉树规则挖掘业务报告.xlsx``。

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
       max_n_bins=4, max_depth=3, criterion="chi2",
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
   tree.set_params(max_n_bins=4)
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

独立缺失分支不计入 ``max_n_bins``，也不受普通分支 ``min_samples_leaf`` 限制。
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
自动分裂执行 ``min_samples_split``、普通叶节点的 ``min_samples_leaf``、``max_depth`` 和 ``max_n_bins`` 约束。
``get_rules()`` 返回互斥且完整覆盖输入的叶子规则；``to_ruleset(nodes=[...])`` 可把选定节点合成
风险策略。默认全叶 OR 规则集会命中所有样本，不能直接当作拒绝策略。

初始化规则筛选与分裂参数
------------------------

``ManualTreeExtractor``、``MultiwayTreeExtractor`` 和 ``DecisionTreeAnalyzer`` 共用
``rule_filter`` 配置与 ``report(..., save=...)`` 导出逻辑。
默认 ``rule_filter=None`` 不筛选；显式传入字典后才启用筛选。

.. code-block:: python

   from hscredit.report.mining import ManualTreeExtractor, MultiwayTreeExtractor

   rule_filter = {
       "leaf_only": True,
       "metric": "LIFT",        # 也支持“LIFT值”和“坏样本率”
       "threshold": 1.5,
       "operator": ">=",        # >、>=、<、<=
       "min_samples": 0.02,     # 浮点数表示训练样本占比
       "max_samples": 500,      # 整数表示订单个数
   }
   binary = ManualTreeExtractor(
       target="FPD", features=features, max_depth=3,
       min_samples_split=50, min_samples_leaf=25,
       criterion="gini", rule_filter=rule_filter,
   ).fit(df_train)
   multiway = MultiwayTreeExtractor(
       target="FPD", features=features, max_depth=3, max_n_bins=4,
       min_samples_split=50, min_samples_leaf=25,
       criterion="gini", rule_filter=rule_filter,
   ).fit(df_train)
   binary.report(datasets, target="FPD", amount="放款金额", save="二叉树策略挖掘.xlsx")
   multiway.report(datasets, target="FPD", amount="放款金额", save="多叉树策略挖掘.xlsx")
   selected = multiway.get_rule_table()
   complete = multiway.get_rule_table(apply_rule_filter=False)

显式空字典 ``{}`` 的默认条件为：仅叶节点、``LIFT值 >= 1``、样本量不设上下限。
坏样本率阈值用小数表示，例如 0.2 表示 20%；样本量比例使用 [0, 1] 内的浮点数，
整数 1 表示 1 单，浮点数 1.0 表示全部样本。上下限均包含边界。
``leaf_only=False`` 允许筛选中间节点；根节点只作总体基准，不参与显式筛选。

筛选使用当前最终树在保存的训练集上的、未舍入的订单指标与单一 0/1 训练目标。
其他数据集、金额口径和其他评估标签沿用同一组节点及顺序，不在验证集上重新筛选。
每次报告重新检查最终树，因此剪枝、重新分裂后筛选自动更新。
筛选为空时输出空规则表，仍展示完整树；``apply_rule_filter=False`` 可临时关闭筛选。
``get_rules``、``get_node_table``、预测与树结构始终保持完整。
分裂参数继续使用已有构造参数，二叉树的其他 sklearn 参数通过关键字传入；
``DecisionTreeAnalyzer`` 也保留已有的 ``tree_params`` 字典入口。

2026-10-30 验证：Python 3.11.5（仓库验收解释器
``.audit_tmp/defect-repairs/runtime/Scripts/python.exe``）。树报告、数据集传参、规则筛选、
ExcelWriter、二叉/多叉树及其可视化共覆盖 338 项用例；规则筛选模块补充后单独运行 79 项通过。
真实数据对两种手工树执行节点分裂后，按训练集 LIFT 与样本占比筛选，再核对三个数据集的
多标签订单/金额结果、日期与商品类别分组、Excel 合并范围及数值格式。业务 Notebook 已完整执行。
筛选与分组用例可运行 ``python -m pytest tests/test_report/test_tree_rule_filter.py -q --tb=short``。

报告与模型效果的输入输出
------------------------

数据集传参沿用 ``auto_model_report`` 的常用约定：每个 DataFrame 自带目标列，
统一通过 ``datasets`` 传入字典或列表，通过 ``target`` 指定目标列名。

.. list-table:: 数据集容器与命名
   :header-rows: 1
   :widths: 40 35 25

   * - 传参
     - 报告中的数据集名称
     - 树 report 返回值
   * - ``{"验证集": df_valid}``
     - 保留字典键和插入顺序
     - 单项字典
   * - ``{"训练集": train, "验证集": valid, "测试集": test}``
     - 保留全部名称和顺序，可继续增加数据集
     - 同名表格字典
   * - ``[df_valid]``
     - 数据集1
     - 单元素列表
   * - ``[train, valid, test, oot1, oot2]``
     - 数据集1 至 数据集5
     - 同序表格列表

.. code-block:: python

   # 每张表均包含 FPD 以及规则使用的特征。
   datasets = {"训练集": df_train, "验证集": df_valid, "测试集": df_test}
   reports = tree.report(datasets=datasets, target="FPD", save="规则报告.xlsx")
   valid_rules = reports["验证集"]

   # 列表形式不需要额外命名，单数据集也按同一约定调用。
   single = tree.report(datasets=[df_valid], target="FPD")
   multiple = tree.report(datasets=[df_train, df_valid, df_test], target="FPD")

``get_rule_table`` 使用相同的数据集约定。推荐显式传入 ``target``，避免依赖模型的默认目标列。
已有的单个 DataFrame 便捷调用仍返回 DataFrame。
若传入 ``amount``、``date`` 或 ``overdue``，每张表也应包含对应字段。
``target='FPD'`` 是单标签；``overdue=['MOB1'], dpds=[7]`` 也是单标签，二者不混用。
数据集数量与标签数量独立，单标签可以同时分析多个数据集。

``auto_model_report`` 返回 ``ModelReport`` 对象，并使用 ``excel_path`` 保存；
树报告返回表格容器，并使用 ``save`` 保存。数据集命名和目标列的传参约定一致。

2026-10-30 验证：Python 3.11.5，三种树的 dict/list、单标签、1/2/5 个数据集共 48 项测试通过。
使用 ``examples/hscredit_yyp.xlsx`` 实跑字典单集、列表单集、字典五集、列表五集，
核对返回类型、名称与顺序、规则命中数、坏样本数和金额，以及 Excel 图表数量、规则顺序与筛选表。
复现测试：``python -m pytest tests/test_report/test_tree_report_dataset_inputs.py -q --tb=short``。

多叉树、手工二叉树和 ``DecisionTreeAnalyzer`` 使用一致的规则接口：
``get_rule_table(datasets=None, target=None, overdue=None, dpds=None, ...)`` 默认展示所有节点效果，
``get_rules()`` 返回叶节点 ``Rule``，``get_rules(leaf_only=False)`` 返回含根节点的全部路径。
旧多叉树 ``get_rule_table(data=...)`` 仍作为兼容别名保留。

``report`` 参考 scorecardpipeline ``DecisionTreeRuleExtractor.report`` 的 ``sheet`` / ``save``
调用方式，新增直接导出能力；保存前后返回的 DataFrame、列表或字典结构保持一致：

.. code-block:: python

   datasets = {"训练集": df_train, "验证集": df_valid, "测试集": df_test}
   reports = tree.report(
       datasets, overdue=["MOB1"], dpds=[7, 3, 0], leaf_only=False,
       round_digits=False, date="放款时间", freq="M", data_source="真实放款数据",
       sheet="组合策略汇总", save="多叉树挖掘报告.xlsx",
   )

工作簿使用单张连续报告页，按“基本信息与模型性能、挖掘摘要、策略详情、最终树在各数据集上的应用”排列。
基本信息复用模型报告的样本统计、日期分布和预测性能；AUC、KS 复用 ``ModelReport``。
主报告沿用模板列宽；仅特征分箱页自动调整列宽，最大 25，所有页面不自动换行。
订单和金额的样本统计均按“样本总数、好样本数、坏样本数、坏样本率、AUC、KS”排列；
金额维度前三列为相应金额，保留两位小数。``group="商品类别"`` 或字段列表可与日期分组组合，
样本分布按“数据集、日期、其他分组”逐层合并相同单元格，不跨父分组合并。
特征效果汇总保持宽表，按入参数据集顺序、指标名称、订单/金额、分箱方法排序并逐层合并元数据列。

挖掘摘要包括规则总数（含根节点）、叶节点、中间节点（含非叶根节点）、树深度（根为 0）、参数配置，
以及成功执行的人工修改记录。LIFT 按 ``[0,0.25)``、``[0.25,0.5)``、``[0.5,1)``、
``[1,1.5)``、``[1.5,2)``、``[2,3)``、``[3,5)``、``[5,+inf)`` 统计训练规则分布。
重新 fit 清空编辑历史；旧模型没有历史时仅展示其当前人工调整节点，不推测过去的操作。

策略详情及每个数据集区块均按“标题、树图、规则表”排列。规则表统一包含节点编号、是否叶子、是否调整，
不再重复输出完整节点表。数据集上的树图只替换命中统计，不重新训练或改变分裂阈值，所有分支均保留。
连线只显示分裂条件，节点卡片显示命中订单数或金额；未命中的节点显示 0。
每个区块的规则与顺序严格遵循训练集策略详情。
多标签表保留全部标签，树图使用该区块的首个标签，并在标题注明。

传入 ``amount`` 后同时生成订单与金额两套树图和规则表，以及金额加权 AUC/KS。
``report`` 返回值保持兼容：传入 amount 时返回金额表，Excel 同时包含两种口径。
训练详情使用建树标签，数据集应用区块使用本次传入的 target 或 overdue/dpds 标签。
``get_node_table()`` 返回完整结构，二分类增加坏率、LIFT 等规则指标，多分类保留各类别计数和占比。
叶子自身没有分裂特征，所属客群由“指标含义”和“分支条件”描述；父节点编号使用可空整数。

可以用已有 ``ExcelWriter`` 汇总不同口径，使用不同的 ``sheet`` 避免覆盖同名报告：

.. code-block:: python

   from hscredit.excel import ExcelWriter

   with ExcelWriter() as writer:
       writer.set_filename("多口径树报告.xlsx")
       tree.report(datasets, target="FPD", sheet="FPD件数", excel_writer=writer)
       tree.report(datasets, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额",
                   sheet="MOB1金额", excel_writer=writer)

同名报告页重建，其他工作表保留。金额策略表中的命中数、好坏样本数表示金额，
坏样本率为金额坏率；准确率、精确率、召回率和 F1 仍按件数计算。
规则清单包括重叠的父子路径，不能对所有节点的样本数直接求和。

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
``report`` / ``get_rule_table`` 默认 ``long_format=True``，按节点、标签顺序返回长表，
前五列为节点编号、指标含义、组合策略、是否叶子、是否调整。``long_format=False`` 保留旧宽表接口；
``report`` 和 ``get_rule_table`` 默认包含全部节点，父子节点命中存在重叠，不能直接相加计数。
只需互斥叶节点结果时传 ``leaf_only=True``。

最大分叉数统一使用 ``max_n_bins``，例如 ``MultiwayTreeExtractor(max_n_bins=4)``。
该上限对每个节点的普通分支生效，不含独立缺失分支，不是整棵树的叶节点总数。
例如 4 个普通分支加 1 个独立缺失分支时只需要 ``max_n_bins=4``；缺失合并时不增加分支。
``max_n_bins=1, missing_separate=True`` 允许仅按“非缺失 / 缺失”分成两支。
两个参数均未提供时，默认最多 4 个普通分支，独立缺失分支另计。
旧参数 ``max_branches`` 显式传入时保留原来的总分支数含缺失语义；同时提供时显式 ``max_n_bins`` 优先。
``set_params(max_n_bins=...)`` 控制后续训练和节点分裂，
已有节点需重新 fit 或重分裂才改变结构。
报告参数 ``bin_params={"max_n_bins": 5}`` 只设置特征分箱页的箱数，不改变树的分叉上限。

分裂搜索与 SAS 的关系
----------------------------------------

参考 SAS Enterprise Miner 的多叉分支和节点编辑操作，采用其公开的 Pearson 卡方、Gini、Entropy
准则思想。这里是独立实现，不声明与 SAS 输出相同：

* ``chi2`` 使用分支与类别的列联表，按 ``-log10(调整后 P 值)`` 排序，``alpha`` 控制自动生长。
  默认按该特征实际搜索方案数作 Bonferroni 调整；不实现 SAS 的 Kass 或深度调整。
* ``gini`` 与 ``entropy`` 最大化父节点不纯度减去按子节点样本量加权的不纯度。
* 数值候选池复用 ``hscredit.core.binning.QuantileBinning`` 的无监督切点计算。
  候选组合不超过 ``max_candidates`` 时联合穷举多切点；超过时逐步增加切点，不能保证全局最优。
* 类别从各类别独立成组开始，按最大 Pearson P 值逐次合并相似组，比较不超过 ``max_n_bins``
  的候选分组。这是启发式搜索，不强加类别顺序，也不声称实现完整 CHAID。

模型效果中二分类提供 AUC、KS、准确率和交叉熵，多分类提供 OVR AUC、准确率和交叉熵。
节点规则表使用 ``Rule.report``；多分类默认按各类别对其余类别输出指标。
二分类测试集只有一类时 AUC 为 NaN、KS 为 0；多分类测试集未覆盖全部训练类别时 OVR AUC 为 NaN。

参考：`SAS 分裂准则 <https://support.sas.com/documentation/cdl/en/stathpug/68163/HTML/default/stathpug_hpsplit_details02.htm>`_，
`SAS 交互式训练与剪枝 <https://support.sas.com/documentation/cdl/en/emgsj/64144/HTML/default/n1vdha504xxy5tn10t9azejt0n7t.htm>`_。

最优性的核验（2026-10-31）
----------------------------------------

``fit`` 从根节点递归生长；每个节点按当前路径取得训练样本，搜索各特征的候选分裂，
选择当前评分最高的方案后递归处理子节点。没有子树前瞻、回溯或全局树优化。
数值候选、类别合并、节点选择及整树递归是不同层次，不能混称“全局最优”。

默认 ``max_prebins=12`` 时数值特征通常最多 11 个候选点，4 个普通分支对应
``C(11,1)+C(11,2)+C(11,3)=231`` 个组合，在默认 ``max_candidates=5000`` 内会穷举。
它只保证候选池及样本约束内的最佳分裂；唯一值不超过 max_prebins 时才会使用全部相邻唯一值中点。
将 max_prebins 提高到 50 后，49 个候选点对应 19649 个上述组合，会超过默认预算并进入贪心搜索。
仅提高预分箱数不能作为最优性保证。

核验使用 Python 3.11.5、``examples/hscredit_yyp.xlsx``，按业务 Notebook 的 FPD 分层随机拆分
取 582 条训练样本，随机种子 42。保持最多 4 个普通分支、每叶至少 25 单，
用全部相邻唯一值边界上的动态规划与原始行 Gini 公式独立对照；另用 15 组小样本穷举核验动态规划结果。

.. list-table:: 根节点数值特征的 Gini 不纯度下降
   :header-rows: 1

   * - 特征
     - 当前候选池最优
     - 全部有效边界最优
   * - 衡枢鉴真分老客版
     - 0.00818176
     - 0.01721117
   * - 近六个月非银多头机构数
     - 0.00370336
     - 0.00438628
   * - 青云24
     - 0.00508643
     - 0.00820892

最终树节点 2 的 436 条样本中，固定“近六个月非银多头机构数”：业务手工切点 ``[50, 65, 75]``
的增益为 0.00178994；候选池自动切点 ``[64, 66, 73]`` 为 0.00223014；
相同约束下全部边界搜索的 ``[63.5, 66.5, 75.5]`` 为 0.00341347。
这些数值只证明训练节点存在更好的不纯度下降，不保证验证集 AUC、KS、LIFT 或业务收益更好。

其他已复现的边界：

* 120 条有序样本，前 55 条为 0、后 65 条为 1，最大二叉、每叶至少 10 单。
  等频候选选出 59.5，增益 0.42013889；全部边界中的 54.5 可达 0.49652778。
* 同一 12 条样本的候选池，强制 ``max_candidates=1`` 后贪心选出 ``[1.5, 2.5]``，
  增益 0.11574074；枚举可找到 ``[6.5, 8.5]``，增益 0.13690476。
* 类别特征 5 个类别、最多 3 组，穷举 40 种分区得到增益 0.12256061，当前逐步合并为 0.12186546。
* XOR 数据上自动树因单层增益为 0 停在根节点，准确率 50%；手工指定第一层后继续 grow，
  可得到两层、准确率 100% 的树，证明局部贪心不保证整树最优。

自动停止存在一个已复现的问题：``_grow`` 只对排名第一的候选检查 ``_allowed``。
在 ``criterion="chi2"``、``min_impurity_decrease=0.0974`` 的 100 条样本中，
候选 A 调整后 LogWorth 更高，但 Gini 增益为 0.0968；候选 B 增益为 0.098 且通过显著性门槛。
当前实现因 A 不合格而停止，没有继续选择合格的 B。后续应先在搜索层筛出满足全部自动生长约束的方案，
再比较评分，而不只在每个特征最佳候选选出后过滤。

``manual_split(node=...)`` 和指定 feature 但不传 thresholds 时复用同一候选搜索，
但手工操作不要求通过 alpha / min_impurity_decrease。明确给出 thresholds 或 category_groups 时
直接安装指定分裂，不做最优性搜索；``grow=True`` 仅继续生长后代，不回头优化指定切点。
训练评分使用目标标签与订单计数；规则筛选和报告金额口径不参与建树目标。

后续优化可优先评估复用 ``core/binning/_partition_solver.py`` 的连续分区求解能力，
保留 OPTIMAL / FEASIBLE 求解状态，并适配树的右闭区间、缺失路由和多分类。
候选池最优、全边界单节点最优和整树最优仍需分别说明。

API
---

完整参数和方法见 :class:`hscredit.report.mining.MultiwayTreeExtractor`。

长表、并列表和特征分箱
----------------------------------------

``del_grey`` 沿用逐标签剔灰规则，订单与金额按相同的有效样本计算。
主报告的统计与规则表采用订单维度在左、金额维度在右的布局，中间空一列；
数据集为总标题，附简要口径说明。``long_format=False`` 可切换为宽表。
样本分布与数据样本统计共用指标构造，区别仅为按日期和业务字段分组后重新计算；两者固定使用多标签宽表，long_format 只控制规则表。

额外的“报告名_特征分箱”页复用 ``swap_out_report`` 使用的 ``feature_binning_summary``。
只分析最终规则引用的特征，顶部是效果 summary，各数据集的明细左右排列；
单个特征的多方法、多标签分箱明细始终为长表。默认方法为等频、等距、决策树，
可通过 ``methods`` 和 ``bin_params`` 配置；各数据集独立拟合分箱以比较方法效果，
不修改最终树。summary 使用工作表筛选，各数据集明细使用原生 Excel Table 筛选按钮。


特征效果汇总使用多标签宽表；KS 和 AUC 分别直接调用 ``hscredit.core.metrics.ks`` / ``auc``，
基于原始数值特征和对应标签的有效样本，金额维度传入金额权重。AUC 使用统一入口的自动方向识别；
类别特征、无有效样本或无有效正负样本时，不伪造 AUC。单类 KS 沿用统一入口的 0。
原始特征的 KS/AUC 在不同分箱方法行重复，分箱明细仍保留原有分档 KS。
宽表汇总保留分层表头及工作表筛选，明细长表使用原生 Table 筛选。

所有表格统一设置数值格式：比例与 KS 为 ``0.00%``，AUC/IV/WOE/LIFT 等为 ``0.0000``，
计数为 ``#,##0``，金额为 ``#,##0.00``。金额口径中沿用“样本数”命名的权重列同样显示两位小数。
显示格式不舍入中间值、不改变单元格数值；主报告保留模板列宽，特征分箱页自适应列宽且最大为 25。

报告速度与并行执行
----------------------------------------

手工二叉树、多叉树和 DecisionTreeAnalyzer 共用报告计算流程：
样本分布仅用 ``Rule.report`` 计算每组总体统计；每个数据集的最终树预测在本次报告内复用，
各标签、订单/金额与业务分组的 AUC/KS 使用 ModelReport 的同一指标 worker。
预测按行位置切片，支持重复索引；每次调用 report 都重新计算预测，不跨报告缓存模型状态。

数据集评估和特征分箱复用 hscredit 的 joblib 调度，结果按输入顺序汇总，Excel 写入及绘图串行执行。
配置沿用模型初始化时的 ``n_jobs``、``parallel_backend``、``parallel_config``：

.. code-block:: python

   from hscredit.report.mining import ManualTreeExtractor, MultiwayTreeExtractor

   tree = MultiwayTreeExtractor(
       target="FPD", features=features, max_n_bins=4,
       n_jobs=2, parallel_backend="loky",  # joblib 多进程
   ).fit(df_train)
   tree.report(datasets, overdue=["MOB1"], dpds=[7, 3, 0],
               amount="放款金额", date="放款时间", save="多叉树策略挖掘.xlsx")
   # ManualTreeExtractor 使用相同的并行参数。
   # parallel_backend="threading" 使用多线程；n_jobs=1 为串行。

``n_jobs=-1`` 按现有工作量和资源预算自适应调度；显式后端及 worker 上限优先。
外层数据集并行时节点内部串行，避免叠加启动多层 worker。
小数据集上，线程调度及进程首次启动可能抵消并行收益；应按实际数据规模选择。

2026-10-31，Python 3.11.5 验收环境，使用真实数据的 582/194/194 三数据集、
MOB1 的 7/3/0 三标签、剔灰、金额、日期加商品类别分组，以及 quantile/uniform/tree 三种分箱。
下表为绘图和 Excel 模板热身后的单次端到端计时，包含生成并保存完整工作簿：

.. list-table:: 报告生成耗时（秒）
   :header-rows: 1

   * - 模型
     - 优化前串行
     - 优化后串行
     - 2 线程
     - 2 进程
   * - 手工二叉树（11 节点）
     - 32.58
     - 12.45
     - 12.94
     - 13.12
   * - 多叉树（9 节点）
     - 25.90
     - 9.92
     - 10.38
     - 8.79

多进程测量中二叉树先执行，包含进程池首次启动；随后多叉树复用进程池，因此不能忽略启动成本比较模型。
优化前后返回表逐值完全一致。两份工作簿共 47,335 个非空单元格，
工作表 XML、数值格式、合并、列宽、筛选、树图内容保持一致；并行版本只有参数配置与文件创建时间不同。
对应回归入口为 ``tests/test_report/test_tree_report_parallel.py`` 与原有树挖掘报告测试。
