分箱 ``hscredit.core.binning``
==============================

提供 17 种分箱算法与二维交互分箱，全部继承统一基类 ``BaseBinning``，并由工厂类
``OptimalBinning`` 作为统一入口。所有分箱指标统一通过 ``core.metrics`` 计算。

.. rubric:: 数值变量的箱数与样本约束

``max_n_bins`` 表示普通箱（不含 missing/special）的最大箱数，``min_n_bins`` 表示
最小箱数，唯一值不足时以可区分的取值数为上限。大小、坏率和单调约束无法同时满足时，
默认 ``constraint_policy="raise"`` 会抛出中文异常。选择 ``constraint_policy="legacy"`` 时，
无法同时满足约束则保留原生算法的尽力满足结果；没有可行分区的优化器可回退单箱。
实际未满足项保存在 ``constraint_violations_`` 中，不增加业务统计表列。
MDLP 显式设置 ``force_min_bins=False`` 时允许
由停止准则决定更少的箱数。

``min_bin_size`` 默认是 ``0.01``。取值 ``0 < value < 1`` 时按本次有效标签训练样本总数占比解释，
取值 ``value >= 1`` 时按绝对样本数解释；显式传入 ``None`` 表示关闭最小样本数限制。
缺失值和特殊值不属于普通箱，但计入比例的样本基数。最少样本数向上取整，最多样本数向下取整。

``uniform`` 会先按 ``max_n_bins`` 生成完整等距切点，直接类与统一入口都会应用
``min_bin_size`` 等通用约束并合并相邻箱，因此约束收口后的区间允许宽度不等。
当 ``min_bin_size=None`` 时保留原始等距切点，也允许规则空间中存在训练样本数为 0 的箱。

``quantile`` 按分位数生成等频边界；多个分位点落在相同重复值上时会像
``qcut(duplicates='drop')`` 一样合并重复边界，所以实际箱数会正常减少且不会人为拆分相同值。
自定义 ``quantiles`` 决定初始分位点，最终仍需满足公共硬约束。
需要完整保留手工边界时使用 ``user_splits`` 和 ``user_splits_fixed=True``。

``max_bin_size``、``min_bad_rate`` 和 ``monotonic`` 在最终规则上统一验收；
完全固定的用户规则和已导入规则保持原有优先级。约束策略适用于全部一维方法及二维分箱；
二维缺失值语义组始终保留，不会为满足箱数而丢弃样本。
无效参数、非法输入、用户评价函数错误、求解超时不被 ``legacy`` 静默吞掉。
显式 ``target_bad_rates`` 的 ``strict_mode=True`` 分档要求仍然保持严格。
默认策略下约束不可满足时会抛出中文异常；
发生在特征任务内部的异常保留 ``ParallelExecutionError`` 包装及原始原因。

.. rubric:: 算法目标、精度与预分箱

``OptimalBinning`` 默认不再执行额外的 Lift 调整；只有显式设置 ``lift_refine=True`` 才启用。
该选项和各方法公共参数会被 ``sklearn.clone`` 完整保留。实际切点保存完整精度，``decimal``
只控制标签显示；标签不足以区分相邻边界时会提高显示精度。导入规则也不会因显示精度被舍入。

搜索、训练统计和 ``transform`` 均使用左闭右开的 ``[a,b)`` 边界。预分箱先将数据转成
不可拆分的原子箱，再由所选方法拟合；结果只能包含原始预分箱候选边界，不做最近点映射。

``cp_sat`` 和 ``or_tools(use_cp_sat=True)`` 的内置目标使用真实 CP-SAT 区间路径模型。
``or_tools`` 默认使用有时限的动态规划/枚举，不附加隐式 Lift 或形状奖励。
``solver_info_`` 按字段记录后端、目标和求解状态；CP 模型同时记录整数目标及上界。
CP 目标按 1e9 比例整数化，``OPTIMAL`` 表示该候选模型的最优解；``FEASIBLE`` 表示尚未证明最优。
这类元信息不加入业务指标表。

.. rubric:: 按名称或函数配置评价目标

本次接口扩展前，只有 ``ORBinning`` 接受任意自定义回调；``CPSATBinning`` 和
``GeneticBinning`` 只支持固定原生目标。现在这三种方法均支持统一的 ``objective`` 参数：

.. list-table:: 评价目标支持范围
   :header-rows: 1

   * - 方法
     - 原生目标
     - 预定义复合指标 / 自定义函数
   * - ``or_tools`` / ``ORBinning``
     - ``iv``、``ks``、``gini``、``entropy``、``chi2``
     - 支持
   * - ``cp_sat`` / ``CPSATBinning``
     - ``iv``、``ks``、``gini``
     - 支持；使用有时限的候选枚举，后端如实记为 ``enumeration``
   * - ``genetic`` / ``GeneticBinning``
     - ``iv``、``ks``、``gini``
     - 支持；作为遗传适应度，不保证全局最优
   * - 其他一维方法
     - 使用自身算法准则；``best_lift`` 保留 ``optimization`` 专用选项
     - 不接受任意 ``objective``，显式传入会报错
   * - ``OptimalBinning2D``
     - 二维区域合并仍使用原生准则
     - 可通过 ``x_params`` / ``y_params`` 为支持的轴方法传入目标

``CustomObjectives.names()`` 返回以下八个复合指标：``max_lift_iv``、``min_lift_iv``、
``max_min_lift_sum_iv``、``max_lift_distance_iv``、``min_lift_distance_iv``、
``lift_distance_sum_iv``、``max_ks_iv``、``woe_variance_iv``。
``OptimalBinning.available_objectives(method)`` 可查看某个方法接受的名称。

.. code-block:: python

   from hscredit.core.binning import OptimalBinning, CustomObjectives

   binner = OptimalBinning(
       method="or_tools",
       objective="max_lift_iv",
       objective_params={"iv_weight": 1.0, "lift_weight": 0.2},
       constraint_policy="legacy",  # 也可选 "raise"，默认是 "raise"
       max_n_bins=5,
   )
   binner.fit(X_train, y_train)

   def my_objective(bin_stats, total_good, total_bad, penalty=0.0):
       return sum(row["分档IV值"] for row in bin_stats) - penalty * len(bin_stats)

   custom = OptimalBinning(
       method="genetic",
       objective=my_objective,
       objective_params={"penalty": 0.01},
       random_state=42,
   ).fit(X_train, y_train)

   # 也可直接复用已有工厂函数生成的回调。
   callback = CustomObjectives.max_ks_iv(ks_weight=0.3)
   selected = OptimalBinning(method="cp_sat", objective=callback).fit(X_train, y_train)

``objective_params`` 为可选字典：名称模式传给对应工厂配置权重，函数模式作为额外关键字参数
传给回调。原生目标不接收额外权重。旧写法
``objective="custom", custom_objective=函数`` 保持可用；不能同时指定另一个目标和
``custom_objective``，避免静默忽略参数。clone、set_params、持久化均保留配置。

回调接收普通搜索样本的 ``bin_stats``、总体好样本数和坏样本数。``bin_stats`` 为按箱号排列的
字典列表，包含未舍入的中文统计列，并保留英文键 ``count/good/bad/bad_rate/good_rate/lift/woe``；
新增 ``iv/good_dist/bad_dist`` 对应分档 IV 与总体好坏分布。``bad_rate/good_rate`` 是箱内率，
不是总体好坏分布。``woe`` 统一为 ``log(坏分布/好分布)``，IV 应直接读取 ``iv`` / ``分档IV值``。
预定义复合目标已修正此前的 IV 条件坏率/WOE 符号混用，统计复用 ``compute_bin_stats`` 内核。

评价返回有限实数，统一取最大值；需要最小化损失时返回其相反数。NaN、无穷、布尔值、数组和
回调异常会明确报错。缺失/特殊箱不参与普通区间的目标搜索；固定用户规则不会为了目标而移动。
OR/CP 的自定义目标用一次排序建立前缀计数，后续候选只聚合箱计数，避免重复扫描全部训练行。
原生 ``gini`` 的含义依方法而异：OR-Tools/CP-SAT 最大化
``1 - sum(箱占比 * 坏率 * 好率)`` 的箱纯度评分；GeneticBinning 使用箱内坏率排序的
``2 * AUC - 1``。需要跨方法完全一致的评价口径时，应传入同一个回调或复合目标。

完整真实数据示例见 ``examples/30_binning_objectives_and_constraints.ipynb``。

``target_bad_rate`` 的显式边界作用于最终区间坏率。严格模式每个占用档位最多对应一个箱；
``merge_empty_bins=False`` 还要求每个档位都有箱，不可行时明确报错。宽松模式优先覆盖更多目标档位。
``smooth`` 的 ``merge_criterion`` 区分 IV、卡方和联合准则；统一入口通过 ``smoothing_method``
选择直接类的平滑方法。KDE 的 ``iv_weight`` 用于候选谷的 IV/谷深权衡。
``bandwidth='adaptive_reference'`` 是默认自适应参考带宽；旧 ``'isj'`` 仅保留为带弃用提示的兼容别名，
不宣称实现完整 ISJ 算法。``best_lift.min_lift`` 的非零过滤值会被拒绝，应在分箱表或规则分析中筛选箱。

.. rubric:: 类别变量分箱

类别变量会先转换为有序编码，再交给所选 ``method`` 的原生数值分箱算法决定边界。
因此等宽、等频、树、卡方、KS、IV、MDLP 等方法仍分别使用自己的切分和合并标准，
不会在 ``OptimalBinning`` 中被替换为同一种类别合并结果。

默认顺序按训练集坏样本率升序生成；坏样本率相同的类别保持首次出现顺序。也可以通过
``category_order`` 为指定字段提供完整顺序。下面使用 ``hscredit_hsk.xlsx`` 的 ``工资``
字段主动指定顺序：

.. code-block:: python

   import pandas as pd
   from hscredit.core.binning import OptimalBinning

   df = pd.read_excel("examples/hscredit_hsk.xlsx")
   X = df.drop(columns="target")
   y = df["target"]
   wage_order = X["工资"].dropna().drop_duplicates().tolist()

   binner = OptimalBinning(
       method="best_iv",
       max_n_bins=5,
       cat_cutoff=10,
       category_order={"工资": wage_order},
   ).fit(X, y)

   assert binner._category_orders_["工资"] == wage_order

``category_order`` 也可传入 ``callable(feature, x, y)``。函数应返回当前字段的完整非缺失
类别序列；缺少类别、重复类别或包含训练集以外的类别都会抛出中文 ``ValueError``。
数值编码的状态字段不会默认当作类别变量，需要通过 ``cat_cutoff`` 显式启用。例如
``cat_cutoff=10`` 表示唯一值数量不超过 10 的数值字段按类别处理。

.. rubric:: 自定义类别分箱

自定义类别分箱使用 ``List[List]``，必须完整覆盖训练类别，且类别和缺失标记不能重复。
缺失值既可以单独成箱，也可以与普通类别放在同一箱：

.. code-block:: python

   import numpy as np

   missing_alone = {
       "工资": [wage_order[:4], wage_order[4:8], wage_order[8:], [np.nan]],
   }
   missing_mixed = {
       "工资": [wage_order[:4], wage_order[4:8], [*wage_order[8:], np.nan]],
   }

   strict = OptimalBinning(
       user_splits=missing_alone,
       user_splits_fixed=True,
   ).fit(X[["工资"]], y)

   mixed = OptimalBinning(
       user_splits=missing_mixed,
       user_splits_fixed=True,
       missing_separate=False,
   ).fit(X[["工资"]], y)

``user_splits_fixed=True`` 会完整保留用户分组；非严格模式把每个用户组视为不可拆分的
原子单位，再用当前 ``method`` 决定是否合并相邻组。未知预测期类别默认转换为索引 ``-3``、
标签 ``unknown`` 和中性 WOE ``0.0``。``handle_unknown='value'`` 与默认 ``-3`` 等价；
``handle_unknown='raise'`` 会在 transform 遇到训练期未知类别时直接报错；也可指定任意已记录的整数箱号。

``max_n_bins``、``min_bin_size``、``max_bin_size``、``min_bad_rate`` 和明确的单调方向同样
适用于类别路径。如果单个类别或用户原子组本身已使约束不可满足，分箱器会指出字段、参数和
实际值，而不是静默忽略限制。

.. automodule:: hscredit.core.binning
   :members:
   :imported-members:
   :show-inheritance:
