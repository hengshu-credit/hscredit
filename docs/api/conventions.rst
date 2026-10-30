接口约定
========

API 页面和 ``help(对象)`` 使用源码中同一份中文 docstring。参数名称、类型、默认值、
枚举值和返回结构应以具体接口为准；相同参数名在不同组件中可能具有不同取值。

参数说明
------------

类文档采用“**参数** → **属性** → **参考样例**”结构，算法背景放在“**引用**”中。
函数和方法使用 ``:param 参数名:``、``:type 参数名:``、``:return:`` 和
``:raises 异常类型:`` 说明输入、输出和异常。属性部分只说明拟合后状态及公开结果，
构造配置写在参数部分。

- ``None`` 的含义逐参数定义，不能统一理解成“关闭”。例如分箱器 ``cat_cutoff=None``
  表示不把数值字段自动识别为类别；公共调度器 ``n_jobs=None`` 则表示串行。
- ``fit(X, y)`` 中的显式标签优先；支持 ``target`` 的组件也可从 ``fit(df)`` 提取目标列。
  预测期应只传特征，复用训练好的分箱器、编码器和模型。
- 支持 ``overdue+dpds`` 的分析与报告入口统一接受 ``target`` 列名、一维 Series/ndarray/list、
  callable 或标签字典，两种标签模式互斥。标签方向、逐标签剔灰及索引对齐见 :doc:`../overdue_labels`。
- 参数中的比例、绝对数量和利率按各接口说明使用。金融函数的利率是每期利率，
  例如月度还款使用 ``年利率 / 12``，现金流流入和流出使用相反符号。
- ``**kwargs`` 不是任意配置入口；应按方法说明传给对应下游组件，优先使用有明确说明的命名参数。

枚举取值
------------

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 接口与参数
     - 取值
     - 含义
   * - 分箱器 ``transform(metric=...)``
     - ``'indices'``、``'bins'``、``'woe'``
     - 箱索引 / 区间或类别标签 / WOE；默认 ``'indices'``。
   * - 分箱器 ``handle_unknown``
     - ``-3``、``'value'``、``'raise'`` 或已记录整数箱号
     - 默认未知箱为 ``-3``；``'value'`` 等价于默认；``'raise'`` 遇未知类别报错。
   * - 编码器 ``handle_unknown``、``handle_missing``
     - ``'value'``、``'error'``、``'return_nan'``
     - 子类默认编码值 / 报错 / 缺失值；编码值的含义见各编码器。
   * - AUC、Gini 的 ``score_direction``
     - ``'auto'``、``'higher_risk'``、``'higher_safe'``
     - 自动判断 / 高分高风险 / 高分低风险。评估已知坏样本概率时显式使用 ``'higher_risk'``。
   * - 逾期分析 ``overdue_operator``
     - ``'>'``、``'>='``、``'<'``、``'<='``
     - 与 DPD 阈值比较，满足条件记为坏样本；通常默认 ``'>'``，``distribution_plot`` 和 ``bin_overdues_plot`` 默认 ``'>='``。
   * - 分组报告 ``group_order``
     - ``'asc'``、``'desc'``、``'appearance'``，也支持排序函数或显式序列
     - 升序 / 降序 / 首次出现顺序；默认 ``'asc'``。
   * - 筛选报告 ``to_dataframe(kind=...)``
     - ``'summary'``、``'details'``
     - 阶段摘要 / 逐特征明细；默认 ``'summary'``。
   * - ``auto_feature_analysis(mode=...)``
     - ``None``、``'strict'``、``'best_effort'``
     - 传统异常行为 / 必需章节失败时拒绝发布 / 记录章节状态。

分箱方法由 ``OptimalBinning.VALID_METHODS`` 给出。当前 17 个值是
``'uniform'``、``'quantile'``、``'tree'``、``'chi'``、``'best_ks'``、``'best_iv'``、``'mdlp'``、
``'or_tools'``、``'cp_sat'``、``'cart'``、``'kmeans'``、``'monotonic'``、``'genetic'``、``'smooth'``、
``'kernel_density'``、``'best_lift'``、``'target_bad_rate'``。``'optimal'`` 不是方法值。
每种方法的附加参数和依赖见 :doc:`binning`。

公共调度参数的全部值、边界和嵌套预算见 :doc:`../parallelism`；
字段选择、日期频率和参数迁移见 :doc:`../analysis_parameters`。

返回示例
------------------

返回值是接口契约的一部分，不能仅按参数名推断：

- ``bin_plot(result=True)`` 返回 ``(图对象, 分箱统计表)``。
- ``feature_binning_summary`` 返回 ``(分箱表字典, 摘要表)``；分箱表按字段和方法组织。
- ``auto_feature_analysis`` 默认返回 ``(end_row, end_col)``；
  ``return_result=True`` 返回包含章节状态和表格的 ``ReportResult``。
- 返回结构化报告时，应同时查看章节状态，不能只检查是否存在输出文件。

:download:`下载按步骤运行的公开 API Notebook <../../examples/29_api_reference.ipynb>`。
它使用本地 ``examples/hscredit_yyp.xlsx``，涵盖数值与类别特征、MOB1 多标签、金额、
日期、建模评估、绘图及 Excel 导出。业务数据需由使用者自行准备，不随下载的 Notebook 提供。
示例输出默认位于项目的 ``outputs/api_reference``。

维护规范
--------------

参数枚举统一使用行内代码：字符串写作 ``'value'``，布尔值和空值写作
``True``、``False``、``None``。多个值用顿号分隔；需要逐项解释时使用列表，
列表前留空行，列表及续行缩进到所属的 ``:param:`` 字段内。
docstring 和 RST 页面使用双反引号，Markdown 页面使用单反引号；
两种源格式都应渲染成相同的行内代码样式。不要用斜体或普通引号代替行内代码。

API 叶子页面的目录固定为“模块 → 类或函数 → 方法”。介绍、样例和主题分组使用
``.. rubric:: 标题``，避免后续 API 被归入最后一个介绍小节，挤占类方法的目录层级。
普通指南与 API 分类索引仍使用章节标题和 ``toctree``。
模块页标题统一为“简短中文名称 + 模块路径”，例如 ``编码器 hscredit.core.encoders``。
API 分类按源码包目录组织：``models`` 位于 ``core`` 下，``mining`` 位于 ``report`` 下；
公共顶层包直接列在 API 参考中，不新增跨目录的功能分组。类与方法继续位于各自所属模块下。
教程和审查记录分别归入建模实践与项目记录，页面名保持简短，章节小标题保留在下级目录中。

新增或修改接口说明时，应对照函数签名和校验逻辑核对默认值、枚举与互斥条件。
示例应包含导入和数据准备，区分可直接运行的示例与依赖前文状态的片段；
可复用演示统一放在 Notebook 中。引用说明算法来源或接口背景，不代表第三方库与
HSCredit 的参数、枚举和计算结果完全一致。

- `scikit-learn 估计器开发指南 <https://scikit-learn.org/stable/developers/develop.html>`_：
  估计器与转换器的接口背景。
- `OptBinning 官方文档 <https://gnpalencia.org/optbinning/>`_：约束分箱的算法背景。
- `Category Encoders 官方文档 <https://contrib.scikit-learn.org/category_encoders/>`_：
  类别编码的接口与方法背景；HSCredit 的字段选择参数使用 ``features``。
