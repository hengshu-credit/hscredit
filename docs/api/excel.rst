Excel ``hscredit.excel``
========================

``ExcelWriter`` 上下文管理器与 ``dataframe2excel`` 等工具，支持写入数据表、插入图片/
超链接、条件格式、单元格样式、数字格式、冻结窗格、行高/列宽调整与模板化输出。

.. rubric:: 单元格内容与样式

``insert_value2sheet`` 保留原有位置参数，新增样式选项均为关键字参数。可控制水平/垂直对齐、
换行、缩进、加粗、斜体、删除线、字体、字号、文本颜色、实色背景、显示格式、尺寸和批注：

.. code-block:: python

   from hscredit.excel import ExcelWriter

   writer = ExcelWriter()
   worksheet = writer.get_sheet_by_name("报告")
   writer.insert_value2sheet(
       worksheet, "B2", "第一行说明\n第二行说明",
       horizontal="left", vertical="top", wrap_text=True, indent=1,
       bold=True, italic=False, strike=False, font_name="宋体", font_size=12,
       font_color="#2639E9", background_color="#EEF0FF",
       column_width=28, auto_height=True,
       comment="指标口径说明", comment_author="logo hscredit",
   )
   writer.insert_value2sheet(
       worksheet, "C2", 0.123456789, decimal=None, number_format="0.00%",
       horizontal="right", vertical="center", row_height=30,
   )
   writer.save("报告.xlsx")

水平对齐常用 ``left/center/right``，垂直对齐常用 ``top/center/bottom``。
``wrap_text=True`` 开启内容换行，``False`` 关闭，``None`` 保留基础样式。
``bold``、``italic``、``strike`` 也用 ``None`` 表示保留，``False`` 可显式取消已有字体效果。
``indent`` 为 0–255 的缩进级别，传 0 取消缩进。

``style`` 支持样式名称或 ``openpyxl.styles.NamedStyle``；传 ``None`` 可保留单元格已有样式。
覆盖顺序为基础样式、``align``、显式属性参数。``align`` 可传字典（仅覆盖指定属性）或
``Alignment`` 对象（替换完整对齐设置），不会因只调整水平对齐而清除换行、缩进等属性。

``column_width``、``row_height`` 优先于 ``auto_width``、``auto_height``，合并区域中的每列/
每行均设置为指定尺寸。自动行高在列宽确定后计算，支持普通单格及横向合并；跨行合并不自动
重分配行高。默认最大自动行高为 120 磅，可用 ``max_row_height`` 调整。

``number_format`` 采用 Excel 格式字符串，不负责舍入数值；需保留完整精度时传 ``decimal=None``。
数字样字符串（如 ``"00123"``）始终保留文本类型和 ``@`` 格式，使用 ``writer.save()`` 保存时
只关闭“数字存为文本”的错误提示，不关闭其他错误检查，也不会为了消除提示而丢失前导零。

``comment`` 可传批注文本或 ``openpyxl.comments.Comment``；文本批注默认作者为
``logo hscredit``，也可用 ``comment_author`` 指定。传 ``None`` 保留已有批注，传空字符串
删除批注；传 ``Comment`` 对象时保留其作者、宽度和高度。这里生成的是普通文本批注，不支持
账号头像或在线会话式评论界面；详见 `openpyxl 批注文档 <https://openpyxl.readthedocs.io/en/stable/comments.html>`_。

.. rubric:: 行高与列宽

``set_row_height`` 设置单行固定高度，``adjust_rows_height`` 与 ``adjust_columns_width``
对应，支持单行、行列表、连续行范围或自动检测。``min_width`` 和 ``min_height`` 默认均为
``None``，不设置最小尺寸，只有显式传入时才限制。行高和上下留白均以磅为单位，默认最大行高
为 120 磅，额外留白为 2 磅。

.. code-block:: python

   from hscredit.excel import ExcelWriter
   from openpyxl.styles import Alignment

   writer = ExcelWriter()
   worksheet = writer.get_sheet_by_name("报告")
   writer.insert_value2sheet(worksheet, "B2", "第一行说明\n第二行说明")
   worksheet["B2"].alignment = Alignment(wrap_text=True)

   writer.adjust_columns_width(worksheet, max_width=25)
   writer.adjust_rows_height(worksheet)  # 自动检测有内容的行
   writer.adjust_columns_width(worksheet, min_width=8)  # 显式设置最小列宽
   writer.adjust_rows_height(worksheet, min_height=15)  # 显式设置最小行高
   writer.adjust_rows_height(worksheet, rows=[2, 4])
   writer.adjust_rows_height(worksheet, start_row=5, end_row=20, max_height=90)
   writer.set_row_height(worksheet, row=1, height=30)
   writer.save("报告.xlsx")

自动行高依据字号、显式换行以及已有的 ``wrap_text`` 设置估算，不修改单元格的换行设置。
开启自动换行时，建议先确定列宽，再计算行高。横向合并单元格使用合并后的总宽度；跨行合并
内容不参与单行计高。自动检测会保留空行和仅有跨行合并内容的行；显式选中的空行只有在设置了
``min_height`` 时才设为最小行高。未设置最小尺寸时，无可计量内容的行、列保留原尺寸。
该方法保留已有样式及行的隐藏、分组等属性，但不执行 Excel 的字体排版或公式计算。

``adjust_columns_width`` 先按各列的独立内容计宽，再检查合并单元格：合并区域总宽度已
足够时不再扩宽；不足部分均摊到本次选中的列。未选中的列保留原宽度，参与调整的列仍受
``max_width`` 限制；达到上限后仍过长的内容可通过换行配合行高调整展示。

.. rubric:: 按层级合并分组列

``insert_df2sheet`` 与 ``dataframe2excel`` 的 ``merge_hierarchy`` 默认是 ``True``。
启用 ``merge=True`` 后，按 ``merge_column`` 的传入顺序从父层到子层合并连续相同值。
子层必须同时满足所有父层的分组边界，训练集末尾和验证集开头即使都是 ``2026-02``，
也不会跨数据集合并。表格的物理列顺序和原始行顺序不变。

.. code-block:: python

   from hscredit.excel import dataframe2excel

   # 默认：数据集 -> 数据分组，月份不跨数据集合并。
   dataframe2excel(
       df, writer, sheet_name="层级合并",
       merge_column=["数据集", "数据分组"], merge=True,
   )

   # 恢复各列独立合并，允许相同月份跨数据集边界合并。
   dataframe2excel(
       df, writer, sheet_name="独立合并",
       merge_column=["数据集", "数据分组"], merge=True, merge_hierarchy=False,
   )

   # 同一个参数也可以直接指定从父到子的层级列表。
   dataframe2excel(
       df, writer, sheet_name="指定层级",
       merge=True, merge_hierarchy=["数据集", "数据分组"],
   )

同时传入 ``merge_column`` 和层级列表时，列表必须包含所有待合并字段；额外的父层字段只用于
限定合并边界，本身不合并。多层列名可使用完整元组键，也可使用末层列名。普通与快速写入使用
相同的层级规则，``merge=False`` 时仍不合并数据单元格。

使用交替填充色时，可显式开启 ``merge_border=True``，用主题色上下边框区分相邻分组：

.. code-block:: python

   dataframe2excel(
       df, writer, sheet_name="分组边框", fill=True, merge=True,
       merge_column=["数据集", "数据分组"], merge_border=True,
   )

``merge_border`` 默认 ``False``，兼容层级合并与 ``merge_hierarchy=False`` 的独立合并。
边框只应用于待合并数据列的分组（包括单行分组），各列按自身的分组边界绘制，不横穿其他列的
父层分组。内部分隔线为主题色细线，数据首行上边框及末行下边框为主题色加粗线。
``fill=False`` 或 ``merge=False`` 时，此开关不改变现有边框。

.. rubric:: 大数据集保样式快速写入

``dataframe2excel`` 的 ``speed`` 默认是 ``"auto"``，会根据 DataFrame 的大小和维数
自动选择写入路径。满足以下任一条件时使用快速路径：数据行数不少于 500、有效列数
（包含写出的索引层级）不少于 50，或有效单元格数不少于 10000。其余情况使用普通路径。

无论选择哪条路径，都会保留标题、表头、内容填充、边框、合并、数字格式、条件格式、
图片、筛选和自动列宽，不会生成无样式 Excel。通常直接使用默认设置即可：

.. code-block:: python

   from hscredit.excel import dataframe2excel

   dataframe2excel(
       df,
       "特征明细.xlsx",
       sheet_name="特征明细",
       auto_width=True,
   )

需要固定写入路径时，可显式指定 ``speed="normal"`` 或 ``speed="fast"``，显式设置会
覆盖自动判断：

.. code-block:: python

   dataframe2excel(df, "兼容模式.xlsx", speed="normal")
   dataframe2excel(df, "快速模式.xlsx", speed="fast")

浮点值精度使用与 ``ScoreCard`` 一致的 ``decimal`` 参数。默认 ``decimal=4`` 保持历史
行为；传入 ``decimal=None`` 时不主动舍入 DataFrame 浮点值：

.. code-block:: python

   dataframe2excel(df, "原始精度.xlsx", speed="fast", decimal=None)

对于包含多个工作表或多张表格的报告，应复用同一个 ``ExcelWriter``，全部写入完成后只
保存一次。``dataframe2excel`` 的 ``auto_width``、``auto_height`` 默认均为 ``False``。
开启后，会在本次调用的标题、数据、合并、数字格式、对齐和筛选全部完成后各批量处理一次；
写入期间不计算或调整尺寸，普通和快速模式使用相同的最终调整步骤。
同时启用时先调整列宽，再按最终列宽计算行高。尺寸调整只更新行/列维度，保留单元格边框、
合并区域外沿、填充、字体及数字格式。

.. code-block:: python

   dataframe2excel(
       df, writer, sheet_name="报告", percent_cols=["占比"],
       auto_width=True, auto_height=True,
   )

如果同一个工作表需要多次插入表格，并希望整张表只调整一次，写入时保持这两个开关关闭，
完成所有插入后显式调用：

.. code-block:: python

   writer.adjust_columns_width(worksheet)
   writer.adjust_rows_height(worksheet)
   writer.save("报告.xlsx")

.. rubric:: 指标条件格式

分箱和规则表优先使用 ``color_cols`` 色阶：低值接近白色，高值接近指标色，常量列不整列染色。
样本占比、金额占比与通过率使用黄绿 ``#BADC58``，规则命中率和坏样本率使用正样本强调色珊瑚红 ``#F76E6C``，
Lift 使用莓红 ``#D4145A``。合计行不参与样本占比的色阶缩放。
坏账改善和累积坏账改善保留居中双向数据条，负值向左使用浅蓝 ``#A3ABF5``，正值向右使用珊瑚红。
零仅显示轴线；同一列仅有正值或仅有负值时，中心仍为零。

.. code-block:: python

   from hscredit.excel import dataframe2excel

   table.show()
   dataframe2excel(table, "规则效果.xlsx", sheet_name="效果",
                   color_cols=["样本占比", "坏样本率", "LIFT值"],
                   centered_bar_cols=["坏账改善"])

两种入口均支持通过 ``condition_color`` 按指标覆盖颜色。需要普通数据条时，显式使用 ``condition_cols``。
``table.show(color_cols=[])`` 可关闭自动高亮。双向数据条使用颜色列表的首色、末色作为负值、正值颜色。
单个区域可用 ``writer.add_conditional_formatting(ws, "B2", "B20", bar_type="centered")``。

.. automodule:: hscredit.excel
   :members:
   :imported-members:
   :show-inheritance:
