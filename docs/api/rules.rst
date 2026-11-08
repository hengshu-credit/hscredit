规则引擎 ``hscredit.core.rules``
================================

``Rule`` 支持与、或、非及嵌套条件。用 ``Rule.predict(df)`` 获取每行的命中结果，
用 ``Rule.report`` 计算命中指标。

.. rubric:: 空值与非空值

用 ``字段.isna()`` 匹配缺失，用 ``字段.notna()`` 匹配非缺失。两者适用于数值、字符串、
布尔、分类和 ``object`` 列，可以识别 ``None``、``np.nan``、``pd.NA`` 和 ``NaT``。
空字符串 ``''`` 属于非缺失值，业务上需要将它视为缺失时，可在规则中显式合并。

以下示例使用 ``from hscredit.core.rules import Rule`` 导入规则类。

.. list-table:: 常见命中条件
   :header-rows: 1
   :widths: 38 62

   * - 希望命中的样本
     - 规则写法
   * - 评分为空
     - ``Rule("评分.isna()")``
   * - 评分非空
     - ``Rule("评分.notna()")``
   * - 评分大于 600，排除缺失
     - ``Rule("评分 > 600")``
   * - 评分大于 600，或评分为空
     - ``Rule("(评分 > 600) | 评分.isna()")``
   * - 渠道属于线上或门店
     - ``Rule("渠道 in ['线上', '门店']")``
   * - 渠道非空，且属于线上、门店以外的渠道
     - ``Rule("渠道.notna() & (渠道 not in ['线上', '门店'])")``
   * - 评分和渠道都为空
     - ``Rule("评分.isna() & 渠道.isna()")``
   * - 评分和渠道至少一个为空
     - ``Rule("评分.isna() | 渠道.isna()")``
   * - 渠道为空或等于空字符串
     - ``Rule("渠道.isna() | (渠道 == '')")``
   * - 评分非空且不等于 100
     - ``Rule("评分.notna() & (评分 != 100)")``
   * - 评分不等于 100，或评分为空
     - ``Rule("(评分 != 100) | 评分.isna()")``

单写 ``评分 != 100`` 时，缺失行是否命中取决于列的数据类型。普通 ``float64`` 列的
``NaN != 100`` 为 ``True``，该行会命中。可空 ``Int64``、``Float64`` 列的
``pd.NA != 100`` 结果为 ``pd.NA``，``Rule.predict`` 将最终结果中的缺失值记为
未命中。表格最后两种写法可以在这些类型上得到一致的结果。

>>> import pandas as pd
>>> from hscredit.core.rules import Rule
>>> df = pd.DataFrame({"评分": [100.0, 99.0, float("nan")]})
>>> rule = Rule("评分 != 100")
>>> rule.predict(df).tolist()
[False, True, True]
>>> rule.predict(df.astype({"评分": "Int64"})).tolist()
[False, True, False]

``not in`` 也可能命中缺失值，因此“非空且不在名单中”要加上 ``.notna()``。
用 ``字段 != 字段`` 检查可空类型会漏掉 ``pd.NA``，应使用 ``字段.isna()``。

.. rubric:: API 参考

.. automodule:: hscredit.core.rules
   :members:
   :imported-members:
   :show-inheritance:
