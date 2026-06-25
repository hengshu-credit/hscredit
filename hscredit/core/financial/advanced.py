"""高级金融计算函数.

提供基于现金流序列的投资评价指标：净现值、内部收益率、修正内部收益率。
约定与 ``numpy_financial`` 一致：现金流序列 ``values[0]`` 位于第 0 期（当前时点，
不折现），``values[k]`` 位于第 k 期期末。

**引用**

- numpy-financial 文档：https://numpy.org/numpy-financial/latest/
- NPV/IRR/MIRR 概念参考：https://en.wikipedia.org/wiki/Net_present_value
"""

import numpy as np


def npv(rate, values):
    """计算净现值 (Net Present Value).

    将一段现金流序列按固定折现率折算到第 0 期并求和，用于判断项目是否创造价值
    （NPV > 0 表示在该折现率下项目可接受）。

    .. note::
        采用 ``numpy_financial`` 约定：``values[0]`` 位于第 0 期、**不折现**，
        与 Excel ``NPV``（从第 1 期开始折现所有值）相差一个 ``(1+rate)`` 因子。

    :param rate: 每期折现率，小数表示（如 0.05 表示每期 5%）
    :param values: 现金流序列（类数组），``values[0]`` 通常为初始投资（负值），
        其后为各期回报
    :return: 净现值，正值表示在该折现率下项目盈利

    **参考样例**

    初始投入 1000，其后四期回报 300/400/400/300，折现率 5%::

        >>> npv(0.05, [-1000, 300, 400, 400, 300])
        240.87185894766066

    **引用**

    对应 ``numpy_financial.npv``：
    https://numpy.org/numpy-financial/latest/functions/npv.html
    """
    values = np.asarray(values)
    rate = np.asarray(rate)

    # 计算现值
    n = np.arange(len(values))
    pv = values / (1 + rate) ** n

    return np.sum(pv, axis=0)


def irr(values):
    """计算内部收益率 (Internal Rate of Return).

    求解使净现值（NPV）恰好为零的每期折现率，即项目隐含的真实回报率。
    内部使用二分法（bisection）在 ``[-0.99, +∞)`` 区间搜索，要求现金流同时包含
    正值与负值（至少一次变号）以保证解存在。

    .. note::
        IRR 仅在现金流方向单次变号时唯一；多次变号可能存在多个解，本实现返回
        二分法在默认区间内找到的第一个根。

    :param values: 现金流序列（类数组），通常 ``values[0]`` 为初始投资（负值），
        且序列中至少各有一个正值与一个负值
    :return: 内部收益率（每期，小数表示）
    :raises ValueError: 现金流未同时包含正负值、无法确定搜索上界或迭代不收敛时抛出

    **参考样例**

    初始投入 1000，其后四期回报 300/400/400/300 的内部收益率::

        >>> irr([-1000, 300, 400, 400, 300])
        0.14895028127237311

    **引用**

    对应 ``numpy_financial.irr``：
    https://numpy.org/numpy-financial/latest/functions/irr.html
    """
    values = np.asarray(values)

    # 使用 numpy_financial 的实现思路
    # 解决 NPV = 0 的方程

    # 检查现金流符号变化
    signs = np.sign(values)
    if not ((signs > 0).any() and (signs < 0).any()):
        raise ValueError("现金流必须至少包含一个正值和一个负值")

    # 定义 NPV 函数
    def _npv(rate):
        if rate <= -1:
            return float('inf')
        return np.sum(values / (1 + rate) ** np.arange(len(values)))

    # 使用二分法寻找 IRR
    # 确定搜索范围
    low, high = -0.99, 1.0

    # 调整 high 直到 NPV 变号
    max_iter = 100
    for _ in range(max_iter):
        npv_high = _npv(high)
        if npv_high < 0:
            break
        high *= 2
        if high > 1e10:
            raise ValueError("无法找到合适的利率搜索上界")

    # 二分搜索
    tol = 1e-8
    for _ in range(max_iter):
        mid = (low + high) / 2
        npv_mid = _npv(mid)

        if abs(npv_mid) < tol:
            return mid

        if _npv(low) * npv_mid < 0:
            high = mid
        else:
            low = mid

    raise ValueError(f"迭代 {max_iter} 次后仍未收敛")


def mirr(values, finance_rate, reinvest_rate):
    """计算修正内部收益率 (Modified Internal Rate of Return).

    MIRR 修正了 IRR 隐含"按 IRR 自身再投资"的不现实假设：正现金流按再投资收益率
    复利至末期，负现金流按融资成本折现至初期，再由两者之比反解年化收益。相比 IRR
    更贴近真实资金成本，且对单变号以外的现金流也唯一。

    公式：``MIRR = (FV(正现金流) / -PV(负现金流)) ** (1/(n-1)) - 1``，其中 ``n``
    为现金流期数。

    :param values: 现金流序列（类数组），需同时包含正值与负值
    :param finance_rate: 融资成本率（每期），用于将负现金流折现到第 0 期
    :param reinvest_rate: 再投资收益率（每期），用于将正现金流复利到末期
    :return: 修正内部收益率（每期，小数表示）
    :raises ValueError: 现金流中不含负值或不含正值时抛出

    **参考样例**

    现金流 -1000/300/400/400/300，融资成本 5%、再投资收益 8%::

        >>> mirr([-1000, 300, 400, 400, 300], 0.05, 0.08)
        0.1205253227096672

    **引用**

    对应 Excel ``MIRR`` 与 ``numpy_financial.mirr``：
    https://numpy.org/numpy-financial/latest/functions/mirr.html
    """
    values = np.asarray(values)
    n = len(values)

    # 分离正负现金流
    negative = values < 0
    positive = values > 0

    # 计算负现金流的现值 (使用融资成本)
    pv_negative = np.sum(values[negative] / (1 + finance_rate) ** np.arange(n)[negative])

    # 计算正现金流的未来值 (使用再投资收益率)
    fv_positive = np.sum(values[positive] * (1 + reinvest_rate) ** (n - 1 - np.arange(n)[positive]))

    # MIRR 公式
    if pv_negative == 0:
        raise ValueError("现金流中不存在负值")
    if fv_positive == 0:
        raise ValueError("现金流中不存在正值")

    return (fv_positive / -pv_negative) ** (1 / (n - 1)) - 1
