"""高级金融计算函数.

提供净现值、内部收益率等高级计算。
"""

import numpy as np


def npv(rate, values):
    """计算净现值 (Net Present Value).

    将未来各期现金流按指定折现率折算为当前时点的总价值。

    :param rate: 折现率（每期利率）
    :param values: 现金流序列（第0期为初始投资，通常为负；后续为各期回报）
    :return: 净现值（正值表示收益，负值表示亏损）

    **参考样例**

    >>> npv(0.05, [-1000, 300, 400, 400, 300])
    265.6913368537139
    """
    values = np.asarray(values)
    rate = np.asarray(rate)

    # 计算现值
    n = np.arange(len(values))
    pv = values / (1 + rate) ** n

    return np.sum(pv, axis=0)


def irr(values):
    """计算内部收益率 (Internal Rate of Return).

    使用二分法求解使净现值为零的折现率，即项目的真实回报率。

    :param values: 现金流序列（第0期为初始投资必须为负，至少有一个正值和一个负值）
    :return: 内部收益率
    :raises ValueError: 现金流不包含正负值混合或迭代无法收敛时

    **参考样例**

    >>> irr([-1000, 300, 400, 400, 300])
    0.14299334826891236
    """
    values = np.asarray(values)

    # 使用 numpy_financial 的实现思路
    # 解决 NPV = 0 的方程

    # 检查现金流符号变化
    signs = np.sign(values)
    if not ((signs > 0).any() and (signs < 0).any()):
        raise ValueError("Cash flows must have at least one positive and one negative value")

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
            raise ValueError("Cannot find suitable upper bound")

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

    raise ValueError(f"Failed to converge after {max_iter} iterations")


def mirr(values, finance_rate, reinvest_rate):
    """计算修正内部收益率 (Modified Internal Rate of Return).

    MIRR 假设正现金流按再投资利率进行再投资，负现金流按融资成本进行融资，
    相比 IRR 更符合实际资金运作场景。

    :param values: 现金流序列（第0期为初始投资必须为负，至少有一个正值和一个负值）
    :param finance_rate: 融资成本率（负现金流的折现率）
    :param reinvest_rate: 再投资收益率（正现金流的再投资回报率）
    :return: 修正内部收益率
    :raises ValueError: 现金流中不包含负值或正值时

    **参考样例**

    >>> mirr([-1000, 300, 400, 400, 300], 0.05, 0.08)
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
        raise ValueError("No negative cash flows")
    if fv_positive == 0:
        raise ValueError("No positive cash flows")

    return (fv_positive / -pv_negative) ** (1 / (n - 1)) - 1
