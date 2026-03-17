"""基础金融计算函数.

提供现值、未来值、付款额等基础计算。
"""

import numpy as np


def _convert_when(when):
    """转换付款时间参数."""
    _when_to_num = {
        'end': 0, 'begin': 1,
        'e': 0, 'b': 1,
        0: 0, 1: 1,
        'beginning': 1,
        'start': 1,
        'finish': 0
    }

    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]


def fv(rate, nper, pmt, pv, when='end'):
    """计算未来值 (Future Value).

    :param rate: 每期利率
    :param nper: 总期数
    :param pmt: 每期付款额
    :param pv: 现值
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 未来值

    示例:
        >>> fv(0.05/12, 10*12, -100, -100)  # 每月存100，年利率5%，10年后
        15692.92889433575
    """
    when = _convert_when(when)
    rate = np.asarray(rate)
    nper = np.asarray(nper)
    pmt = np.asarray(pmt)
    pv = np.asarray(pv)
    when = np.asarray(when)

    if rate.ndim == 0:
        # 标量情况
        if rate == 0:
            return -(pv + pmt * nper)
        else:
            return -(pv * (1 + rate) ** nper +
                     pmt * (1 + rate * when) / rate *
                     ((1 + rate) ** nper - 1))

    # 数组情况
    result = np.zeros_like(rate)
    zero_rate = rate == 0
    result[zero_rate] = -(pv[zero_rate] + pmt[zero_rate] * nper[zero_rate])

    non_zero = ~zero_rate
    result[non_zero] = -(pv[non_zero] * (1 + rate[non_zero]) ** nper[non_zero] +
                         pmt[non_zero] * (1 + rate[non_zero] * when[non_zero]) /
                         rate[non_zero] *
                         ((1 + rate[non_zero]) ** nper[non_zero] - 1))

    return result


def pv(rate, nper, pmt, fv=0, when='end'):
    """计算现值 (Present Value).

    :param rate: 每期利率
    :param nper: 总期数
    :param pmt: 每期付款额
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 现值

    示例:
        >>> pv(0.05/12, 10*12, -100)  # 每月还款100，年利率5%，10年期的现值
        9428.135032823439
    """
    when = _convert_when(when)
    rate = np.asarray(rate)
    nper = np.asarray(nper)
    pmt = np.asarray(pmt)
    fv = np.asarray(fv)
    when = np.asarray(when)

    if rate.ndim == 0:
        if rate == 0:
            return -(fv + pmt * nper)
        else:
            return -(fv + pmt * (1 + rate * when) / rate *
                     ((1 + rate) ** nper - 1)) / (1 + rate) ** nper

    result = np.zeros_like(rate)
    zero_rate = rate == 0
    result[zero_rate] = -(fv[zero_rate] + pmt[zero_rate] * nper[zero_rate])

    non_zero = ~zero_rate
    result[non_zero] = (-(fv[non_zero] +
                          pmt[non_zero] * (1 + rate[non_zero] * when[non_zero]) /
                          rate[non_zero] *
                          ((1 + rate[non_zero]) ** nper[non_zero] - 1)) /
                        (1 + rate[non_zero]) ** nper[non_zero])

    return result


def pmt(rate, nper, pv, fv=0, when='end'):
    """计算每期付款额 (Payment).

    :param rate: 每期利率
    :param nper: 总期数
    :param pv: 现值
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 每期付款额

    示例:
        >>> pmt(0.05/12, 10*12, 10000)  # 贷款10000，年利率5%，10年，每月还款额
        -106.06557415332299
    """
    when = _convert_when(when)
    rate = np.asarray(rate)
    nper = np.asarray(nper)
    pv = np.asarray(pv)
    fv = np.asarray(fv)
    when = np.asarray(when)

    if rate.ndim == 0:
        if rate == 0:
            return -(fv + pv) / nper
        else:
            return -(fv + pv * (1 + rate) ** nper) * rate / \
                   ((1 + rate * when) * ((1 + rate) ** nper - 1))

    result = np.zeros_like(rate)
    zero_rate = rate == 0
    result[zero_rate] = -(fv[zero_rate] + pv[zero_rate]) / nper[zero_rate]

    non_zero = ~zero_rate
    result[non_zero] = (-(fv[non_zero] +
                          pv[non_zero] * (1 + rate[non_zero]) ** nper[non_zero]) *
                        rate[non_zero] /
                        ((1 + rate[non_zero] * when[non_zero]) *
                         ((1 + rate[non_zero]) ** nper[non_zero] - 1)))

    return result


def nper(rate, pmt, pv, fv=0, when='end'):
    """计算期数 (Number of Periods).

    :param rate: 每期利率
    :param pmt: 每期付款额
    :param pv: 现值
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 期数

    示例:
        >>> nper(0.05/12, -100, 10000)  # 每月还100，利率5%，还清10000贷款需要多少期
        129.62843690651015
    """
    when = _convert_when(when)
    rate = np.asarray(rate)
    pmt = np.asarray(pmt)
    pv = np.asarray(pv)
    fv = np.asarray(fv)
    when = np.asarray(when)

    if rate.ndim == 0:
        if rate == 0:
            return -(fv + pv) / pmt
        else:
            return (np.log((-fv * rate + pmt * (1 + rate * when)) /
                          (pv * rate + pmt * (1 + rate * when))) /
                    np.log(1 + rate))

    result = np.zeros_like(rate)
    zero_rate = rate == 0
    result[zero_rate] = -(fv[zero_rate] + pv[zero_rate]) / pmt[zero_rate]

    non_zero = ~zero_rate
    result[non_zero] = (np.log((-fv[non_zero] * rate[non_zero] +
                                pmt[non_zero] * (1 + rate[non_zero] * when[non_zero])) /
                               (pv[non_zero] * rate[non_zero] +
                                pmt[non_zero] * (1 + rate[non_zero] * when[non_zero]))) /
                        np.log(1 + rate[non_zero]))

    return result


def ipmt(rate, per, nper, pv, fv=0, when='end'):
    """计算给定期间的利息部分 (Interest Payment).

    :param rate: 每期利率
    :param per: 特定期间 (1 到 nper)
    :param nper: 总期数
    :param pv: 现值
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 利息部分
    """
    when = _convert_when(when)
    total_pmt = pmt(rate, nper, pv, fv, when)

    # 使用摊销公式计算利息
    if rate == 0:
        return 0

    # 计算截至 per-1 期的余额
    balance = fv if when == 1 else 0
    for i in range(1, int(per)):
        interest = rate * (balance if when == 1 else (balance + pv))
        principal = total_pmt - interest
        balance = balance + principal

    if when == 1:
        return rate * balance
    else:
        return rate * (balance + pv)


def ppmt(rate, per, nper, pv, fv=0, when='end'):
    """计算给定期间的本金部分 (Principal Payment).

    :param rate: 每期利率
    :param per: 特定期间 (1 到 nper)
    :param nper: 总期数
    :param pv: 现值
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :return: 本金部分
    """
    total = pmt(rate, nper, pv, fv, when)
    interest = ipmt(rate, per, nper, pv, fv, when)
    return total - interest


def rate(nper, pmt, pv, fv=0, when='end', guess=0.1, tol=1e-6, max_iter=100):
    """计算利率 (Rate).

    使用牛顿迭代法求解。

    :param nper: 总期数
    :param pmt: 每期付款额
    :param pv: 现值
    :param fv: 未来值，默认为0
    :param when: 付款时间 ('end' 期末或 'begin' 期初)
    :param guess: 初始猜测值
    :param tol: 收敛容差
    :param max_iter: 最大迭代次数
    :return: 利率

    示例:
        >>> rate(10*12, -100, 10000)  # 每月还100，10年还清10000，求月利率
        0.004291074821880434
    """
    when = _convert_when(when)

    def _f(r):
        if r == 0:
            return fv + pv + pmt * nper
        return fv + pv * (1 + r) ** nper + pmt * (1 + r * when) / r * ((1 + r) ** nper - 1)

    def _fprime(r):
        if r == 0:
            return pmt * nper * (nper + 1) / 2 + nper * pv
        # 数值导数
        h = 1e-8
        return (_f(r + h) - _f(r - h)) / (2 * h)

    r = guess
    for _ in range(max_iter):
        f_val = _f(r)
        if abs(f_val) < tol:
            return r

        fprime_val = _fprime(r)
        if abs(fprime_val) < 1e-12:
            raise ValueError("Derivative too small, cannot continue")

        r_new = r - f_val / fprime_val
        if abs(r_new - r) < tol:
            return r_new
        r = r_new

    raise ValueError(f"Failed to converge after {max_iter} iterations")
