"""基础金融计算函数.

提供货币时间价值（Time Value of Money）相关的基础计算，包括未来值、现值、
每期付款额、期数、利息/本金拆分以及利率反解。函数命名与参数约定对齐 Excel
财务函数与 ``numpy_financial``，所有函数均支持标量与数组（向量化）输入。

**符号约定（现金流方向）**

遵循"现金流入为正、现金流出为负"的通用约定：贷款本金对借款人是流入（正），
每期还款是流出（负）；因此 ``pmt`` 等结果常为负值。

**向量化约定**

标量输入返回标量；如需对多组参数批量计算，须将 *所有* 数值参数（含 ``when``）
统一传为等长 numpy 数组，函数对其逐元素并行求解。

**子函数**

- fv: 未来值 Future Value
- pv: 现值 Present Value
- pmt: 每期付款额 Payment
- nper: 期数 Number of Periods
- ipmt: 某期利息部分 Interest Payment
- ppmt: 某期本金部分 Principal Payment
- rate: 每期利率 Rate（牛顿迭代反解）

**引用**

- numpy-financial 文档：https://numpy.org/numpy-financial/latest/
- Excel 财务函数 FV/PV/PMT/NPER/IPMT/PPMT/RATE：
  https://support.microsoft.com/zh-cn/office/fv-函数-2eef9f44-a084-4c61-bdd8-4fe4bb1b71b3
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

    在固定每期利率与等额分期付款条件下，计算一系列现金流在最后一期期末的累计价值。
    常用于储蓄/投资终值、定投账户余额测算等场景。

    满足等式 ``fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0``
    （当 ``rate == 0`` 时退化为 ``fv = -(pv + pmt*nper)``）。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param nper: 总付款（计息）期数
    :param pmt: 每期固定付款额，按现金流方向约定（流出为负，如每月储蓄 -100）
    :param pv: 现值，即期初一次性金额（流出为负，如初始投入 -100）
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金 ordinary
          annuity），每期现金流在期末发生——最常见
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款
          （预付年金 annuity-due），每期现金流在期初发生，比期末多计一期利息
        - 也可传入 numpy 数组对每个元素分别指定时点（向量化）

    :return: 未来值（与输入现金流方向相反，通常为正表示终值收入）；标量入参返回标量，
        数组入参返回 numpy 数组

    **参考样例**

    标量计算（月利率 0.05/12，10 年共 120 期，每月储蓄 -100，初始 -100）::

        >>> fv(0.05/12, 10*12, -100, -100)
        15692.92889433575

    期初付款（预付年金，终值更高）::

        >>> fv(0.05/12, 10*12, -100, -100, when='begin')
        15757.629844104778

    向量化计算（同时评估多个利率；注意所有数值参数需为等长数组，含 ``when``）::

        >>> import numpy as np
        >>> fv(np.array([0.05/12, 0.06/12]), np.array([120, 120]),
        ...    np.array([-100, -100]), np.array([-100, -100]), np.array([0, 0]))
        array([15692.92889434, 16569.87435405])

    **引用**

    对应 Excel ``FV`` 函数与 ``numpy_financial.fv``：
    https://numpy.org/numpy-financial/latest/functions/fv.html
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


# 函数别名：供 ipmt 等内部调用，避免被 ``fv`` 形参遮蔽
_fv = fv


def pv(rate, nper, pmt, fv=0, when='end'):
    """计算现值 (Present Value).

    在固定每期利率与等额分期付款条件下，将未来各期现金流与终值折算到期初的等价价值。
    常用于贷款可借本金测算、未来收益的当前估值等场景。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param nper: 总付款（计息）期数
    :param pmt: 每期固定付款额，按现金流方向约定（流出为负）
    :param fv: 未来值，即最后一期期末的目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款（预付年金）
        - 也可传入 numpy 数组对每个元素分别指定时点（向量化）

    :return: 现值（与未来现金流方向相反，通常为负表示期初支出）

    **参考样例**

    每月收款 100、共 120 期、月利率 0.05/12 时，期初一次性等价价值::

        >>> pv(0.05/12, 10*12, -100)
        9428.135032823473

    向量化计算（所有数值参数需为等长数组）::

        >>> import numpy as np
        >>> pv(np.array([0.05/12, 0.06/12]), np.array([120, 120]),
        ...    np.array([-100, -100]), np.array([0, 0]), np.array([0, 0]))
        array([9428.13503282, 9007.34533272])

    **引用**

    对应 Excel ``PV`` 函数与 ``numpy_financial.pv``：
    https://numpy.org/numpy-financial/latest/functions/pv.html
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

    在给定现值、每期利率与期数条件下，计算等额本息分期的每期偿付额。
    是评分卡/信贷场景中测算月供的核心函数。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param nper: 总付款（计息）期数
    :param pv: 现值，即贷款本金或投资额（借款人视角下本金为正流入）
    :param fv: 未来值，即最后一期期末的目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款（预付年金）
        - 也可传入 numpy 数组对每个元素分别指定时点（向量化）

    :return: 每期付款额（与本金方向相反，通常为负表示每期支出）

    **参考样例**

    本金 10000、月利率 0.05/12、分 120 期等额本息的月供::

        >>> pmt(0.05/12, 10*12, 10000)
        -106.06551523907554

    **引用**

    对应 Excel ``PMT`` 函数与 ``numpy_financial.pmt``：
    https://numpy.org/numpy-financial/latest/functions/pmt.html
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

    在给定每期利率、每期付款额与现值条件下，计算达到目标未来值所需的付款期数。
    返回值一般为非整数，表示理论上的精确期数。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param pmt: 每期固定付款额，按现金流方向约定（流出为负）
    :param pv: 现值，即初始投资或贷款本金
    :param fv: 未来值，即目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款（预付年金）
        - 也可传入 numpy 数组对每个元素分别指定时点（向量化）

    :return: 达到目标未来值所需的期数（通常为非整数）

    **参考样例**

    本金 10000、每期还 -100、月利率 0.05/12 时所需期数::

        >>> nper(0.05/12, -100, 10000)
        129.62847166352213

    **引用**

    对应 Excel ``NPER`` 函数与 ``numpy_financial.nper``：
    https://numpy.org/numpy-financial/latest/functions/nper.html
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

    将等额本息分期中第 ``per`` 期的还款额拆分出"利息"部分。基于摊销公式：
    第 ``per`` 期利息 = 期初剩余本金 × ``rate``。满足 ``ipmt + ppmt == pmt``。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param per: 指定期次，取值范围 1 ~ ``nper``（第 1 期至第 ``nper`` 期）
    :param nper: 总付款（计息）期数
    :param pv: 现值，即贷款本金或投资额
    :param fv: 未来值，即最后一期期末的目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款
          （预付年金；此时首期无利息，返回 0，其余期相应折现一期）

    :return: 第 ``per`` 期的利息支付额（与付款额同向，通常为负）

    **参考样例**

    本金 10000、月利率 0.05/12、共 120 期，第 1 期的利息（≈本金×月利率）::

        >>> ipmt(0.05/12, 1, 12*10, 10000)
        -41.666666666666664

    **引用**

    对应 Excel ``IPMT`` 函数与 ``numpy_financial.ipmt``：
    https://numpy.org/numpy-financial/latest/functions/ipmt.html
    """
    when = _convert_when(when)
    rate_a = np.asarray(rate, dtype=float)
    per_a = np.asarray(per)
    when_a = np.asarray(when)

    # 每期等额付款额
    total_pmt = pmt(rate, nper, pv, fv, when)
    # 第 per 期期初的剩余本金 = 现值经过 (per-1) 期后的未来值
    remaining_balance = _fv(rate, per_a - 1, total_pmt, pv, when)
    # 利息 = 剩余本金 × 利率（与 pmt 同号，表示支出为负）
    result = remaining_balance * rate_a
    # 期初付款（when='begin'）修正：首期无利息，其余期需折现一期
    result = np.where(when_a == 1, result / (1 + rate_a), result)
    result = np.where((when_a == 1) & (per_a == 1), 0.0, result)

    if result.ndim == 0:
        return float(result)
    return result


def ppmt(rate, per, nper, pv, fv=0, when='end'):
    """计算给定期间的本金部分 (Principal Payment).

    将等额本息分期中第 ``per`` 期的还款额拆分出"本金"部分，等于该期总付款额减去利息
    （``ppmt = pmt - ipmt``）。随着期次推进，本金占比逐期增大、利息占比逐期减小。

    :param rate: 每期利率，小数表示（如年利率 5% 按月计息则传 ``0.05/12``）
    :param per: 指定期次，取值范围 1 ~ ``nper``（第 1 期至第 ``nper`` 期）
    :param nper: 总付款（计息）期数
    :param pv: 现值，即贷款本金或投资额
    :param fv: 未来值，即最后一期期末的目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款（预付年金）

    :return: 第 ``per`` 期的本金支付额（与付款额同向，通常为负）

    **参考样例**

    本金 10000、月利率 0.05/12、共 120 期，第 1 期偿还的本金::

        >>> ppmt(0.05/12, 1, 12*10, 10000)
        -64.39884857240887

    **引用**

    对应 Excel ``PPMT`` 函数与 ``numpy_financial.ppmt``：
    https://numpy.org/numpy-financial/latest/functions/ppmt.html
    """
    total = pmt(rate, nper, pv, fv, when)
    interest = ipmt(rate, per, nper, pv, fv, when)
    return total - interest


def rate(nper, pmt, pv, fv=0, when='end', guess=0.1, tol=1e-6, max_iter=100):
    """计算每期利率 (Rate).

    在给定期数、每期付款额、现值与未来值条件下，使用牛顿迭代法（Newton-Raphson）
    反解使现金流等式成立（净现值为零）的 *每期* 利率。如需年化，自行乘以每年期数。

    :param nper: 总付款（计息）期数
    :param pmt: 每期固定付款额，按现金流方向约定（流出为负）
    :param pv: 现值，即初始投资或贷款本金
    :param fv: 未来值，即最后一期期末的目标余额，默认为 ``0``
    :param when: 每期付款发生的时点，默认 ``'end'``。可取以下枚举值：

        - ``'end'`` / ``'e'`` / ``'finish'`` / ``0``：期末付款（普通年金）
        - ``'begin'`` / ``'b'`` / ``'beginning'`` / ``'start'`` / ``1``：期初付款（预付年金）

    :param guess: 牛顿迭代的初始猜测利率，默认为 ``0.1``（即 10%）。当方程存在多解或
        迭代不收敛时，可调整该初值
    :param tol: 收敛容差，残差或步长小于该值即视为收敛，默认为 ``1e-6``
    :param max_iter: 最大迭代次数，默认为 ``100``
    :return: 每期利率（小数表示）
    :raises ValueError: 导数过小或在 ``max_iter`` 次迭代内无法收敛时抛出

    **参考样例**

    本金 10000、每期还 -100、共 120 期时反解出的月利率::

        >>> rate(10*12, -100, 10000)
        0.0031141819460226306

    **引用**

    对应 Excel ``RATE`` 函数与 ``numpy_financial.rate``；牛顿迭代法参见
    https://numpy.org/numpy-financial/latest/functions/rate.html
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
