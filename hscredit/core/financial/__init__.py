"""金融计算模块.

提供常用的金融计算函数，参考 numpy_financial 实现。

子模块:
    - basic: 基础金融计算（FV、PV、PMT、NPER、IPMT、PPMT、RATE）
    - advanced: 高级金融计算（NPV、IRR、MIRR）

**参数**

所有函数均支持标量和数组输入，自动进行向量化计算。

**参考样例**

>>> from hscredit.core.financial import fv, pv, npv, irr
>>> fv(0.05/12, 10*12, -100, -100)  # 未来值
15692.93
>>> pv(0.05/12, 10*12, -100)       # 现值
9428.14
>>> npv(0.05, [-1000, 300, 400, 400, 300])  # 净现值
265.69
>>> irr([-1000, 300, 400, 400, 300])         # 内部收益率
0.143
"""

from .basic import (
    fv,  # 未来值
    pv,  # 现值
    pmt,  # 每期付款额
    nper,  # 期数
    ipmt,  # 利息部分
    ppmt,  # 本金部分
    rate,  # 利率
)
from .advanced import (
    npv,  # 净现值
    irr,  # 内部收益率
    mirr,  # 修正内部收益率
)

__all__ = [
    'fv',
    'pv',
    'pmt',
    'nper',
    'ipmt',
    'ppmt',
    'rate',
    'npv',
    'irr',
    'mirr',
]
