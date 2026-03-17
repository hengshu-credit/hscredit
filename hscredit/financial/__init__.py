"""金融计算模块.

提供常用的金融计算函数，参考 numpy_financial 实现。
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
