"""特征工程模块.

提供特征衍生和转换功能，支持基于表达式的特征高效计算。

**核心功能**

- 表达式特征衍生 (NumExprDerive): 使用 numexpr 引擎根据表达式计算新特征，
  支持 where、sin、cos、abs 等数学函数，支持条件逻辑

**快速开始**

    >>> import pandas as pd
    >>> from hscredit.core.feature_engineering import NumExprDerive
    >>> X = pd.DataFrame({
    ...     "f0": [2, 1.0, 3],
    ...     "f1": [1, 2, 3],
    ...     "f2": [2, 3, 4],
    ...     "f3": [2.1, 1.4, -6.2]
    ... })
    >>> fd = NumExprDerive(derivings=[
    ...     ("f4", "where(f1>1, 0, 1)"),
    ...     ("f5", "f1+f2"),
    ...     ("f6", "sin(f1)"),
    ...     ("f7", "abs(f3)")
    ... ])
    >>> X_new = fd.fit_transform(X)
"""

from .expression import NumExprDerive

__all__ = [
    'NumExprDerive',
]
