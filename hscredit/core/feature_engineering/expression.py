"""基于表达式的特征衍生.

支持任意类型数据（数值、字符串、布尔等），使用 pandas eval + numexpr 进行高效计算。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class NumExprDerive(BaseEstimator, TransformerMixin):
    """基于表达式的特征衍生。

    支持任意类型数据（数值、字符串、布尔等），使用 pandas eval + numexpr
    进行高效计算。

    :param derivings: 衍生规则列表，每个元素是 (name, expr) 元组

    **参考样例**

    >>> import pandas as pd
    >>> from hscredit.core.feature_engineering import NumExprDerive
    >>> X = pd.DataFrame({
    ...     "f0": [2, 1.0, 3],
    ...     "f1": [np.inf, 2, 3],
    ...     "f2": [2, 3, 4],
    ...     "f3": [2.1, 1.4, -6.2]
    ... })
    >>> fd = NumExprDerive(derivings=[
    ...     ("f4", "where(f1>1, 0, 1)"),  # 条件表达式
    ...     ("f5", "f1+f2"),              # 加法运算
    ...     ("f6", "sin(f1)"),            # 三角函数
    ...     ("f7", "abs(f3)")              # 绝对值
    ... ])
    >>> fd.fit_transform(X)

    **混合类型样例**

    >>> X = pd.DataFrame({
    ...     "score": [650, 580, 720, 490],
    ...     "status": ["正常", "逾期", "正常", "关注"],
    ...     "is_vip": [True, False, True, False]
    ... })
    >>> fd = NumExprDerive(derivings=[
    ...     ("score_band", "where(score >= 600, '高', '低')"),  # 数值条件字符串
    ...     ("flag", "where((status == '逾期') | is_vip, 1, 0)"),  # 混合类型条件
    ...     ("score_level", "where(score > 600, score * 1.1, score * 0.9)"),  # 数值条件
    ... ])
    >>> fd.fit_transform(X)
    """

    def __init__(self, derivings=None):
        """
        :param derivings: list, default=None.
            每个元素是 (name, expr) 元组，表示一个衍生规则。
            name 是新特征的名称，expr 是 pandas eval 表达式。
        """
        self.derivings = derivings

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, TargetTags, TransformerTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
        )

    def fit(self, X, y=None):
        """拟合特征衍生器。"""
        self._check_keywords()
        if isinstance(X, pd.DataFrame):
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
        return self

    def _convert_where_to_np(self, expr):
        """将 where(cond, a, b) 转换为 np.where(cond, a, b).

        pandas eval 不支持 where() 函数，使用 np.where() 代替，
        并通过 Python eval + 列数组来执行。
        """
        import re
        pattern = re.compile(r'where\s*\(')
        result = expr
        while True:
            m = pattern.search(result)
            if not m:
                break

            # Find the matching ')' by counting nesting depth
            depth = 0
            end = m.end()
            while end < len(result):
                if result[end] == '(':
                    depth += 1
                elif result[end] == ')':
                    if depth == 0:
                        end += 1
                        break
                    depth -= 1
                end += 1
            else:
                break

            full_call = result[m.start():end]
            inner = full_call[len(m.group(0)):-1]

            # Split by top-level comma (respecting nested parentheses)
            args = []
            depth = 0
            current = ''
            for ch in inner:
                if ch == '(':
                    depth += 1
                    current += ch
                elif ch == ')':
                    depth -= 1
                    current += ch
                elif ch == ',' and depth == 0:
                    args.append(current.strip())
                    current = ''
                else:
                    current += ch
            if current.strip():
                args.append(current.strip())

            if len(args) < 3:
                result = result[:m.start()] + full_call + result[end:]
                break

            np_where = f'np.where({args[0]}, {args[1]}, {args[2]})'
            result = result[:m.start()] + np_where + result[end:]

        return result

    def _check_keywords(self):
        """检查参数有效性。"""
        derivings = self.derivings
        if derivings is None:
            raise ValueError("Deriving rules should not be empty!")
        if not isinstance(derivings, list):
            raise ValueError("Deriving rules should be a list!")
        for i, entry in enumerate(derivings):
            if not isinstance(entry, tuple):
                raise ValueError(f"The {i}-th deriving rule should be a tuple!")
            if len(entry) != 2:
                raise ValueError(f"The {i}-th deriving rule is not a two-element (name, expr) tuple!")
            name, expr = entry
            if not isinstance(name, str) or not isinstance(expr, str):
                raise ValueError(f"The {i}-th deriving rule is not a two-string tuple!")

    def _transform_frame(self, X):
        """转换 DataFrame，支持任意类型数据。"""
        feature_names = X.columns.tolist()
        self.features_names_ = feature_names
        result = X.copy()

        for name, expr in self.derivings:
            converted = self._convert_where_to_np(expr)
            context = {col: X[col].values for col in X.columns}
            context['np'] = np
            result[name] = eval(converted, context)

        derived_names = [name for name, _ in self.derivings]
        result = result[feature_names + derived_names]
        return result

    def _transform_ndarray(self, X):
        """转换 ndarray（仅支持数值类型）。"""
        try:
            import numexpr as ne
        except ImportError:
            raise ImportError("numexpr is not installed. Install it with: pip install numexpr")

        context = {"f%d" % i: X[:, i] for i in range(X.shape[1])}
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)

        for i, (name, expr) in enumerate(self.derivings):
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)

        return np.hstack((X, X_derived))

    def transform(self, X):
        """转换数据。"""
        if isinstance(X, DataFrame):
            return self._transform_frame(X)
        return self._transform_ndarray(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": True,
        }
