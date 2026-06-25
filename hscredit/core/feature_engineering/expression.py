"""基于表达式的特征衍生.

支持任意类型数据（数值、字符串、布尔等），使用 pandas eval + numexpr 进行高效计算。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class NumExprDerive(BaseEstimator, TransformerMixin):
    """基于表达式的特征衍生器（sklearn Transformer）。

    通过一组 ``(新特征名, 表达式)`` 规则批量衍生新特征，兼容 sklearn Pipeline。
    根据输入类型自动选择计算后端：

    - 输入 ``DataFrame``：纯数值列走 numpy 向量化计算（最快），含字符串/布尔等
      混合类型列走 pandas Series 计算（类型安全），从而支持任意类型数据。
    - 输入 ``ndarray``：走 numexpr 计算（需安装 ``numexpr``），列以 ``f0``、``f1`` …
      命名引用。

    表达式语法基于 Python/numpy，额外支持 ``where(cond, a, b)``（自动转换为
    :func:`numpy.where`）以及 ``sin``/``cos``/``tan``/``abs``/``exp``/``log``/
    ``sqrt``/``power``/``floor``/``ceil`` 等 numpy 函数。

    **参数**

    :param derivings: 衍生规则列表，每个元素为 ``(name, expr)`` 二元组：

        - ``name`` (str)：新特征列名
        - ``expr`` (str)：基于已有列名的表达式字符串，如 ``"f1 + f2"``、
          ``"where(score >= 600, '高', '低')"``

        为 ``None`` 或空列表时在初始化/fit 阶段抛出 ``ValueError``

    **属性**

    - features_names_: 拟合/转换时记录的原始输入列名列表（DataFrame 输入时）

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

    **引用**

    数值/ndarray 路径使用 numexpr 实现表达式的高性能向量化求值，详见
    https://numexpr.readthedocs.io/ ；DataFrame 路径基于 pandas 计算引擎
    (:func:`pandas.eval`)。
    """

    def __init__(self, derivings=None):
        """初始化特征衍生器。

        :param derivings: 衍生规则列表，每个元素为 ``(name, expr)`` 二元组，
            ``name`` 为新特征列名（str），``expr`` 为表达式字符串（str）。
            默认 ``None``，但 ``None``/空列表会立即抛出 ``ValueError``
        :raises ValueError: derivings 为空、非列表，或元素不是 (str, str) 二元组时
        """
        self.derivings = derivings
        self._check_keywords()

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, TargetTags, TransformerTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
        )

    def fit(self, X, y=None):
        """拟合特征衍生器（校验规则与输入维度，不学习任何参数）。

        :param X: 输入数据，``DataFrame`` 或 2 维 ``ndarray``
        :param y: 目标变量，未使用，仅为兼容 sklearn 接口而保留
        :return: self，支持链式调用
        :raises ValueError: derivings 非法，或 X 不是 2 维时
        """
        self._check_keywords()
        if isinstance(X, pd.DataFrame):
            if X.ndim != 2:
                raise ValueError("X 必须是二维数据")
        return self

    def _convert_where_to_np(self, expr):
        """将 where(cond, a, b) 转换为 np.where(cond, a, b).

        pandas eval 不支持 where() 函数，使用 np.where() 代替，
        并通过 Python eval + 列数组来执行。
        """
        import re

        pattern = re.compile(r'(?<![\w.])where\s*\(')
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
            raise ValueError("特征衍生规则不能为空")
        if not isinstance(derivings, list):
            raise ValueError("特征衍生规则必须是列表")
        if not derivings:
            raise ValueError("特征衍生规则不能为空")
        for i, entry in enumerate(derivings):
            if not isinstance(entry, tuple):
                raise ValueError(f"第 {i} 条特征衍生规则必须是元组")
            if len(entry) != 2:
                raise ValueError(f"第 {i} 条特征衍生规则必须是二元组 (名称, 表达式)")
            name, expr = entry
            if not isinstance(name, str) or not isinstance(expr, str):
                raise ValueError(f"第 {i} 条特征衍生规则的名称和表达式都必须是字符串")

    def _transform_frame(self, X):
        """转换 DataFrame，支持任意类型数据。

        策略：
        - 纯数值列 -> numpy eval（最快）
        - 含字符串/布尔/日期等 -> pandas Series eval（类型安全）
        np.where 接收 Series 时行为正确（字符串/布尔/数值均能正确处理）。
        """
        import re

        feature_names = X.columns.tolist()
        self.features_names_ = feature_names
        result = X.copy()

        # 预检测列是否全为数值型
        col_pattern = re.compile(r'\b([A-Za-z_]\w*)\b')
        reserved = {
            'np', 'where', 'sin', 'cos', 'tan', 'abs', 'exp', 'log',
            'sqrt', 'power', 'floor', 'ceil', 'round', 'max', 'min',
            'mean', 'sum', 'std', 'var', 'median', 'nan', 'inf',
            'True', 'False', 'None', 'and', 'or', 'not', 'is', 'in',
        }
        numpy_functions = {
            name: getattr(np, name)
            for name in ('sin', 'cos', 'tan', 'abs', 'exp', 'log', 'sqrt', 'power', 'floor', 'ceil')
        }

        for name, expr in self.derivings:
            converted = self._convert_where_to_np(expr)

            # 找出表达式中涉及的列
            raw_cols = col_pattern.findall(converted)
            involved = [c for c in raw_cols if c not in reserved and c in X.columns]
            all_numeric = all(pd.api.types.is_numeric_dtype(X[c].dtype) for c in involved)

            if all_numeric:
                # 数值型：numpy eval（最快）
                context = {col: X[col].values for col in involved}
            else:
                # 混合类型：pandas Series eval（支持字符串/布尔）
                context = {col: X[col] for col in X.columns}
            context.update(numpy_functions)
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
            raise ImportError("未安装 numexpr，请执行命令安装: pip install numexpr")

        context = {"f%d" % i: X[:, i] for i in range(X.shape[1])}
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)

        for i, (name, expr) in enumerate(self.derivings):
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)

        return np.hstack((X, X_derived))

    def transform(self, X):
        """按 derivings 规则衍生新特征并追加到原始特征之后。

        :param X: 输入数据：

            - ``DataFrame``：表达式按列名引用，返回 ``原始列 + 衍生列`` 的新 DataFrame
            - 2 维 ``ndarray``：列以 ``f0``/``f1``/… 引用，返回水平拼接的新数组
              （需安装 ``numexpr``）

        :return: 含衍生特征的 ``DataFrame``（DataFrame 输入）或 ``ndarray``（数组输入）
        :raises ImportError: 输入为 ndarray 且未安装 numexpr 时
        """
        if isinstance(X, DataFrame):
            return self._transform_frame(X)
        return self._transform_ndarray(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "dataframe"],
            "allow_nan": True,
        }
