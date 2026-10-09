"""基于表达式的特征衍生.

默认使用受限 AST 对数值、字符串和布尔列进行向量化计算。
只有显式 trusted_python=True 才允许执行可信开发者提供的 Python 表达式。
"""

import numpy as np
import ast
import operator
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


_FUNCTIONS = {name: getattr(np, name) for name in (
    "where", "sin", "cos", "tan", "abs", "exp", "log", "sqrt", "power", "floor", "ceil",
    "round", "minimum", "maximum", "clip", "isnan", "isfinite", "mean", "sum", "std", "var", "median",
)}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow, ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert, ast.Not: np.logical_not}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_MAX_ARGS = {name: 1 for name in _FUNCTIONS}
_MAX_ARGS.update(where=3, power=2, round=2, minimum=2, maximum=2, clip=3)
_KEYWORDS = {"axis", "keepdims", "decimals", "ddof", "a_min", "a_max"}


def _parse_expression(expression):
    """仅接收数据计算语法；无属性遍历、下标、导入、推导式或任意调用。"""
    if len(expression) > 10000:
        raise ValueError("衍生表达式过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("衍生表达式语法无效；非标字段请使用 col('字段名')") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 256:
        raise ValueError("衍生表达式过于复杂")
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
               ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Call, ast.Attribute, ast.keyword,
               *_BINARY, *_UNARY, *_COMPARE)
    for node in nodes:
        if not isinstance(node, allowed):
            raise ValueError(f"衍生表达式不允许语法: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("衍生表达式不允许访问内部名称")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "np" or node.attr not in {*_FUNCTIONS, "nan", "inf"}:
                raise ValueError("衍生表达式不允许任意属性访问")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name not in {*_FUNCTIONS, "col"}:
                raise ValueError("衍生表达式仅允许白名单函数")
            if len(node.args) > _MAX_ARGS.get(name, 1) or any(keyword.arg not in _KEYWORDS for keyword in node.keywords):
                raise ValueError("衍生表达式的函数参数不受支持；禁止参数展开和原地输出")
    return tree.body


def _evaluate_expression(node, context):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id in ("nan", "inf"):
            return getattr(np, node.id)
        raise ValueError(f"衍生表达式引用了不存在的字段: {node.id}")
    if isinstance(node, ast.Attribute):
        if node.attr in ("nan", "inf"):
            return getattr(np, node.attr)
        raise ValueError("函数名称只能用于调用")
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate_expression(node.left, context), _evaluate_expression(node.right, context))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate_expression(node.operand, context))
    if isinstance(node, ast.BoolOp):
        result = _evaluate_expression(node.values[0], context)
        operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        for value in node.values[1:]:
            result = operation(result, _evaluate_expression(value, context))
        return result
    if isinstance(node, ast.Compare):
        left = _evaluate_expression(node.left, context)
        result = True
        for operation, right in zip(node.ops, node.comparators):
            right = _evaluate_expression(right, context)
            result = np.logical_and(result, _COMPARE[type(operation)](left, right))
            left = right
        return result
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        if name == "col":
            if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                raise ValueError("col 必须接收一个常量字段名字符串")
            column = node.args[0].value
            if column not in context:
                raise ValueError(f"衍生表达式引用了不存在的字段: {column}")
            return context[column]
        return _FUNCTIONS[name](
            *[_evaluate_expression(arg, context) for arg in node.args],
            **{item.arg: _evaluate_expression(item.value, context) for item in node.keywords},
        )
    raise ValueError("衍生表达式包含不支持的节点")


class NumExprDerive(BaseEstimator, TransformerMixin):
    """基于表达式的特征衍生器（sklearn Transformer）。

    通过一组 ``(新特征名, 表达式)`` 规则批量衍生新特征，兼容 sklearn Pipeline。
    根据输入类型自动选择计算后端：

    - 输入 ``DataFrame``：受限 AST 调用 numpy/pandas 向量化运算，支持中文列名。
      非标字段通过 ``col('年 龄')`` 引用；后续规则可引用此前产生的衍生字段。
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
    :param trusted_python: 默认 False，禁止导入、任意属性访问和任意函数调用。
        True 允许完整 Python 表达式，仅适用于可信开发者代码，不能用于不可信配置。

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

    常见 ndarray 表达式使用 numexpr；DataFrame 及扩展语法使用受限 AST
    驱动 numpy/pandas 向量运算，而不是 pandas.eval 或不受限 Python eval。
    """

    def __init__(self, derivings=None, trusted_python=False):
        """初始化特征衍生器。

        :param derivings: 衍生规则列表，每个元素为 ``(name, expr)`` 二元组，
            ``name`` 为新特征列名（str），``expr`` 为表达式字符串（str）。
            默认 ``None``，但 ``None``/空列表会立即抛出 ``ValueError``
        :raises ValueError: derivings 为空、非列表，或元素不是 (str, str) 二元组时
        """
        self.derivings = derivings
        self.trusted_python = trusted_python
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
        if getattr(X, "ndim", np.ndim(X)) != 2:
            raise ValueError("X 必须是二维数据")
        if not self.trusted_python:
            for _, expression in self.derivings:
                _parse_expression(expression)
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
        if not isinstance(self.trusted_python, (bool, np.bool_)):
            raise ValueError("trusted_python 必须为布尔值，不能使用字符串开启可信模式")
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
        feature_names = X.columns.tolist()
        if not X.columns.is_unique:
            raise ValueError("输入特征列名不能重复")
        derived_names = [name for name, _ in self.derivings]
        if len(set(derived_names)) != len(derived_names) or set(derived_names).intersection(feature_names):
            raise ValueError("衍生字段名不能重复或覆盖原始字段")
        self.features_names_ = feature_names
        result = X.copy()
        for name, expr in self.derivings:
            context = {column: result[column] for column in result.columns}
            if self.trusted_python:
                context.update(_FUNCTIONS)
                context["np"] = np
                result[name] = eval(expr, context)
            else:
                result[name] = _evaluate_expression(_parse_expression(expr), context)
        result = result[feature_names + derived_names]
        return result

    def _transform_ndarray(self, X):
        """转换 ndarray（仅支持数值类型）。"""
        try:
            import numexpr as ne
        except ImportError:
            raise ImportError("未安装 numexpr，请执行命令安装: pip install numexpr")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数据")
        context = {"f%d" % i: X[:, i] for i in range(X.shape[1])}
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)

        for i, (name, expr) in enumerate(self.derivings):
            if self.trusted_python:
                X_derived[:, i] = eval(expr, {**context, **_FUNCTIONS, "np": np})
            else:
                tree = _parse_expression(expr)
                # 保持常见 ndarray 表达式的 numexpr 路径；扩展语法走同一安全 AST。
                if any(isinstance(node, (ast.Attribute, ast.BoolOp)) or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "col") for node in ast.walk(tree)):
                    X_derived[:, i] = _evaluate_expression(tree, context)
                else:
                    X_derived[:, i] = ne.evaluate(expr, local_dict=context, global_dict={})
            context[name] = X_derived[:, i]

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
        self._check_keywords()
        if isinstance(X, DataFrame):
            return self._transform_frame(X)
        return self._transform_ndarray(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "dataframe"],
            "allow_nan": True,
        }
