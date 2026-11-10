"""规则表达式优化器。

将规则表达式解析为表达式树（AST），应用布尔代数定律做等价化简，并以
``&``/``|``/``~`` 符号形式重新输出规范、易读的表达式。供 :class:`~hscredit.core.rules.Rule`
在 ``&``/``|``/``~``/``^`` 组合时自动调用。

**对外函数**

- :func:`optimize_expr`：化简表达式（幂等律、吸收律、双重否定等）
- :func:`beautify_expr`：美化表达式（统一运算符符号、去除冗余括号）
- :func:`get_expr_variables`：提取表达式引用的变量名

模块内的 ``ExprNode`` / ``ExprParser`` / ``ExprOptimizer`` 等为内部实现细节，
不属于稳定对外接口。

**引用**

- 布尔代数化简定律（幂等律 Idempotent、吸收律 Absorption、双重否定 Double negation）：
  https://en.wikipedia.org/wiki/Boolean_algebra#Laws
"""

import ast
import re
from dataclasses import dataclass
from typing import Set, List, Optional, Union

import numpy as np


def _canonical(node: "ExprNode") -> str:
    """返回节点的规范化字符串（统一空白），用于等价比较。

    复用 AST 格式化输出（已将 ``and`` / ``or`` 统一为 ``&`` / ``|`` 符号），
    做大小写敏感的精确比较，保留列名和字符串字面量内部的空白。
    """
    return node.to_string().strip()


class ExprNode:
    """表达式节点基类。"""

    def __init__(self):
        self.parent = None

    def get_variables(self) -> Set[str]:
        """获取表达式中使用的变量名。"""
        raise NotImplementedError

    def to_string(self, parent_op: Optional[str] = None) -> str:
        """转换为字符串表达式。"""
        raise NotImplementedError

    def simplify(self):
        """简化表达式。"""
        raise NotImplementedError


class VariableNode(ExprNode):
    """变量节点，如 age > 18 这样的原子表达式。"""

    def __init__(self, expr: str, syntax: Optional[ast.AST] = None):
        super().__init__()
        self.expr = expr
        self.syntax = syntax

    def get_variables(self) -> Set[str]:
        # 使用正则提取变量名
        variables = set()
        # 匹配形如 "age > 18" 中的变量名
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
        for match in re.finditer(pattern, self.expr):
            var = match.group(1)
            # 排除 Python 关键字
            if var not in {"and", "or", "not", "True", "False", "None"}:
                variables.add(var)
        return variables

    def to_string(self, parent_op: Optional[str] = None) -> str:
        return self.expr

    def simplify(self):
        interval = _numeric_interval(self)
        return _interval_node(interval) if interval is not None else self


class BinaryOpNode(ExprNode):
    """二元运算符节点 (AND, OR, XOR)。"""

    def __init__(self, left: ExprNode, right: ExprNode, op: str):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op  # '&', '|', '^'
        left.parent = self
        right.parent = self

    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()

    def to_string(self, parent_op: Optional[str] = None) -> str:
        left_str = self.left.to_string(self.op)
        right_str = self.right.to_string(self.op)

        # 转换运算符为符号形式
        op_str = self.op_symbol

        # 根据父级运算符决定是否需要括号
        if parent_op and self.need_parens(parent_op):
            return f"({left_str} {op_str} {right_str})"
        return f"{left_str} {op_str} {right_str}"

    @property
    def op_symbol(self) -> str:
        """获取可读的运算符符号。"""
        # 将 'and' 转换为 '&', 'or' 转换为 '|'
        symbols = {"&": "&", "|": "|", "^": "^", "and": "&", "or": "|"}
        return symbols.get(self.op, self.op)

    def need_parens(self, parent_op: str) -> bool:
        """判断是否需要括号。"""
        # 同级运算符不需要括号（满足结合律）
        if parent_op == self.op:
            return False
        precedence = {"|": 1, "^": 2, "&": 3, "not": 4}
        return precedence.get(self.op, 0) < precedence.get(parent_op, 4)

    def simplify(self):
        """简化二元运算表达式。

        仅做  **保证等价**  的化简，确保化简前后表达式逻辑完全一致：

        - 幂等律： ``A & A = A`` 、 ``A | A = A``

        - 吸收律： ``A & (A | B) = A`` 、 ``A | (A & B) = A`` （含交换形式）

        这些定律只对 ``&`` / ``|`` 成立，对 ``^`` （异或）不成立（ ``A ^ A = False`` ），
        故仅在 ``&`` / ``|`` 上启用；吸收律仅在内层运算符与外层 *相反* 时成立。
        """
        # 递归简化子节点
        self.left = self.left.simplify()
        self.right = self.right.simplify()

        # 幂等律 / 吸收律仅对与、或运算成立，异或不能套用
        if self.op in ("&", "|"):
            left_expr = _canonical(self.left)
            right_expr = _canonical(self.right)

            # 幂等律: A & A = A, A | A = A
            if left_expr == right_expr:
                return self.left

            # 吸收律: A & (A | B) = A, A | (A & B) = A（及交换形式）。
            # 仅当内层运算符与外层相反时成立；若内外层运算符相同（如 A & (A & B)），
            # 不可吸收——否则会错误丢弃操作数导致语义改变。
            opposite = "|" if self.op == "&" else "&"
            if isinstance(self.right, BinaryOpNode) and self.right.op == opposite:
                if left_expr in self._flatten_same_op(self.right, self.right.op):
                    return self.left
            if isinstance(self.left, BinaryOpNode) and self.left.op == opposite:
                if right_expr in self._flatten_same_op(self.left, self.left.op):
                    return self.right

        simplified = _merge_numeric_conjunction(self) if self.op == "&" else self
        return _merge_category_conditions(simplified)

    @staticmethod
    def _flatten_same_op(node: ExprNode, op: str) -> List[str]:
        """沿同一运算符 ``op`` 展开 node，返回各操作数的规范化字符串列表。

        例如 ``A | B | C`` 沿 ``|`` 展开为 ``[A, B, C]`` ，用于吸收律中判断某子表达式
        是否为内层运算的操作数之一。
        """
        if isinstance(node, BinaryOpNode) and node.op == op:
            return BinaryOpNode._flatten_same_op(node.left, op) + BinaryOpNode._flatten_same_op(node.right, op)
        return [_canonical(node)]


class UnaryOpNode(ExprNode):
    """一元运算符节点 (NOT)。"""

    def __init__(self, operand: ExprNode, op: str = "not"):
        super().__init__()
        self.operand = operand
        self.op = op
        operand.parent = self

    def get_variables(self) -> Set[str]:
        return self.operand.get_variables()

    def to_string(self, parent_op: Optional[str] = None) -> str:
        operand_str = self.operand.to_string()
        return f"~({operand_str})"

    def simplify(self):
        """简化一元运算表达式。"""
        # 递归简化子节点
        self.operand = self.operand.simplify()

        # 双重否定: ~~A = A
        if isinstance(self.operand, UnaryOpNode):
            # 返回内层操作数（去掉两层not）
            return self.operand.operand

        # NOT True = False, NOT False = True (如果能确定的话)
        # 这里暂时不处理，因为我们的变量是表达式而非布尔值

        return self


def _ast_unparse(node: ast.AST) -> str:
    """保留负数、集合字面量及算术优先级，使用项目支持的 Python 3.9+ AST 格式化。"""
    return ast.unparse(node)


def _flatten_nodes(node, op):
    """按同一布尔运算符展开，保留其他子表达式的分组。"""
    if isinstance(node, BinaryOpNode) and node.op == op:
        return _flatten_nodes(node.left, op) + _flatten_nodes(node.right, op)
    return [node]


def _intersect_intervals(left, right):
    """合并同一字段的上下界；缺失值必须满足路径上的每个条件。"""
    feature, lower, upper, missing = left
    _, other_lower, other_upper, other_missing = right
    # pandas/numexpr 可能按输入列的 float32 精度比较常量。若更紧的闭区间
    # 与较松的开区间舍入到同一阈值，开区间仍是必要条件，不能丢弃。
    for first, second, is_lower in ((lower, other_lower, True), (upper, other_upper, False)):
        if first is None or second is None:
            continue
        tighter, looser = sorted((first, second), key=lambda bound: (bound[0], not bound[1]) if is_lower else bound, reverse=is_lower)
        if tighter[1] and not looser[1]:
            with np.errstate(over="ignore"):
                if np.float32(tighter[0]) == np.float32(looser[0]):
                    return None
    if other_lower is not None and (lower is None or (other_lower[0], not other_lower[1]) > (lower[0], not lower[1])):
        lower = other_lower
    if other_upper is not None and (upper is None or other_upper < upper):
        upper = other_upper
    return feature, lower, upper, missing and other_missing


def _numeric_interval(node):
    """仅识别字段与数值常量的区间，以及同字段的显式缺失分支。"""
    if isinstance(node, BinaryOpNode) and node.op == "&":
        left, right = _numeric_interval(node.left), _numeric_interval(node.right)
        if left is not None and right is not None and left[0] == right[0]:
            return _intersect_intervals(left, right)
        return None
    if isinstance(node, BinaryOpNode) and node.op == "|":
        for condition, missing in ((node.left, node.right), (node.right, node.left)):
            interval = _numeric_interval(condition)
            syntax = getattr(missing, "syntax", None)
            if interval is not None and isinstance(syntax, ast.Call) and not syntax.args and not syntax.keywords and isinstance(syntax.func, ast.Attribute) and syntax.func.attr == "isna" and isinstance(syntax.func.value, ast.Name) and syntax.func.value.id == interval[0]:
                return (*interval[:3], True)
        return None
    syntax = getattr(node, "syntax", None)
    if not isinstance(syntax, ast.Compare):
        return None
    interval = None
    operands = [syntax.left, *syntax.comparators]
    for left, operator, right in zip(operands, syntax.ops, operands[1:]):
        if not isinstance(operator, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            return None
        if isinstance(left, ast.Name):
            feature, value = left.id, right
            lower = isinstance(operator, (ast.Gt, ast.GtE))
        elif isinstance(right, ast.Name):
            feature, value = right.id, left
            lower = isinstance(operator, (ast.Lt, ast.LtE))
        else:
            return None
        try:
            value = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if type(value) not in (int, float):
            return None
        # 不知道执行列的 dtype 时，大整数与浮点比较可能有不同的舍入行为。
        if isinstance(value, int) and abs(value) > 2**53:
            return None
        closed = isinstance(operator, (ast.GtE, ast.LtE))
        bound = (value, closed)
        current = (feature, bound if lower else None, None if lower else bound, False)
        if interval is not None and interval[0] != feature:
            return None
        interval = current if interval is None else _intersect_intervals(interval, current)
        if interval is None:
            return None
    return interval


def _interval_node(interval):
    """按“特征 操作符 阈值”输出各边界，保留开闭边界与显式缺失分支。"""
    feature, lower, upper, missing = interval
    name = ast.Name(id=feature, ctx=ast.Load())
    bounds = []
    if lower is not None:
        syntax = ast.Compare(left=name, ops=[ast.GtE() if lower[1] else ast.Gt()], comparators=[ast.Constant(lower[0])])
        bounds.append(VariableNode(_ast_unparse(syntax), syntax))
    if upper is not None:
        syntax = ast.Compare(left=name, ops=[ast.LtE() if upper[1] else ast.Lt()], comparators=[ast.Constant(upper[0])])
        bounds.append(VariableNode(_ast_unparse(syntax), syntax))
    result = BinaryOpNode(bounds[0], bounds[1], "&") if len(bounds) == 2 else bounds[0]
    if missing:
        call = ast.Call(func=ast.Attribute(value=name, attr="isna", ctx=ast.Load()), args=[], keywords=[])
        result = BinaryOpNode(result, VariableNode(_ast_unparse(call), call), "|")
    return result


def _merge_numeric_conjunction(node):
    """集中简化路径交集，供 Rule 组合与树规则导出共同使用。"""
    terms, intervals, positions, seen = [], {}, {}, set()
    for term in _flatten_nodes(node, "&"):
        canonical = _canonical(term)
        if canonical in seen:
            continue
        seen.add(canonical)
        interval = _numeric_interval(term)
        if interval is None:
            terms.append(term)
            continue
        feature = interval[0]
        if feature not in intervals:
            positions[feature] = len(terms)
            intervals[feature] = interval
            terms.append(term)
        else:
            merged = _intersect_intervals(intervals[feature], interval)
            if merged is None:
                terms.append(term)
            else:
                intervals[feature] = merged
                terms[positions[feature]] = _interval_node(merged)
    result = terms[0]
    for term in terms[1:]:
        result = BinaryOpNode(result, term, "&")
    return result


@dataclass(frozen=True)
class _CategoryCondition:
    """非缺失类别的有限集合或补集；缺失值的命中状态独立保存。"""

    feature: str
    values: tuple
    excluded: bool = False
    include_missing: bool = False


def _category_values(syntax):
    """读取可安全比较的类别字面量，按原顺序去重，不猜测缺失标记。"""
    if not isinstance(syntax, (ast.List, ast.Tuple, ast.Set)):
        return None
    values = []
    for item in syntax.elts:
        try:
            value = ast.literal_eval(item)
        except (ValueError, TypeError):
            return None
        if type(value) not in (str, bool, int, float):
            return None
        if isinstance(value, int) and abs(value) > 2**53:
            return None
        if isinstance(value, float) and not np.isfinite(value):
            return None
        # isin 会根据集合成员推断数组类型；混合类型的增删可能改变比较精度。
        if values and type(value) is not type(values[0]):
            return None
        if value not in values:
            values.append(value)
    return tuple(values)


def _combine_categories(left, right, op):
    """对同字段的有限集合/补集做交并运算，不枚举未知类别全集。"""
    a, b = left.values, right.values
    if a and b and type(a[0]) is not type(b[0]):
        return None

    def difference(first, second):
        return tuple(value for value in first if value not in second)

    def intersection(first, second):
        return tuple(value for value in first if value in second)

    def union(first, second):
        return first + difference(second, first)

    if left.excluded == right.excluded:
        excluded = left.excluded
        operation = union if (op == "|") != excluded else intersection
        values = operation(a, b)
    else:
        included, rejected = (b, a) if left.excluded else (a, b)
        excluded = op == "|"
        values = difference(rejected, included) if excluded else difference(included, rejected)
    missing = (left.include_missing and right.include_missing) if op == "&" else (left.include_missing or right.include_missing)
    return _CategoryCondition(left.feature, values, excluded, missing)


def _category_condition(node):
    """识别类别集合、字符串等值条件及同字段的缺失/非缺失组合。"""
    if isinstance(node, BinaryOpNode) and node.op in ("&", "|"):
        left, right = _category_condition(node.left), _category_condition(node.right)
        if left is not None and right is not None and left.feature == right.feature:
            return _combine_categories(left, right, node.op)
        return None
    if isinstance(node, UnaryOpNode):
        condition = _category_condition(node.operand)
        if condition is not None:
            return _CategoryCondition(condition.feature, condition.values, not condition.excluded, not condition.include_missing)
        return None

    syntax = getattr(node, "syntax", None)
    excluded = False
    if isinstance(syntax, ast.Compare) and len(syntax.ops) == 1 and isinstance(syntax.left, ast.Name):
        feature = syntax.left.id
        operator, members = syntax.ops[0], syntax.comparators[0]
        if isinstance(operator, (ast.In, ast.NotIn)):
            values = _category_values(members)
            excluded = isinstance(operator, ast.NotIn)
        elif isinstance(operator, (ast.Eq, ast.NotEq)) and isinstance(members, ast.Constant) and isinstance(members.value, str):
            # pandas eval 将字符串等值比较按单元素 isin 执行，可与 in/not in 合并。
            values = (members.value,)
            excluded = isinstance(operator, ast.NotEq)
        else:
            return None
    elif isinstance(syntax, ast.Call) and isinstance(syntax.func, ast.Attribute) and isinstance(syntax.func.value, ast.Name) and not syntax.keywords:
        feature, method = syntax.func.value.id, syntax.func.attr
        if method in ("isna", "notna") and not syntax.args:
            return _CategoryCondition(feature, (), method == "notna", method == "isna")
        if method != "isin" or len(syntax.args) != 1:
            return None
        values = _category_values(syntax.args[0])
    else:
        return None
    return None if values is None else _CategoryCondition(feature, values, excluded, excluded)


def _category_node(condition):
    """输出字段在前的集合条件，显式表达独立的缺失路由。"""
    name = ast.Name(id=condition.feature, ctx=ast.Load())

    def missing_node(method):
        syntax = ast.Call(func=ast.Attribute(value=name, attr=method, ctx=ast.Load()), args=[], keywords=[])
        return VariableNode(_ast_unparse(syntax), syntax)

    if not condition.values and condition.include_missing != condition.excluded:
        return missing_node("notna" if condition.excluded else "isna")
    syntax = ast.Compare(left=name, ops=[ast.NotIn() if condition.excluded else ast.In()], comparators=[ast.List(elts=[ast.Constant(value) for value in condition.values], ctx=ast.Load())])
    result = VariableNode(_ast_unparse(syntax), syntax)
    if condition.excluded and not condition.include_missing:
        result = BinaryOpNode(result, missing_node("notna"), "&")
    elif not condition.excluded and condition.include_missing:
        result = BinaryOpNode(result, missing_node("isna"), "|")
    return result


def _merge_category_conditions(node):
    """合并同一布尔层级的同字段类别条件，保留其他字段及必要分组。"""
    if not isinstance(node, BinaryOpNode) or node.op not in ("&", "|"):
        return node
    terms, conditions, positions = [], {}, {}
    for term in _flatten_nodes(node, node.op):
        condition = _category_condition(term)
        if condition is None:
            terms.append(term)
            continue
        feature = condition.feature
        if feature not in conditions:
            positions[feature] = len(terms)
            conditions[feature] = condition
            terms.append(term)
        else:
            merged = _combine_categories(conditions[feature], condition, node.op)
            if merged is None:
                terms.append(term)
            else:
                conditions[feature] = merged
                terms[positions[feature]] = _category_node(merged)
    result = terms[0]
    for term in terms[1:]:
        result = BinaryOpNode(result, term, node.op)
    return result


class ExprParser:
    """表达式解析器，将字符串解析为 AST。"""

    def __init__(self, expr: str):
        self.expr = expr
        self.variables: List[str] = []
        self.quoted_names = {}

    def parse(self) -> ExprNode:
        """解析表达式字符串为 AST。"""
        # 预处理表达式
        processed = self._preprocess(self.expr)

        # 使用 AST 解析
        try:
            tree = ast.parse(processed, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in self.quoted_names:
                    node.id = self.quoted_names[node.id]
            return self._visit(tree.body)
        except SyntaxError:
            # 如果解析失败，返回原子节点
            return VariableNode(self.expr)

    def _preprocess(self, expr: str) -> str:
        """预处理表达式：将 pandas 逻辑运算符统一替换为 Python 布尔运算符。

        pandas eval 中 ``&`` / ``|`` / ``~`` 作为逐元素逻辑运算符使用，其相对比较运算符的
        优先级与 Python 位运算符相反——pandas 中比较运算符优先级更高， ``&`` / ``|`` 的行为
        类似 ``and`` / ``or`` 。为借助  :mod:`ast`  解析出与 pandas 一致的结合关系，必须在

        **所有括号层级**  将其替换为 ``and`` / ``or`` / ``not`` 。若仅在顶层替换，嵌套括号内的
        ``&`` / ``|`` 会被 ast 当作高优先级位运算符，与比较运算符结合成链式比较而错误折叠，
        丢失括号导致优化前后语义不一致。

        字符串字面量（单/双引号）与反引号列名内部的符号原样保留，避免误转换。
        """
        result = []
        i = 0
        n = len(expr)
        quote = None  # 当前所处的字符串/反引号字面量定界符，None 表示在字面量之外

        while i < n:
            char = expr[i]

            # 字面量内部：原样保留，直到遇到配对的定界符
            if quote is not None:
                result.append(char)
                if char == "\\" and i + 1 < n:
                    result.append(expr[i + 1])
                    i += 2
                    continue
                if char == quote:
                    quote = None
                i += 1
                continue

            if char == "`":
                end = expr.find("`", i + 1)
                if end == -1:
                    return expr
                placeholder = f"__rule_field_{len(self.quoted_names)}"
                while placeholder in self.expr:
                    placeholder += "_"
                self.quoted_names[placeholder] = expr[i : end + 1]
                result.append(placeholder)
                i = end + 1
                continue

            if char in ('"', "'"):
                quote = char
                result.append(char)
                i += 1
                continue

            if char == "~":
                result.append(" not ")
                i += 1
                continue

            if char == "&":
                result.append(" and ")
                # 同时兼容 && 写法
                i += 2 if (i + 1 < n and expr[i + 1] == "&") else 1
                continue

            if char == "|":
                result.append(" or ")
                i += 2 if (i + 1 < n and expr[i + 1] == "|") else 1
                continue

            result.append(char)
            i += 1

        # 去除首尾空白：ast.parse(mode='eval') 不允许表达式以空白开头（会被当作缩进），
        # 而前导的 ``~`` 被替换为 `` not `` 后会在行首引入空格
        return "".join(result).strip()

    def _visit(self, node: ast.AST) -> ExprNode:
        """访问 AST 节点。"""
        if isinstance(node, ast.BoolOp):
            # 处理布尔运算 (and, or)
            op = node.op
            if isinstance(op, ast.And):
                op_str = "&"
            elif isinstance(op, ast.Or):
                op_str = "|"
            elif isinstance(op, ast.Xor):
                op_str = "^"
            else:
                op_str = "&"

            # 处理多个操作数的情况 (a & b & c)
            result = self._visit(node.values[0])
            for value in node.values[1:]:
                result = BinaryOpNode(result, self._visit(value), op_str)
            return result

        elif isinstance(node, ast.BinOp):
            # 处理二元运算 (&, |, ^)
            op = node.op
            if isinstance(op, (ast.BitAnd, ast.And)):
                op_str = "&"
            elif isinstance(op, (ast.BitOr, ast.Or)):
                op_str = "|"
            elif isinstance(op, ast.BitXor):
                op_str = "^"
            else:
                return VariableNode(_ast_unparse(node), node)

            return BinaryOpNode(self._visit(node.left), self._visit(node.right), op_str)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            # 处理 NOT 操作
            return UnaryOpNode(self._visit(node.operand))

        elif isinstance(node, ast.Compare):
            # 处理比较表达式 (age > 18)
            comp_expr = _ast_unparse(node)
            self.variables.extend(self._extract_variables(comp_expr))
            return VariableNode(comp_expr, node)

        elif isinstance(node, ast.Name):
            # 处理变量名
            return VariableNode(node.id)

        elif isinstance(node, ast.Constant):
            # 处理常量
            return VariableNode(str(node.value))

        else:
            # 其他情况作为原子表达式处理
            try:
                comp_expr = _ast_unparse(node)
                return VariableNode(comp_expr, node)
            except Exception:
                return VariableNode(self.expr)

    def _extract_variables(self, expr: str) -> List[str]:
        """提取表达式中的变量名。"""
        # 预处理：移除 ~ 和括号等
        variables = []
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
        for match in re.finditer(pattern, expr):
            var = match.group(1)
            if var not in {"and", "or", "not", "True", "False", "None", "inf", "nan"}:
                variables.append(var)
        return list(set(variables))


class ExprOptimizer:
    """表达式优化器。"""

    def __init__(self):
        self.collapse_rules = []

    def optimize(self, expr: str) -> str:
        """优化表达式字符串。"""
        # 解析表达式
        parser = ExprParser(expr)
        ast_tree = parser.parse()

        # 简化表达式
        simplified = ast_tree.simplify()

        # 生成优化后的字符串
        return simplified.to_string()

    def beautify(self, expr: str) -> str:
        """美化表达式，使其更易读。"""
        parser = ExprParser(expr)
        ast_tree = parser.parse()
        return ast_tree.to_string()


# 全局优化器实例
_optimizer = ExprOptimizer()


def optimize_expr(expr: str) -> str:
    """简化规则表达式字符串。

    解析表达式为表达式树后，应用以下布尔代数定律做等价化简，并去除冗余括号、
    将 ``and`` / ``or`` 统一为 ``&`` / ``|`` ：

    - **幂等律（Idempotent）** ： ``A & A → A`` ， ``A | A → A``

    - **吸收律（Absorption）** ： ``A | (A & B) → A`` ， ``A & (A | B) → A``

    - **双重否定（Double negation）** ： ``~~A → A``

    - **数值区间交集** ： ``age > 18 & age > 25 & age <= 60 → age > 25 & age <= 60``，
      保留开闭边界；仅当同字段各条件均接收缺失值时保留 ``age.isna()`` 分支。

    - **类别集合合并** ：同字段的 ``in`` / ``not in`` / ``isin()`` 及字符串 ``==`` / ``!=``
      支持交集、并集、差集；缺失值单独计算，补集继续接收未列出的未知类别。

    .. note::
        数值区间仅合并同一字段与数值常量之间的比较，不做跨变量或字段类型的推断
        （如 ``age > 18`` 与 ``age >= 19`` 不会被判定为等价）。
        类别集合使用同类型的普通标量字面量；混合类型、包含 None、非有限数值、
        大整数或动态变量的集合保留原条件，避免猜测缺失标记和执行时的类型转换。

    :param expr: 原始规则表达式字符串，支持 ``&`` / ``|`` / ``~`` / ``and`` / ``or`` / ``not``
    :type expr: str
    :return: 化简后的等价表达式字符串

    **参考样例**

    >>> optimize_expr("(age > 18) & (age > 18)")    # 幂等律
    'age > 18'
    >>> optimize_expr("~~(age > 18)")               # 双重否定
    'age > 18'

    **引用**

    布尔代数化简定律：https://en.wikipedia.org/wiki/Boolean_algebra#Laws
    """
    return _optimizer.optimize(expr)


def beautify_expr(expr: str) -> str:
    """美化规则表达式字符串。

    在不改变逻辑的前提下规范化表达式的书写：将 ``and`` / ``or`` / ``not`` 统一为
    ``&`` / ``|`` / ``~`` 符号形式，按运算符结合律去除同级冗余括号，得到格式一致、
    便于展示与比较的表达式。与  :func:`optimize_expr`  的区别在于不做幂等/吸收等化简。

    :param expr: 原始规则表达式字符串
    :type expr: str
    :return: 美化后的等价表达式字符串

    **参考样例**

    >>> beautify_expr("(age > 18) & (income > 5000)")
    'age > 18 & income > 5000'
    >>> beautify_expr("age > 18 and income > 5000")
    'age > 18 & income > 5000'
    """
    return _optimizer.beautify(expr)


def get_expr_variables(expr: str) -> List[str]:
    """提取规则表达式中引用的变量（列）名。

    用正则匹配表达式中的标识符，剔除 ``and`` / ``or`` / ``not`` / ``True`` / ``False`` /
    ``None`` / ``inf`` / ``nan`` 等保留字，返回去重后的变量名列表。

    .. note::
        返回顺序  **不保证稳定** （基于集合去重）。如需有序且能正确处理含空格/中文/
        反引号的列名，请使用  :func:`hscredit.core.rules.get_columns_from_query`
        （返回去重并按字母排序的列表）。

    :param expr: 规则表达式字符串
    :type expr: str
    :return: 表达式引用的变量名列表（去重，顺序不保证）

    **参考样例**

    >>> sorted(get_expr_variables("(age > 18) & (income > 5000)"))
    ['age', 'income']
    """
    # 直接使用正则表达式提取变量名
    variables = set()
    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
    for match in re.finditer(pattern, expr):
        var = match.group(1)
        # 排除 Python 关键字和运算符
        if var not in {"and", "or", "not", "True", "False", "None", "inf", "nan"}:
            # 排除以开头的函数或方法调用
            # 检查是否是表达式中的比较运算符的一部分
            variables.add(var)
    return list(variables)
