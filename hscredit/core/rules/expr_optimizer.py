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
from typing import Set, List, Optional, Union


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

    def __init__(self, expr: str):
        super().__init__()
        self.expr = expr

    def get_variables(self) -> Set[str]:
        # 使用正则提取变量名
        variables = set()
        # 匹配形如 "age > 18" 中的变量名
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        for match in re.finditer(pattern, self.expr):
            var = match.group(1)
            # 排除 Python 关键字
            if var not in {'and', 'or', 'not', 'True', 'False', 'None'}:
                variables.add(var)
        return variables

    def to_string(self, parent_op: Optional[str] = None) -> str:
        return self.expr

    def simplify(self):
        return self


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
        symbols = {'&': '&', '|': '|', '^': '^', 'and': '&', 'or': '|'}
        return symbols.get(self.op, self.op)

    def need_parens(self, parent_op: str) -> bool:
        """判断是否需要括号。"""
        # 同级运算符不需要括号（满足结合律）
        if parent_op == self.op:
            return False
        # 不同运算符默认需要括号以避免优先级问题
        return True

    def normalize_expr(self, expr: str) -> str:
        """规范化表达式字符串以便比较。"""
        # 移除多余空格，统一小写，将 and/or 转换为 &/|
        result = ' '.join(expr.split()).lower()
        result = result.replace('and', '&')
        result = result.replace('or', '|')
        return result

    def simplify(self):
        """简化二元运算表达式。"""
        # 递归简化子节点
        self.left = self.left.simplify()
        self.right = self.right.simplify()

        # 规范化表达式以便比较
        left_expr = self.normalize_expr(self.left.to_string())
        right_expr = self.normalize_expr(self.right.to_string())

        # 幂等律: A & A = A, A | A = A
        if left_expr == right_expr:
            return self.left

        # 吸收率: A | (A & B) = A
        # 检查 left 是否包含 right（即 left 中是否有 right 这个子表达式）
        if self.op == '|':
            if self._contains_expr(self.left, right_expr):
                return self.right

        # 吸收率: A & (A | B) = A
        # 检查 right 是否包含 left
        if self.op == '&':
            if self._contains_expr(self.right, left_expr):
                return self.left

        # 分配率展开 (可选): (A | B) & (A | C) = A | (B & C)
        # 这个比较复杂，暂时不实现

        return self

    def _contains_expr(self, node: ExprNode, target: str) -> bool:
        """检查节点树中是否包含目标表达式。"""
        if isinstance(node, VariableNode):
            node_expr = self.normalize_expr(node.to_string())
            return node_expr == target
        elif isinstance(node, BinaryOpNode):
            return (self._contains_expr(node.left, target) or
                    self._contains_expr(node.right, target))
        elif isinstance(node, UnaryOpNode):
            return self._contains_expr(node.operand, target)
        return False


class UnaryOpNode(ExprNode):
    """一元运算符节点 (NOT)。"""

    def __init__(self, operand: ExprNode, op: str = 'not'):
        super().__init__()
        self.operand = operand
        self.op = op
        operand.parent = self

    def get_variables(self) -> Set[str]:
        return self.operand.get_variables()

    def to_string(self, parent_op: Optional[str] = None) -> str:
        operand_str = self.operand.to_string(self.op)
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
    """Python 3.8 兼容的 AST 转字符串函数 (ast.unparse 是 Python 3.9+).

    手动格式化常见 AST 节点类型。
    """
    if isinstance(node, ast.Compare):
        # e.g. age > 18
        left = _ast_unparse(node.left)
        parts = [left]
        for op, comparator in zip(node.ops, node.comparators):
            op_str = _get_op_symbol(op)
            parts.append(f" {op_str} {_ast_unparse(comparator)}")
        return "".join(parts)
    elif isinstance(node, ast.BinOp):
        left = _ast_unparse(node.left)
        right = _ast_unparse(node.right)
        op_str = _get_op_symbol(node.op)
        return f"{left} {op_str} {right}"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Call):
        # 处理函数调用，如 purpose.isin(["education", "business"])
        func = _ast_unparse(node.func)
        args = ", ".join(_ast_unparse(arg) for arg in node.args)
        if node.keywords:
            kwargs = ", ".join(f"{kw.arg}={_ast_unparse(kw.value)}" for kw in node.keywords)
            args = f"{args}, {kwargs}" if args else kwargs
        return f"{func}({args})"
    elif isinstance(node, ast.Attribute):
        # 处理属性访问，如 purpose.isin
        value = _ast_unparse(node.value)
        return f"{value}.{node.attr}"
    else:
        return ""


def _get_op_symbol(op):
    """获取比较/二元运算符的符号."""
    ops = {
        ast.Gt: ">",
        ast.Lt: "<",
        ast.GtE: ">=",
        ast.LtE: "<=",
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.And: "&",
        ast.Or: "|",
    }
    return ops.get(type(op), str(op))


class ExprParser:
    """表达式解析器，将字符串解析为 AST。"""

    def __init__(self, expr: str):
        self.expr = expr
        self.variables: List[str] = []

    def parse(self) -> ExprNode:
        """解析表达式字符串为 AST。"""
        # 预处理表达式
        processed = self._preprocess(self.expr)

        # 使用 AST 解析
        try:
            tree = ast.parse(processed, mode='eval')
            return self._visit(tree.body)
        except SyntaxError:
            # 如果解析失败，返回原子节点
            return VariableNode(self.expr)

    def _preprocess(self, expr: str) -> str:
        """预处理表达式。"""
        # 将 ~ 转换为 not
        expr = expr.replace('~', 'not ')

        # 智能替换 & 和 | 为 and 和 or
        # 需要跟踪括号深度来正确处理
        import re

        result = []
        paren_depth = 0

        i = 0
        while i < len(expr):
            char = expr[i]

            if char == '(':
                paren_depth += 1
                result.append(char)
            elif char == ')':
                paren_depth -= 1
                result.append(char)
            elif paren_depth == 0 and i + 1 < len(expr):
                # 只在顶层括号外替换
                if char == '&' and expr[i+1] == '&':
                    result.append('and')
                    i += 1  # skip next &
                elif char == '|' and expr[i+1] == '|':
                    result.append('or')
                    i += 1  # skip next |
                elif char == '&':
                    # 检查是否是运算符 (前后有空格或括号或比较运算符)
                    prev_ok = i == 0 or expr[i-1] in ' ('
                    next_ok = i + 1 >= len(expr) or expr[i+1] in ' )'
                    if prev_ok or next_ok:
                        result.append('and')
                    else:
                        result.append(char)
                elif char == '|':
                    prev_ok = i == 0 or expr[i-1] in ' ('
                    next_ok = i + 1 >= len(expr) or expr[i+1] in ' )'
                    if prev_ok or next_ok:
                        result.append('or')
                    else:
                        result.append(char)
                else:
                    result.append(char)
            else:
                result.append(char)

            i += 1

        return ''.join(result)

    def _visit(self, node: ast.AST) -> ExprNode:
        """访问 AST 节点。"""
        if isinstance(node, ast.BoolOp):
            # 处理布尔运算 (and, or)
            op = node.op
            if isinstance(op, ast.And):
                op_str = '&'
            elif isinstance(op, ast.Or):
                op_str = '|'
            elif isinstance(op, ast.Xor):
                op_str = '^'
            else:
                op_str = '&'

            # 处理多个操作数的情况 (a & b & c)
            result = self._visit(node.values[0])
            for value in node.values[1:]:
                result = BinaryOpNode(result, self._visit(value), op_str)
            return result

        elif isinstance(node, ast.BinOp):
            # 处理二元运算 (&, |, ^)
            op = node.op
            if isinstance(op, (ast.BitAnd, ast.And)):
                op_str = '&'
            elif isinstance(op, (ast.BitOr, ast.Or)):
                op_str = '|'
            elif isinstance(op, ast.BitXor):
                op_str = '^'
            else:
                op_str = '&'

            return BinaryOpNode(self._visit(node.left), self._visit(node.right), op_str)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            # 处理 NOT 操作
            return UnaryOpNode(self._visit(node.operand))

        elif isinstance(node, ast.Compare):
            # 处理比较表达式 (age > 18)
            comp_expr = _ast_unparse(node)
            self.variables.extend(self._extract_variables(comp_expr))
            return VariableNode(comp_expr)

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
                return VariableNode(comp_expr)
            except Exception:
                return VariableNode(self.expr)

    def _extract_variables(self, expr: str) -> List[str]:
        """提取表达式中的变量名。"""
        # 预处理：移除 ~ 和括号等
        variables = []
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        for match in re.finditer(pattern, expr):
            var = match.group(1)
            if var not in {'and', 'or', 'not', 'True', 'False', 'None', 'inf', 'nan'}:
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
    将 ``and``/``or`` 统一为 ``&``/``|``：

    - **幂等律（Idempotent）**：``A & A → A``，``A | A → A``
    - **吸收律（Absorption）**：``A | (A & B) → A``，``A & (A | B) → A``
    - **双重否定（Double negation）**：``~~A → A``

    .. note::
        化简基于子表达式字符串的规范化比较，仅识别字面等价的原子条件，不做跨变量的
        逻辑推理（如 ``age > 18`` 与 ``age >= 19`` 不会被判定为等价）。

    :param expr: 原始规则表达式字符串，支持 ``&``/``|``/``~``/``and``/``or``/``not``
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

    在不改变逻辑的前提下规范化表达式的书写：将 ``and``/``or``/``not`` 统一为
    ``&``/``|``/``~`` 符号形式，按运算符结合律去除同级冗余括号，得到格式一致、
    便于展示与比较的表达式。与 :func:`optimize_expr` 的区别在于不做幂等/吸收等化简。

    :param expr: 原始规则表达式字符串
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

    用正则匹配表达式中的标识符，剔除 ``and``/``or``/``not``/``True``/``False``/
    ``None``/``inf``/``nan`` 等保留字，返回去重后的变量名列表。

    .. note::
        返回顺序 **不保证稳定**（基于集合去重）。如需有序且能正确处理含空格/中文/
        反引号的列名，请使用 :func:`hscredit.core.rules.get_columns_from_query`
        （返回去重并按字母排序的列表）。

    :param expr: 规则表达式字符串
    :return: 表达式引用的变量名列表（去重，顺序不保证）

    **参考样例**

    >>> sorted(get_expr_variables("(age > 18) & (income > 5000)"))
    ['age', 'income']
    """
    # 直接使用正则表达式提取变量名
    variables = set()
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    for match in re.finditer(pattern, expr):
        var = match.group(1)
        # 排除 Python 关键字和运算符
        if var not in {'and', 'or', 'not', 'True', 'False', 'None', 'inf', 'nan'}:
            # 排除以开头的函数或方法调用
            # 检查是否是表达式中的比较运算符的一部分
            variables.add(var)
    return list(variables)
