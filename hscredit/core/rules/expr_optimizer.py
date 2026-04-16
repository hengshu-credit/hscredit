"""规则表达式优化器。

提供布尔表达式简化、优化和美化功能。
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
            comp_expr = ast.unparse(node)
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
                comp_expr = ast.unparse(node)
                return VariableNode(comp_expr)
            except:
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
    """简化表达式字符串。

    应用布尔代数定律简化表达式，移除冗余括号，消除双重否定等。

    :param expr: 原始表达式字符串
    :return: 简化后的表达式字符串

    **参考样例**

    >>> optimize_expr("(age > 18) & (age > 18)")
    'age > 18'
    >>> optimize_expr("~~(age > 18)")
    'age > 18'
    """
    return _optimizer.optimize(expr)


def beautify_expr(expr: str) -> str:
    """美化表达式字符串。

    生成格式规范、易读性好的表达式。

    :param expr: 原始表达式字符串
    :return: 美化后的表达式字符串

    **参考样例**

    >>> beautify_expr("(age > 18) & (income > 5000)")
    'age > 18 & income > 5000'
    """
    return _optimizer.beautify(expr)


def get_expr_variables(expr: str) -> List[str]:
    """获取表达式中使用的变量名列表。

    :param expr: 表达式字符串
    :return: 变量名列表

    **参考样例**

    >>> get_expr_variables("(age > 18) & (income > 5000)")
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
