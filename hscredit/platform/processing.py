"""将可视化 fx 配置编译为 NumExprDerive 表达式，统一预览与执行语义。"""

import ast
import json
import math
from uuid import uuid4

import numpy as np
import pandas as pd

from hscredit.core.feature_engineering import NumExprDerive
from .storage import records

OPERATORS = [
    {"value": "+", "label": "加 / 文本拼接"},
    {"value": "-", "label": "减"},
    {"value": "*", "label": "乘"},
    {"value": "/", "label": "除"},
    {"value": "%", "label": "取余"},
    {"value": "==", "label": "等于"},
    {"value": "!=", "label": "不等于"},
    {"value": ">", "label": "大于"},
    {"value": ">=", "label": "大于等于"},
    {"value": "<", "label": "小于"},
    {"value": "<=", "label": "小于等于"},
    {"value": "and", "label": "并且"},
    {"value": "or", "label": "或者"},
]
FUNCTIONS = [
    {"name": "where", "label": "条件取值", "args": ["条件", "满足时", "不满足时"]},
    {"name": "clip", "label": "限制范围", "args": ["数值", "下限", "上限"]},
    {"name": "round", "label": "四舍五入", "args": ["数值", "小数位数"]},
    {"name": "minimum", "label": "逐行最小值", "args": ["数值一", "数值二"]},
    {"name": "maximum", "label": "逐行最大值", "args": ["数值一", "数值二"]},
    *[
        {"name": name, "label": label, "args": ["数值"]}
        for name, label in (
            ("abs", "绝对值"),
            ("sqrt", "平方根"),
            ("log", "自然对数"),
            ("exp", "指数"),
            ("floor", "向下取整"),
            ("ceil", "向上取整"),
            ("isnan", "是否为数值缺失"),
            ("isfinite", "是否为有限数值"),
            ("mean", "整列均值"),
            ("sum", "整列合计"),
            ("median", "整列中位数"),
            ("std", "整列标准差"),
            ("var", "整列方差"),
        )
    ],
]
_FUNCTIONS = {f["name"]: f for f in FUNCTIONS}
_OPERATORS = {o["value"] for o in OPERATORS}
_PRECEDENCE = {"or": 1, "and": 2, "==": 3, "!=": 3, ">": 4, ">=": 4, "<": 4, "<=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}


class DatasetConflict(ValueError):
    """数据已更新或被运行任务占用，不能基于旧版本覆盖。"""


def compile_expression(node, fields, aliases=None):
    """校验配置树并编译表达式；函数与运算符优先级由服务端决定。"""
    aliases = aliases or {}
    seen = 0

    def compile_node(value, depth=0):
        nonlocal seen
        seen += 1
        if depth > 8 or seen > 64:
            raise ValueError("单条表达式最多 64 个节点、8 层嵌套")
        if not isinstance(value, dict):
            raise ValueError("请选择字段、常量、函数或组合表达式")
        kind = value.get("kind")
        if kind == "EMPTY":
            raise ValueError("表达式中仍有未配置的内容")
        if kind == "UNARY":
            op = value.get("operator")
            expression, operand_type = compile_node(value.get("operand"), depth + 1)
            if op == "not" and operand_type == "boolean":
                return f"(not {expression})", "boolean"
            if op in {"+", "-"} and operand_type == "numeric":
                return f"({op}{expression})", "numeric"
            raise ValueError("取反需要布尔值，正负号需要数值")
        if kind == "FIELD":
            name = value.get("name")
            if not isinstance(name, str) or name not in fields:
                raise ValueError(f"字段不存在或尚未生成：{name}")
            return f"col({json.dumps(aliases.get(name, name), ensure_ascii=False)})", fields[name]
        if kind == "LITERAL":
            literal_type, literal = value.get("value_type"), value.get("value")
            if literal_type == "MISSING":
                return "nan", "numeric"
            if literal_type == "BOOLEAN":
                if not isinstance(literal, bool):
                    raise ValueError("布尔常量必须选择真或假")
                return "True" if literal else "False", "boolean"
            if literal_type == "STRING":
                if not isinstance(literal, str) or len(literal) > 500:
                    raise ValueError("文本常量最多 500 个字符")
                return json.dumps(literal, ensure_ascii=False), "string"
            if literal_type == "NUMBER":
                if isinstance(literal, bool) or not isinstance(literal, (int, float)) or not math.isfinite(literal) or abs(literal) > 1e15:
                    raise ValueError("数值常量必须为绝对值不超过 10¹⁵ 的有限数字")
                return repr(literal), "numeric"
            raise ValueError("请选择常量类型")
        if kind == "FUNCTION":
            name, args = value.get("name"), value.get("args")
            if not isinstance(name, str) or name not in _FUNCTIONS:
                raise ValueError("该函数不在 NumExprDerive 配置函数列表中")
            if not isinstance(args, list) or len(args) != len(_FUNCTIONS[name]["args"]):
                raise ValueError(f"{name} 的参数数量不正确")
            compiled = [compile_node(arg, depth + 1) for arg in args]
            if name == "where":
                if compiled[0][1] != "boolean":
                    raise ValueError("条件取值的第一个参数必须为布尔条件")
                a, b = compiled[1][1], compiled[2][1]
                if a != b:
                    raise ValueError("条件取值的两个结果必须同为数值、文本或布尔，避免隐式转换导致缺失值丢失")
                output_type = a
            else:
                if any(t != "numeric" for _, t in compiled):
                    raise ValueError(f"{_FUNCTIONS[name]['label']}需要数值参数")
                output_type = "boolean" if name in {"isnan", "isfinite"} else "numeric"
            return f"{name}({', '.join(text for text, _ in compiled)})", output_type
        if kind == "OPERATION":
            terms = value.get("terms")
            if not isinstance(terms, list) or not 1 <= len(terms) <= 20 or any(not isinstance(term, dict) for term in terms):
                raise ValueError("组合表达式需要 1 到 20 个运算项")
            values, operators = [], []

            def apply_operator():
                op = operators.pop()
                right, left = values.pop(), values.pop()
                if op in {"and", "or"}:
                    if left[1] != "boolean" or right[1] != "boolean":
                        raise ValueError("并且、或者两侧需要布尔条件")
                    output_type = "boolean"
                elif op in {"==", "!=", ">", ">=", "<", "<="}:
                    if left[1] != right[1]:
                        raise ValueError("比较运算两侧的类型必须一致")
                    output_type = "boolean"
                elif op == "+" and left[1] == right[1] == "string":
                    output_type = "string"
                else:
                    if left[1] != "numeric" or right[1] != "numeric":
                        raise ValueError("算术运算需要数值；文本仅支持加号拼接")
                    output_type = "numeric"
                values.append((f"({left[0]} {op} {right[0]})", output_type))

            for index, term in enumerate(terms):
                if index:
                    op = term.get("operator")
                    if not isinstance(op, str) or op not in _OPERATORS:
                        raise ValueError("请选择有效的运算符")
                    while operators and _PRECEDENCE[operators[-1]] >= _PRECEDENCE[op]:
                        apply_operator()
                    operators.append(op)
                values.append(compile_node(term.get("operand"), depth + 1))
            while operators:
                apply_operator()
            return values[0]
        raise ValueError("不支持的表达式节点类型")

    expression, result_type = compile_node(node)
    if len(expression) > 10000:
        raise ValueError("表达式过长，请减少组合项")
    return expression, result_type


def field_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "string"


def referenced_fields(node):
    """提取配置中引用的字段，仅用于前后对比展示。"""
    if node["kind"] == "FIELD":
        return [node["name"]]
    if node["kind"] == "UNARY":
        return referenced_fields(node["operand"])
    if node["kind"] == "FUNCTION":
        return [name for arg in node["args"] for name in referenced_fields(arg)]
    if node["kind"] == "OPERATION":
        return [name for term in node["terms"] for name in referenced_fields(term["operand"])]
    return []


def compile_rules(fields, rules):
    """顺序校验字段配置，供计算和文本编辑共用，避免两种编辑方式语义不同。"""
    fields = dict(fields)
    aliases, derivings, plan = {}, [], []
    token = uuid4().hex
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError("加工步骤格式不正确")
        target, mode = rule.get("target"), rule.get("mode")
        if not isinstance(target, str) or not target.strip() or target != target.strip() or len(target) > 150:
            raise ValueError(f"第 {index + 1} 步：目标字段需要 1 到 150 个字符，且不能以空格开头或结尾")
        if mode not in {"add", "replace"}:
            raise ValueError("加工方式必须选择新增字段或更新字段")
        if mode == "add" and target in fields:
            raise ValueError(f"字段 {target} 已存在，请选择更新字段")
        if mode == "replace" and target not in fields:
            raise ValueError(f"更新字段 {target} 不存在")
        try:
            display, output_type = compile_expression(rule.get("expression"), fields)
            expression, _ = compile_expression(rule.get("expression"), fields, aliases)
        except ValueError as exc:
            raise ValueError(f"第 {index + 1} 步（{target}）：{exc}") from exc
        temporary = f"_hscredit_processing_{token}_{index}"
        derivings.append((temporary, expression))
        aliases[target] = temporary
        fields[target] = output_type
        plan.append({"target": target, "mode": mode, "expression": display, "type": output_type, "references": list(dict.fromkeys(referenced_fields(rule["expression"])))})
    if len(fields) > 2000:
        raise ValueError("加工后不能超过 2000 个字段")
    return fields, aliases, derivings, plan


def parse_expression_text(expression, fields):
    """把受限表达式解析回配置树；只解析 AST，不执行用户输入。"""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 10000:
        raise ValueError("表达式不能为空，且长度不能超过 10000 个字符")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("表达式语法不正确，请检查括号、引号和运算符") from exc
    if len(list(ast.walk(tree))) > 256:
        raise ValueError("表达式过于复杂")
    binary = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.BitAnd: "and", ast.BitOr: "or"}
    comparisons = {ast.Eq: "==", ast.NotEq: "!=", ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<="}

    def operation(left, op, right):
        return {"kind": "OPERATION", "terms": [{"operand": left}, {"operator": op, "operand": right}]}

    def convert(node, depth=0):
        if depth > 8:
            raise ValueError("表达式最多支持 8 层嵌套")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                kind = "BOOLEAN"
            elif isinstance(node.value, (int, float)):
                kind = "NUMBER"
            elif isinstance(node.value, str):
                kind = "STRING"
            else:
                raise ValueError("不支持该常量，缺失值请使用 nan")
            return {"kind": "LITERAL", "value_type": kind, "value": node.value}
        if isinstance(node, ast.Name):
            if node.id in fields:
                return {"kind": "FIELD", "name": node.id}
            if node.id == "nan":
                return {"kind": "LITERAL", "value_type": "MISSING", "value": None}
            raise ValueError(f'字段不存在：{node.id}；特殊字段名请使用 col("字段名")')
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "nan":
            return {"kind": "LITERAL", "value_type": "MISSING", "value": None}
        if isinstance(node, ast.UnaryOp) and type(node.op) in {ast.USub, ast.UAdd, ast.Not, ast.Invert}:
            operand = convert(node.operand, depth + 1)
            op = "-" if isinstance(node.op, ast.USub) else "+" if isinstance(node.op, ast.UAdd) else "not"
            if op in {"+", "-"} and operand.get("value_type") == "NUMBER":
                return {**operand, "value": -operand["value"] if op == "-" else operand["value"]}
            return {"kind": "UNARY", "operator": op, "operand": operand}
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return operation(convert(node.left, depth + 1), binary[type(node.op)], convert(node.right, depth + 1))
        if isinstance(node, ast.BoolOp):
            terms = [{"operand": convert(v, depth + 1)} for v in node.values]
            for term in terms[1:]:
                term["operator"] = "and" if isinstance(node.op, ast.And) else "or"
            return {"kind": "OPERATION", "terms": terms}
        if isinstance(node, ast.Compare):
            left, conditions = node.left, []
            for op, right in zip(node.ops, node.comparators):
                if type(op) not in comparisons:
                    raise ValueError("暂不支持该比较方式")
                conditions.append(operation(convert(left, depth + 1), comparisons[type(op)], convert(right, depth + 1)))
                left = right
            if len(conditions) == 1:
                return conditions[0]
            return {"kind": "OPERATION", "terms": [{"operand": item, **({"operator": "and"} if i else {})} for i, item in enumerate(conditions)]}
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "np" else None
            if node.keywords:
                raise ValueError("请使用位置参数，例如 round(金额, 2)")
            if name == "col" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return {"kind": "FIELD", "name": node.args[0].value}
            if name not in _FUNCTIONS:
                raise ValueError("仅支持配置面板中列出的函数，不允许任意 Python 调用")
            return {"kind": "FUNCTION", "name": name, "args": [convert(arg, depth + 1) for arg in node.args]}
        raise ValueError("表达式包含不支持的语法；请使用字段、常量、运算符和白名单函数")

    result = convert(tree.body)
    compiled, output_type = compile_expression(result, fields)
    return {"expression": result, "text": compiled, "type": output_type}


def process_frame(data, rules):
    """使用一次 NumExprDerive 顺序求值，临时字段最终映射回目标列。"""
    if not isinstance(rules, list) or not 1 <= len(rules) <= 30:
        raise ValueError("一次加工需要配置 1 到 30 个字段")
    fields, aliases, derivings, plan = compile_rules({name: field_type(data[name]) for name in data}, rules)
    try:
        transformed = NumExprDerive(derivings=derivings, trusted_python=False).fit_transform(data)
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError(f"表达式计算失败：{exc}") from exc
    result = transformed[[aliases.get(name, name) for name in fields]].copy()
    result.columns = list(fields)
    changed = list(dict.fromkeys(rule["target"] for rule in rules))
    stats = []
    for name in changed:
        series = result[name]
        missing = int(series.isna().sum())
        infinite = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()) if pd.api.types.is_numeric_dtype(series) else 0
        stats.append({"字段": name, "处理方式": "更新" if name in data else "新增", "处理前类型": str(data[name].dtype) if name in data else "—", "处理后类型": str(series.dtype), "缺失数": missing, "无穷值数": infinite, "唯一值数": int(series.nunique())})
    return result, plan, stats


def preview_result(data, result, plan, stats, revision):
    """全量执行后截取展示样本，整列聚合与保存使用相同输入范围。"""
    columns = list(dict.fromkeys([name for step in plan for name in step["references"] if name in data] + [p["target"] for p in plan]))
    row_name = "预览行号"
    while row_name in columns:
        row_name = "_" + row_name

    def rows(frame):
        frame = frame.head(20).astype(object).replace({np.inf: "正无穷", -np.inf: "负无穷"})
        frame.insert(0, row_name, range(1, len(frame) + 1))
        return records(frame)

    before = data.reindex(columns=columns)
    return {"revision": revision, "rows": len(result), "preview_rows": min(20, len(result)), "columns": columns, "before": rows(before), "after": rows(result[columns]), "statistics": stats, "plan": plan}
