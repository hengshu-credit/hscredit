"""用户可见异常信息中文化测试."""

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from hscredit.core.feature_engineering import NumExprDerive
from hscredit.core.rules import Rule


def test_all_explicit_raise_messages_contain_chinese():
    package_root = Path(__file__).resolve().parents[2] / "hscredit"
    english_only_messages = []

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            if not isinstance(node.exc, ast.Call) or not node.exc.args:
                continue

            message_node = node.exc.args[0]
            literal_parts = []
            if isinstance(message_node, ast.Constant) and isinstance(message_node.value, str):
                literal_parts.append(message_node.value)
            elif isinstance(message_node, ast.JoinedStr):
                literal_parts.extend(
                    value.value
                    for value in message_node.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )

            message = "".join(literal_parts)
            if (
                message
                and re.search(r"[A-Za-z]", message)
                and not re.search(r"[\u4e00-\u9fff]", message)
            ):
                english_only_messages.append(f"{path}:{node.lineno}: {message}")

    assert english_only_messages == []


def test_feature_derivation_error_is_chinese():
    with pytest.raises(ValueError, match="特征衍生规则不能为空"):
        NumExprDerive(derivings=[]).fit(pd.DataFrame({"x": [1]}))


def test_rule_input_error_is_chinese():
    with pytest.raises(Exception, match="只能对 DataFrame"):
        Rule("x > 0").predict([1, 2, 3])
