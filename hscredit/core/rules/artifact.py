"""结构化规则制品：执行表达式、分箱语义和验证来源。"""

import ast
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..binning.spec import BinSpec


def _encode(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__hscredit_float__": str(value)}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__hscredit_float__"}:
            return float(value["__hscredit_float__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


@dataclass
class RuleArtifact:
    """可审计的规则语义包，不包含训练数据；JSON 不执行反序列化代码。"""

    expression: str
    bins: tuple = ()
    preprocessing_version: str = "raw-v1"
    target_spec: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    validation: str = "未验证"
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError(f"不支持的规则制品版本: {self.version}")
        from .rule import get_columns_from_query

        self.dependencies = tuple(get_columns_from_query(self.expression))

    @property
    def ast(self):
        """规范 AST 表示；字段占位符只用于解析，不作为执行表达式。"""
        from .rule import _replace_backtick_columns

        return ast.dump(ast.parse(_replace_backtick_columns(self.expression)[0], mode="eval"))

    def to_rule(self, **kwargs):
        from .rule import Rule

        rule = Rule(self.expression, **kwargs)
        rule.artifact_ = self
        return rule

    def predict(self, data):
        return self.to_rule().predict(data)

    def to_dict(self):
        return _encode({"format": "hscredit-rule", **asdict(self), "dependencies": self.dependencies, "ast": self.ast})

    @classmethod
    def from_dict(cls, payload):
        data = _decode(dict(payload))
        if data.pop("format", None) != "hscredit-rule":
            raise ValueError("规则制品格式无效")
        dependencies = data.pop("dependencies", None)
        tree = data.pop("ast", None)
        specs = []
        for value in data.get("bins", []):
            value["values"] = tuple(value.get("values", ()))
            value["exclude_values"] = tuple(value.get("exclude_values", ()))
            value["include_values"] = tuple(value.get("include_values", ()))
            value["known_values"] = tuple(value.get("known_values", ()))
            specs.append(BinSpec(**value))
        data["bins"] = tuple(specs)
        artifact = cls(**data)
        if dependencies is not None and tuple(dependencies) != artifact.dependencies:
            raise ValueError("规则制品字段依赖与表达式不一致")
        if tree is not None and tree != artifact.ast:
            raise ValueError("规则制品 AST 与表达式不一致")
        return artifact

    def save(self, path):
        """先完成 JSON 校验与临时写入，再原子替换单个制品。"""
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{uuid4().hex}.{target.name}")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
