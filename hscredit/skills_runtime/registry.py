"""Agent Skills 操作白名单注册表。"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .errors import SkillExecutionError


def _default_parameter_schema() -> Mapping[str, Any]:
    return {"type": "object"}


@dataclass(frozen=True)
class OperationSpec:
    """一个可被 Skill 调用的明确操作。"""

    skill: str
    name: str
    handler: Callable[[Any], dict]
    extras: Tuple[str, ...] = ()
    parameter_schema: Optional[Mapping[str, Any]] = field(default_factory=_default_parameter_schema)


class OperationRegistry:
    """按 Skill 和操作名隔离的精确白名单。"""

    def __init__(self) -> None:
        self._operations: Dict[Tuple[str, str], OperationSpec] = {}

    def register(self, spec: OperationSpec) -> None:
        """登记操作，并拒绝覆盖已有定义。"""
        key = (spec.skill, spec.name)
        if key in self._operations:
            raise SkillExecutionError(
                code="OPERATION_NOT_ALLOWED",
                message=f"Skill“{spec.skill}”的操作“{spec.name}”已经登记",
            )
        self._operations[key] = spec

    def get(self, skill: str, operation: str) -> OperationSpec:
        """获取精确匹配的操作。"""
        try:
            return self._operations[(skill, operation)]
        except KeyError as exc:
            raise SkillExecutionError(
                code="OPERATION_NOT_ALLOWED",
                message=f"Skill“{skill}”不允许调用操作“{operation}”",
                field="operation",
                cause=exc,
            ) from exc

    def list_operations(self, skill: str) -> Tuple[str, ...]:
        """按登记顺序列出 Skill 可调用操作。"""
        return tuple(name for registered_skill, name in self._operations if registered_skill == skill)


DEFAULT_REGISTRY = OperationRegistry()
_DEFAULT_OPERATIONS_LOADED = False


def ensure_default_operations() -> OperationRegistry:
    """惰性登记内置操作，避免导入运行时即加载绘图和报告重模块。"""
    global _DEFAULT_OPERATIONS_LOADED
    if not _DEFAULT_OPERATIONS_LOADED:
        from .operations import register_operations

        register_operations(DEFAULT_REGISTRY)
        _DEFAULT_OPERATIONS_LOADED = True
    return DEFAULT_REGISTRY
