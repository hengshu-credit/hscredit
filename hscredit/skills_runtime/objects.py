"""Agent Skills 同进程对象注册表。"""

from typing import Any, Mapping, Optional

from .errors import SkillExecutionError


class ObjectRegistry:
    """按精确引用名称保存调用方提供的 Python 对象。"""

    def __init__(self, objects: Optional[Mapping[str, Any]] = None) -> None:
        self._objects = dict(objects or {})

    def register(self, ref: str, value: Any) -> None:
        """登记对象并拒绝空引用或覆盖。"""
        if not isinstance(ref, str) or not ref.strip():
            raise SkillExecutionError(code="SCHEMA_INVALID", message="对象引用必须是非空字符串", field="ref")
        if ref in self._objects:
            raise SkillExecutionError(code="SCHEMA_INVALID", message=f"对象引用“{ref}”已经登记", field="ref")
        self._objects[ref] = value

    def resolve(self, ref: str) -> Any:
        """解析一个精确对象引用。"""
        try:
            return self._objects[ref]
        except KeyError as exc:
            raise SkillExecutionError(
                code="OBJECT_REF_NOT_FOUND",
                message=f"未找到对象引用“{ref}”",
                field="ref",
                cause=exc,
            ) from exc

    def contains(self, ref: str) -> bool:
        """返回对象引用是否存在。"""
        return ref in self._objects
