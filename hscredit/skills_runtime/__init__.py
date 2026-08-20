"""Agent Skills 公共运行时。"""

import sys
from typing import Any, Mapping, Optional

from .. import __version__
from .artifacts import ArtifactTransaction
from .contracts import ExecutionContext, validate_request
from .errors import SkillExecutionError
from .io import InputResolver
from .objects import ObjectRegistry
from .registry import DEFAULT_REGISTRY, OperationRegistry, ensure_default_operations


def execute_skill(
    skill: str,
    request: Mapping[str, Any],
    objects: Optional[Mapping[str, Any]] = None,
    *,
    registry: Optional[OperationRegistry] = None,
) -> dict:
    """校验并执行一个已登记的 Skill 操作。"""
    active_registry = registry or ensure_default_operations()
    normalized = validate_request(skill, request, active_registry)
    spec = active_registry.get(skill, normalized.operation)
    try:
        object_registry = ObjectRegistry(objects)
        resolver = InputResolver(object_registry)
        with ArtifactTransaction(normalized.output) as transaction:
            context = ExecutionContext(
                request=normalized,
                objects=objects or {},
                resolver=resolver,
                artifacts=transaction,
            )
            payload = spec.handler(context)
            result = {
                "status": "success",
                "operation": normalized.operation,
                "summary": payload.get("summary", {}),
                "artifacts": list(transaction.artifacts),
                "warnings": list(payload.get("warnings", [])),
                "environment": {
                    "mode": normalized.environment.mode,
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "hscredit": __version__,
                    "extras": list(spec.extras),
                },
            }
    except SkillExecutionError:
        raise
    except Exception as exc:
        raise SkillExecutionError(
            code="HSCREDIT_EXECUTION_FAILED",
            message=f"hscredit 操作“{normalized.operation}”执行失败：{exc}",
            cause=exc,
        ) from exc
    if not isinstance(result, dict):
        raise SkillExecutionError(
            code="HSCREDIT_EXECUTION_FAILED",
            message=f"操作“{normalized.operation}”返回了不支持的结果类型 {type(result).__name__}",
        )
    return result


__all__ = ["execute_skill", "SkillExecutionError"]
