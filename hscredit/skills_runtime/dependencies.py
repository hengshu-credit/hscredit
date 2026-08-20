"""Agent Skills 受控依赖和隔离环境规划。"""

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .errors import SkillExecutionError


_OPERATION_EXTRAS = {
    "hsbin": {"*": ("skills",)},
    "hsreport": {"*": ("skills",)},
}


@dataclass(frozen=True)
class EnvironmentPlan:
    """依赖安装和解释器选择计划。"""

    mode: str
    extras: Tuple[str, ...]
    install_missing: bool
    reuse: bool


def _contains_object_ref(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("kind") == "object_ref":
            return True
        return any(_contains_object_ref(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_object_ref(item) for item in value)
    return False


def resolve_required_extras(skill: str, operation: str, request: Mapping[str, Any]) -> Tuple[str, ...]:
    """只从内部映射计算允许安装的 extras。"""
    skill_map = _OPERATION_EXTRAS.get(skill)
    if skill_map is None:
        raise SkillExecutionError(code="OPERATION_NOT_ALLOWED", message=f"未登记 Skill“{skill}”")
    extras = set(skill_map.get(operation, skill_map.get("*", ())))
    return tuple(sorted(extras))


def environment_key(source: str, extras: Sequence[str], python_version: str) -> str:
    """生成与顺序无关的短环境键。"""
    payload = {
        "source": str(source),
        "extras": sorted(set(str(extra) for extra in extras)),
        "python": str(python_version),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def resolve_cache_root() -> Path:
    """返回跨平台的 hscredit Skill 用户缓存目录。"""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "hscredit" / "skills" / "envs").resolve()


def plan_environment(
    skill: str,
    operation: str,
    request: Mapping[str, Any],
    *,
    missing_extras: Sequence[str] = (),
) -> EnvironmentPlan:
    """根据显式授权和对象边界规划执行环境。"""
    environment = request.get("environment", {})
    if not isinstance(environment, Mapping):
        raise SkillExecutionError(code="SCHEMA_INVALID", message="environment 必须是 JSON 对象", field="environment")
    mode = environment.get("mode", "isolated")
    if mode not in {"isolated", "current"}:
        raise SkillExecutionError(code="SCHEMA_INVALID", message=f"不支持的环境模式“{mode}”", field="environment.mode")
    install_missing = environment.get("install_missing", mode == "isolated")
    reuse = environment.get("reuse", True)
    extras = resolve_required_extras(skill, operation, request)

    if mode == "isolated" and _contains_object_ref(request.get("inputs", {})):
        raise SkillExecutionError(
            code="OBJECT_REF_REQUIRES_CURRENT_ENV",
            message="object_ref 不能传入隔离解释器；请使用 environment.mode=current 或先保存为可信制品",
            field="environment.mode",
        )
    if mode == "current" and missing_extras and install_missing is not True:
        missing = "、".join(sorted(set(str(extra) for extra in missing_extras)))
        raise SkillExecutionError(
            code="DEPENDENCY_MISSING",
            message=f"当前环境缺少依赖组：{missing}；如需自动安装请显式设置 install_missing=true",
            field="environment.install_missing",
        )
    return EnvironmentPlan(
        mode=mode,
        extras=extras,
        install_missing=bool(install_missing),
        reuse=bool(reuse),
    )
