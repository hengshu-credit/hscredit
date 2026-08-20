"""Agent Skills 版本化请求与执行上下文。"""

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .errors import SkillExecutionError


_COMMON_REQUEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "operation"],
    "properties": {
        "version": {"const": "1"},
        "operation": {"type": "string", "minLength": 1},
        "inputs": {"type": "object"},
        "parameters": {"type": "object"},
        "output": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "directory": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "overwrite": {"type": "boolean"},
                "format": {"type": "string", "minLength": 1},
            },
        },
        "environment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"enum": ["isolated", "current"]},
                "reuse": {"type": "boolean"},
                "install_missing": {"type": "boolean"},
            },
        },
    },
}


@dataclass(frozen=True)
class OutputSpec:
    """产物目录和覆盖策略。"""

    directory: str
    name: str
    overwrite: bool = False
    format: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentSpec:
    """依赖运行环境策略。"""

    mode: str
    reuse: bool
    install_missing: bool


@dataclass(frozen=True)
class SkillRequest:
    """校验和默认值归一化后的 Skill 请求。"""

    skill: str
    version: str
    operation: str
    inputs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    output: OutputSpec
    environment: EnvironmentSpec


@dataclass(frozen=True)
class ExecutionContext:
    """传给操作适配器的最小执行上下文。"""

    request: SkillRequest
    objects: Mapping[str, Any]
    resolver: Any = None
    artifacts: Any = None


def _validation_field(error) -> Optional[str]:
    path = [str(part) for part in error.absolute_path]
    if path:
        return ".".join(path)
    match = re.search(r"'([^']+)' was unexpected", error.message)
    return match.group(1) if match else None


def _validate(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - 仅在未安装 skills extra 时触发
        raise SkillExecutionError(
            code="DEPENDENCY_MISSING",
            message="缺少 Skills 请求校验依赖，请安装 hscredit[skills]",
            cause=exc,
        ) from exc

    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.absolute_path))
    if not errors:
        return
    error = errors[0]
    raise SkillExecutionError(
        code="SCHEMA_INVALID",
        message=f"Skill 请求不符合 Schema：{error.message}",
        field=_validation_field(error),
        cause=error,
    ) from error


def validate_request(skill: str, request: Mapping[str, Any], registry) -> SkillRequest:
    """校验公共信封和操作参数，并应用安全默认值。"""
    if not isinstance(request, Mapping):
        raise SkillExecutionError(code="SCHEMA_INVALID", message="Skill 请求必须是 JSON 对象")
    _validate(request, _COMMON_REQUEST_SCHEMA)

    operation = str(request["operation"])
    spec = registry.get(skill, operation)
    parameters = dict(request.get("parameters", {}))
    if spec.parameter_schema is not None:
        _validate(parameters, spec.parameter_schema)

    output_raw = dict(request.get("output", {}))
    environment_raw = dict(request.get("environment", {}))
    mode = environment_raw.get("mode", "isolated")
    install_missing = environment_raw.get("install_missing", mode == "isolated")
    return SkillRequest(
        skill=skill,
        version=str(request["version"]),
        operation=operation,
        inputs=dict(request.get("inputs", {})),
        parameters=parameters,
        output=OutputSpec(
            directory=output_raw.get("directory", "."),
            name=output_raw.get("name", operation),
            overwrite=output_raw.get("overwrite", False),
            format=output_raw.get("format"),
        ),
        environment=EnvironmentSpec(
            mode=mode,
            reuse=environment_raw.get("reuse", True),
            install_missing=install_missing,
        ),
    )
