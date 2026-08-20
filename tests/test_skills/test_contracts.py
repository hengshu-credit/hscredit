"""Skills 请求契约与操作白名单测试。"""

import pytest

from hscredit.skills_runtime import execute_skill
from hscredit.skills_runtime.contracts import validate_request
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.skills_runtime.registry import OperationRegistry, OperationSpec


def test_registry_rejects_an_operation_from_another_skill():
    """防止一个 Skill 越权调用另一个 Skill 的操作。"""
    registry = OperationRegistry()
    registry.register(OperationSpec("hsbin", "feature_bin_stats", lambda context: {"status": "success"}))

    with pytest.raises(SkillExecutionError) as exc_info:
        registry.get("hsreport", "feature_bin_stats")

    assert exc_info.value.code == "OPERATION_NOT_ALLOWED"
    assert "feature_bin_stats" in str(exc_info.value)


def test_execute_skill_rejects_an_unknown_top_level_field(tmp_path):
    """防止未声明字段绕过公共请求 Schema。"""
    request = {
        "version": "1",
        "operation": "feature_bin_stats",
        "inputs": {},
        "parameters": {},
        "output": {"directory": str(tmp_path), "name": "stats", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
        "unexpected": True,
    }

    with pytest.raises(SkillExecutionError) as exc_info:
        execute_skill("hsbin", request)

    assert exc_info.value.code == "SCHEMA_INVALID"
    assert exc_info.value.field == "unexpected"


def test_validate_request_applies_isolated_defaults():
    """防止缺省请求意外修改当前 Python 环境或覆盖产物。"""
    registry = OperationRegistry()
    registry.register(OperationSpec("hsbin", "feature_bin_stats", lambda context: {"status": "success"}))

    request = validate_request(
        "hsbin",
        {"version": "1", "operation": "feature_bin_stats"},
        registry,
    )

    assert request.inputs == {}
    assert request.parameters == {}
    assert request.output.directory == "."
    assert request.output.name == "feature_bin_stats"
    assert request.output.overwrite is False
    assert request.environment.mode == "isolated"
    assert request.environment.reuse is True
    assert request.environment.install_missing is True


def test_validate_request_rejects_an_unsupported_protocol_version():
    """防止未知协议版本被按当前语义误执行。"""
    registry = OperationRegistry()
    registry.register(OperationSpec("hsbin", "feature_bin_stats", lambda context: {"status": "success"}))

    with pytest.raises(SkillExecutionError) as exc_info:
        validate_request(
            "hsbin",
            {"version": "2", "operation": "feature_bin_stats"},
            registry,
        )

    assert exc_info.value.code == "SCHEMA_INVALID"
    assert exc_info.value.field == "version"
