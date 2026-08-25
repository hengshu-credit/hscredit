"""CI 测试跳过策略回归测试。"""

from types import SimpleNamespace

import pytest

from tests import conftest


DATABASE_TEST_ENVIRONMENTS = [
    "HSCREDIT_TEST_CLICKHOUSE_HOST",
    "HSCREDIT_TEST_HIVE_HOST",
    "HSCREDIT_TEST_IMPALA_HOST",
    "HSCREDIT_TEST_MAXCOMPUTE_PROJECT",
    "HSCREDIT_TEST_MONGODB_URI",
    "HSCREDIT_TEST_MYSQL_HOST",
    "HSCREDIT_TEST_ORACLE_DSN",
    "HSCREDIT_TEST_REDIS_URL",
    "HSCREDIT_TEST_STARROCKS_HOST",
]


def _run_strict_skip_policy(monkeypatch, reasons):
    reports = [
        SimpleNamespace(
            nodeid=f"tests/test_database/integration/test_{index}.py::test_real_service",
            longrepr=("test_integration.py", 1, f"Skipped: {reason}"),
        )
        for index, reason in enumerate(reasons)
    ]
    terminal = SimpleNamespace(
        stats={"skipped": reports},
        write_sep=lambda *args, **kwargs: None,
        write_line=lambda *args, **kwargs: None,
    )
    session = SimpleNamespace(
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: terminal),
        ),
        exitstatus=pytest.ExitCode.OK,
    )
    monkeypatch.setenv("HSCREDIT_STRICT_CI_SKIPS", "1")

    conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)

    return session.exitstatus


@pytest.mark.parametrize("environment_variable", DATABASE_TEST_ENVIRONMENTS)
def test_strict_ci_allows_database_integration_skip_when_service_is_not_configured(
    monkeypatch,
    environment_variable,
):
    """未配置真实数据库服务时，严格模式不得把条件跳过改判为失败。"""
    exitstatus = _run_strict_skip_policy(
        monkeypatch,
        [f"未配置 {environment_variable}"],
    )

    assert exitstatus == pytest.ExitCode.OK


def test_strict_ci_still_rejects_unexpected_skip(monkeypatch):
    """精确白名单外的跳过仍应让 CI 失败。"""
    exitstatus = _run_strict_skip_policy(monkeypatch, ["可选能力暂不可用"])

    assert exitstatus == pytest.ExitCode.TESTS_FAILED
