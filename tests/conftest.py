"""共享测试配置、演示数据路径与 CI 跳过策略."""

import os
from pathlib import Path
import site

import pytest

# tests/conftest.py -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HSCREDIT_DEMO_XLSX = PROJECT_ROOT / "examples" / "hscredit_yyp.xlsx"
_DATABASE_TEST_ENVIRONMENTS = frozenset(
    {
        "HSCREDIT_TEST_CLICKHOUSE_HOST",
        "HSCREDIT_TEST_HIVE_HOST",
        "HSCREDIT_TEST_IMPALA_HOST",
        "HSCREDIT_TEST_MAXCOMPUTE_PROJECT",
        "HSCREDIT_TEST_MONGODB_URI",
        "HSCREDIT_TEST_MYSQL_HOST",
        "HSCREDIT_TEST_ORACLE_DSN",
        "HSCREDIT_TEST_REDIS_URL",
        "HSCREDIT_TEST_STARROCKS_HOST",
    }
)
_ALLOWED_DATABASE_SKIP_REASONS = frozenset(
    f"未配置 {environment_variable}"
    for environment_variable in _DATABASE_TEST_ENVIRONMENTS
)

# 工作簿存在时不排除任何测试模块。
collect_ignore = []

# 以下历史脚本在模块导入阶段直接读取演示工作簿。工作簿存在时允许 pytest
# 收集并执行；CI 未包含本地工作簿时，仅排除这些数据依赖脚本。
if not HSCREDIT_DEMO_XLSX.exists():
    collect_ignore.extend(
        [
            "test_binning/test_binning_review.py",
            "test_binning/test_binning_fixes.py",
            "test_binning/test_binning_detailed.py",
            "test_binning/test_monotonic_binning.py",
            "test_utils/test_default_behavior.py",
            "test_utils/test_final_verification.py",
            "test_utils/test_feature_type_edge_cases.py",
        ]
    )


@pytest.fixture(scope="session")
def pypmml_model():
    """提供可用的 PyPMML 模型类，并在 Py4J 缺 launcher JAR 时回退 JPype。"""
    pypmml = pytest.importorskip("pypmml")
    from pypmml import PMMLContext
    import py4j
    import py4j.java_gateway as java_gateway

    py4j_jar = java_gateway.find_jar_path()
    if not py4j_jar:
        user_jar = (
            Path(site.USER_BASE)
            / "share"
            / "py4j"
            / f"py4j{py4j.__version__}.jar"
        )
        if user_jar.exists():
            py4j_jar = str(user_jar)
            java_gateway.find_jar_path = lambda: py4j_jar

    if py4j_jar:
        PMMLContext.getOrCreate()
    else:
        PMMLContext.getOrCreate(gateway="jpype")
    return pypmml.Model


def _skip_reason(report):
    """从 pytest skip report 中提取规范化原因。"""
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) >= 3 else longrepr
    reason = str(reason).strip()
    prefix = "Skipped: "
    return reason[len(prefix) :] if reason.startswith(prefix) else reason


def _is_allowed_ci_skip(report):
    """仅允许缺少演示工作簿或未配置真实数据库服务导致的跳过。"""
    if "hscredit_yyp.xlsx" in str(report.longrepr):
        return True
    return _skip_reason(report) in _ALLOWED_DATABASE_SKIP_REASONS


def pytest_sessionfinish(session, exitstatus):
    """CI 拒绝白名单外的测试跳过。"""
    if os.environ.get("HSCREDIT_STRICT_CI_SKIPS") != "1":
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = terminal.stats.get("skipped", []) if terminal is not None else []
    unexpected = [report for report in skipped if not _is_allowed_ci_skip(report)]
    if not unexpected:
        return

    if terminal is not None:
        terminal.write_sep("=", "CI 检测到非工作簿原因的测试跳过")
        for report in unexpected:
            terminal.write_line(f"{report.nodeid}: {report.longrepr}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
