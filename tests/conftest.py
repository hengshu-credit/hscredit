"""共享测试配置、演示数据路径与 CI 跳过策略."""

import os
from pathlib import Path

import pytest

# tests/conftest.py -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HSCREDIT_DEMO_XLSX = PROJECT_ROOT / "examples" / "hscredit_yyp.xlsx"

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


def pytest_sessionfinish(session, exitstatus):
    """CI 仅允许因缺少本地演示工作簿而跳过测试。"""
    if os.environ.get("HSCREDIT_STRICT_CI_SKIPS") != "1":
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = terminal.stats.get("skipped", []) if terminal is not None else []
    unexpected = [report for report in skipped if "hscredit_yyp.xlsx" not in str(report.longrepr)]
    if not unexpected:
        return

    if terminal is not None:
        terminal.write_sep("=", "CI 检测到非工作簿原因的测试跳过")
        for report in unexpected:
            terminal.write_line(f"{report.nodeid}: {report.longrepr}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
