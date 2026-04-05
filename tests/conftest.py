"""共享测试配置与演示数据路径."""

from pathlib import Path

# tests/conftest.py -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HSCREDIT_DEMO_XLSX = PROJECT_ROOT / "examples" / "hscredit.xlsx"

# import 时会执行脚本式逻辑或依赖本地 xlsx，不宜随 pytest 收集；可手动 python 路径运行
collect_ignore = [
    "test_binning/test_binning_review.py",
    "test_binning/test_binning_fixes.py",
    "test_binning/test_binning_detailed.py",
    "test_binning/test_monotonic_binning.py",
    "test_utils/test_default_behavior.py",
    "test_utils/test_final_verification.py",
    "test_utils/test_feature_type_edge_cases.py",
    "test_utils/test_notebook_fix.py",
]
