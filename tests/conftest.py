"""共享测试配置与演示数据路径."""

from pathlib import Path

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
