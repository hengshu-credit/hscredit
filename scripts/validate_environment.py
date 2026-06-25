"""验证 hscredit 开发环境的基础可用性."""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "openpyxl",
    "hscredit",
]


def main() -> int:
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")

    if failures:
        print("环境验证失败:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    import hscredit

    print(f"环境验证通过，hscredit 版本: {hscredit.__version__}")
    print(f"Python 版本: {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
