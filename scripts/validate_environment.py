"""验证 hscredit 开发环境的基础可用性."""

import importlib
from importlib import metadata
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "packaging",
    "scipy",
    "sklearn",
    "statsmodels.api",
    "matplotlib",
    "openpyxl",
    "hscredit",
    "shap",
]


def main() -> int:
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")

    # 可选驱动/模型只有已安装才验证；不能把存在但二进制损坏当作可用。
    for module_name in ("catboost", "xgboost", "lightgbm", "ngboost"):
        if importlib.util.find_spec(module_name) is not None:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                failures.append(f"{module_name}: {exc}")

    from packaging.requirements import Requirement
    from packaging.markers import default_environment

    try:
        environment = {**default_environment(), "extra": ""}
        for entry in metadata.requires("hscredit") or []:
            requirement = Requirement(entry)
            if requirement.marker and not requirement.marker.evaluate(environment):
                installed_boost = (
                    requirement.name in {"ngboost", "xgboost", "catboost", "lightgbm"}
                    and importlib.util.find_spec(requirement.name) is not None
                )
                if not installed_boost or not requirement.marker.evaluate({**environment, "extra": "boost"}):
                    continue
            try:
                installed = metadata.version(requirement.name)
                if requirement.specifier and installed not in requirement.specifier:
                    failures.append(f"依赖版本不满足: {requirement}，当前 {installed}")
            except metadata.PackageNotFoundError:
                failures.append(f"未安装必需依赖: {requirement.name}")
        loaded = sys.modules.get("hscredit")
        if loaded is not None and metadata.version("hscredit") != loaded.__version__:
            failures.append("源码与安装元数据版本不一致，请在当前解释器执行 pip install -e .")
    except metadata.PackageNotFoundError:
        failures.append("当前解释器尚未安装 hscredit，请执行 pip install -e .")

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
