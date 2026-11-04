"""依赖版本快照和环境自检，不为检查而加载可选模型框架。"""

import sys
from importlib import metadata, util
from packaging.requirements import Requirement

_IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "rtdl-revisiting-models": "rtdl_revisiting_models",
    "rtdl-num-embeddings": "rtdl_num_embeddings",
}


def runtime_version(package):
    """已加载模块优先采用实际版本，未加载模块使用发行元数据。"""
    module = sys.modules.get(_IMPORT_NAMES.get(package, package.replace("-", "_")))
    actual = getattr(module, "__version__", None)
    if actual is not None:
        return str(actual)
    return metadata.version(package)


def check_environment(packages=None, *, strict=False):
    """检查声明依赖和实际导入版本；返回诊断字典，strict=True 在异常时抛错。"""
    if packages is None:
        packages = []
        for value in metadata.requires("hscredit") or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                packages.append(requirement)
    else:
        packages = [Requirement(value) for value in packages]
    rows = []
    for requirement in packages:
        name = requirement.name
        module_name = _IMPORT_NAMES.get(name, name.replace("-", "_"))
        try:
            declared = metadata.version(name)
        except metadata.PackageNotFoundError:
            declared = None
        actual = getattr(sys.modules.get(module_name), "__version__", None)
        actual = str(actual) if actual is not None else None
        state = "缺失" if declared is None else "正常"
        if actual is not None and declared != actual:
            state = "导入版本与安装元数据不一致"
        elif declared is not None and requirement.specifier and not requirement.specifier.contains(actual or declared, prereleases=True):
            state = "版本不满足声明"
        rows.append({"依赖": name, "要求": str(requirement.specifier), "安装版本": declared, "导入版本": actual, "状态": state})
    result = {"Python": sys.version.split()[0], "正常": all(row["状态"] == "正常" for row in rows), "依赖": rows}
    if strict and not result["正常"]:
        from ..exceptions import DependencyError

        raise DependencyError("运行环境不满足要求: " + "；".join(f"{row['依赖']}: {row['状态']}" for row in rows if row["状态"] != "正常"))
    return result
