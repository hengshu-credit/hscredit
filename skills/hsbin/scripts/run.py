"""独立安装的 hscredit Skill 标准库引导器。"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description="运行一个 hscredit Agent Skill 请求")
    parser.add_argument("request", help="UTF-8 JSON 请求文件路径")
    parser.add_argument("--debug", action="store_true", help="返回脱敏后的调试信息")
    return parser


def _error(code, message, *, field=None):
    error = {"code": code, "message": message}
    if field is not None:
        error["field"] = field
    print(json.dumps({"status": "error", "error": error}, ensure_ascii=False))


def _contains_object_ref(value):
    if isinstance(value, dict):
        if value.get("kind") == "object_ref":
            return True
        return any(_contains_object_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_object_ref(item) for item in value)
    return False


def _extras(request):
    extras = {"skills"}

    def inspect(value):
        if isinstance(value, dict):
            path = value.get("path")
            if value.get("kind") == "file" and isinstance(path, str) and path.lower().endswith(".parquet"):
                extras.add("parquet")
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(request.get("inputs", {}))
    return tuple(sorted(extras))


def _cache_root():
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "hscredit" / "skills" / "envs").resolve()


def _venv_python(directory):
    if sys.platform == "win32":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _repo_root(script_path):
    for parent in script_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "hscredit").is_dir():
            return parent
    return None


def _requirement(config, extras, repo_root):
    suffix = f"[{','.join(extras)}]"
    if repo_root is not None:
        return f"{repo_root}{suffix}", True
    package = config["hscredit"]["package"]
    repository = config["hscredit"]["repository"]
    ref = config["hscredit"]["ref"]
    return f"{package}{suffix} @ git+{repository}@{ref}", False


def _run_install(python, requirement, editable):
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if editable:
        command.append("-e")
    command.append(requirement)
    result = subprocess.run(command, text=True, capture_output=True, check=False, shell=False)
    if result.returncode != 0:
        detail = "\n".join((result.stdout.splitlines()[-10:] + result.stderr.splitlines()[-20:]))
        _error(
            "DEPENDENCY_INSTALL_FAILED",
            f"自动安装 hscredit 依赖失败，退出码 {result.returncode}：{detail}",
        )
    return result.returncode


def _ensure_venv(config, request, repo_root):
    extras = _extras(request)
    requirement, editable = _requirement(config, extras, repo_root)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    key_payload = json.dumps(
        {"requirement": requirement, "extras": extras, "python": version},
        sort_keys=True,
    ).encode("utf-8")
    key = hashlib.sha256(key_payload).hexdigest()[:16]
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    reuse = request.get("environment", {}).get("reuse", True)
    if reuse:
        directory = root / key
    else:
        directory = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=root))
    python = _venv_python(directory)
    marker = directory / ".hscredit-runtime.json"
    expected = {"requirement": requirement, "extras": list(extras), "python": version}
    if reuse and python.is_file() and marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected:
                return python
        except (OSError, ValueError):
            pass
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, clear=False).create(directory)
    python = _venv_python(directory)
    if _run_install(python, requirement, editable) != 0:
        return None
    marker.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return python


def _ensure_current(config, request, repo_root):
    probe = subprocess.run(
        [sys.executable, "-c", "import hscredit.skills_runtime"],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if probe.returncode == 0:
        return Path(sys.executable).resolve()
    environment = request.get("environment", {})
    if environment.get("install_missing") is not True:
        _error(
            "DEPENDENCY_MISSING",
            "当前环境缺少 hscredit Skills 运行时；如需自动安装请显式设置 install_missing=true",
            field="environment.install_missing",
        )
        return None
    requirement, editable = _requirement(config, _extras(request), repo_root)
    if _run_install(Path(sys.executable), requirement, editable) != 0:
        return None
    return Path(sys.executable).resolve()


def main(argv=None):
    args = _parser().parse_args(argv)
    script_path = Path(__file__).resolve()
    config_path = script_path.parent.parent / "runtime.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        request_path = Path(args.request).expanduser().resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _error("SCHEMA_INVALID", f"无法读取 Skill 配置或请求：{exc}")
        return 1

    environment = request.get("environment", {})
    mode = environment.get("mode", "isolated")
    repo_root = _repo_root(script_path)
    if mode == "isolated":
        if _contains_object_ref(request.get("inputs", {})):
            _error(
                "OBJECT_REF_REQUIRES_CURRENT_ENV",
                "object_ref 不能传入隔离解释器；请使用 current 环境或先保存为可信制品",
                field="environment.mode",
            )
            return 1
        python = _ensure_venv(config, request, repo_root)
    elif mode == "current":
        python = _ensure_current(config, request, repo_root)
    else:
        _error("SCHEMA_INVALID", f"不支持的环境模式“{mode}”", field="environment.mode")
        return 1
    if python is None:
        return 1

    command = [
        str(python),
        "-m",
        "hscredit.skills_runtime",
        "--skill",
        config["skill"],
        "--request",
        str(request_path),
    ]
    if args.debug:
        command.append("--debug")
    child_environment = dict(os.environ)
    child_environment.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        command,
        check=False,
        shell=False,
        env=child_environment,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
