"""Agent Skills 隔离环境创建和受控重执行。"""

import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .dependencies import environment_key, plan_environment, resolve_cache_root
from .errors import SkillExecutionError


def environment_python(environment_dir: Path) -> Path:
    """返回 venv 对应的 Python 解释器路径。"""
    if sys.platform == "win32":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def _run_checked(command: Sequence[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [str(part) for part in command],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        stderr_lines = result.stderr.splitlines()[-20:]
        stdout_lines = result.stdout.splitlines()[-10:]
        detail = "\n".join(stdout_lines + stderr_lines)
        raise SkillExecutionError(
            code="DEPENDENCY_INSTALL_FAILED",
            message=f"隔离依赖安装失败，退出码 {result.returncode}：{detail}",
        )
    return result


def install_requirement(environment_dir: Path, requirement: str) -> Path:
    """在指定的新建或既有 venv 中安装一个受信任 requirement。"""
    environment_dir = Path(environment_dir).resolve()
    python = environment_python(environment_dir)
    if not python.is_file():
        environment_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(environment_dir)
    python = environment_python(environment_dir)
    _run_checked(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            requirement,
        ]
    )
    return python


def _requirement(source: str, extras: Sequence[str]) -> str:
    suffix = f"[{','.join(sorted(set(extras)))}]" if extras else ""
    path = Path(source)
    if path.exists():
        return f"{path.resolve()}{suffix}"
    if source.startswith("git+"):
        return f"hscredit{suffix} @ {source}"
    return f"{source}{suffix}"


def ensure_environment(
    runtime_config: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    source_override: Optional[str] = None,
    cache_root: Optional[Path] = None,
) -> Path:
    """创建或复用隔离环境，并返回其中的 Python。"""
    skill = str(runtime_config["skill"])
    operation = str(request["operation"])
    plan = plan_environment(skill, operation, request)
    if plan.mode == "current":
        return Path(sys.executable).resolve()

    hscredit_config = runtime_config.get("hscredit", {})
    if source_override is not None:
        source = str(Path(source_override).resolve())
    else:
        repository = str(hscredit_config["repository"])
        ref = str(hscredit_config["ref"])
        source = f"git+{repository}@{ref}"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    key = environment_key(source, plan.extras, version)
    root = Path(cache_root).resolve() if cache_root is not None else resolve_cache_root()
    environment_dir = root / key
    marker = environment_dir / ".hscredit-runtime.json"
    expected = {"source": source, "extras": list(plan.extras), "python": version}
    python = environment_python(environment_dir)
    if plan.reuse and python.is_file() and marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected:
                return python
        except (OSError, ValueError):
            pass

    python = install_requirement(environment_dir, _requirement(source, plan.extras))
    marker.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return python


def run_in_environment(
    python: Path,
    skill: str,
    request_path: Path,
    *,
    debug: bool = False,
) -> int:
    """使用目标解释器执行 Skills JSON CLI。"""
    command = [
        str(Path(python).resolve()),
        "-m",
        "hscredit.skills_runtime",
        "--skill",
        skill,
        "--request",
        str(Path(request_path).resolve()),
    ]
    if debug:
        command.append("--debug")
    return subprocess.run(command, check=False, shell=False).returncode
