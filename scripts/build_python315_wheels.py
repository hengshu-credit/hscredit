"""从校验过的官方源码构建 Python 3.15 的兼容依赖 wheel。"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from urllib.request import urlopen


def windows_toolchain(project, environment):
    """识别已安装的 MSVC，并让上游构建支持 Build Tools 和新版 VS 目录。"""
    locator = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = environment.get("VSINSTALLDIR")
    if not installation:
        installation = subprocess.check_output(
            [str(locator), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath", "-utf8"],
            text=True,
            encoding="utf-8",
        ).strip()
    if not installation:
        raise RuntimeError("请先安装 Visual Studio C++ 构建工具和 Windows SDK")
    visual_studio = Path(installation)
    llvm = Path(environment.get("LLVM_ROOT") or visual_studio / "VC/Tools/Llvm/x64").resolve()
    clang = llvm / "bin/clang-cl.exe"
    if not clang.is_file():
        raise RuntimeError("请安装 C++ Clang tools for Windows，或通过 LLVM_ROOT 指定 LLVM 目录")
    toolset = (visual_studio / "VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt").read_text().strip()
    environment.update(VSINSTALLDIR=str(visual_studio) + os.sep, VCToolsVersion=toolset, LLVM_ROOT=str(llvm))
    native = project / "catboost_all_src/build/build_native.py"
    content = native.read_text(encoding="utf-8")
    replacements = {
        "def get_msvs_dir(msvs_installation_path: Optional[str], msvs_version: str) -> str:\n": "def get_msvs_dir(msvs_installation_path: Optional[str], msvs_version: str) -> str:\n" "    if os.environ.get('VSINSTALLDIR'):\n" "        return os.environ['VSINSTALLDIR']\n",
        "def get_msvc_toolset(msvs_version: str, msvc_toolset: Optional[str]) -> str:\n": "def get_msvc_toolset(msvs_version: str, msvc_toolset: Optional[str]) -> str:\n" "    msvc_toolset = msvc_toolset or os.environ.get('VCToolsVersion')\n",
        "            msvs_dir = get_msvs_dir(opts.msvs_installation_path, opts.msvs_version)\n": "            msvs_dir = get_msvs_dir(opts.msvs_installation_path, opts.msvs_version)\n" "            llvm_dir = os.environ['LLVM_ROOT']\n",
    }
    for old, new in replacements.items():
        if content.count(old) != 1:
            raise ValueError("CatBoost Windows 工具链入口与已验证的源码结构不一致")
        content = content.replace(old, new)
    old_compiler = r"{msvs_dir}\\VC\\Tools\\Llvm\\x64\\bin\\"
    if content.count(old_compiler) != 3:
        raise ValueError("CatBoost 的 LLVM 编译器入口与已验证的源码结构不一致")
    content = content.replace(old_compiler, r"{llvm_dir}\\bin\\")
    native.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=["catboost", "scikit-learn", "ngboost", "lifelines"], required=True, help="要构建的依赖")
    parser.add_argument("--wheel-dir", type=Path, required=True, help="wheel 输出目录")
    parser.add_argument("--work-dir", type=Path, help="保留源码及编译产物的独立工作目录，便于增量构建")
    parser.add_argument("--parallel", type=int, default=2, help="并发编译任务数")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 15):
        parser.error("请使用 Python 3.15 执行构建")
    if args.parallel < 1:
        parser.error("--parallel 必须为正整数")
    destination = args.wheel_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = sysconfig.get_path("scripts") + os.pathsep + environment.get("PATH", "")
    version = {"catboost": "1.2.10", "scikit-learn": "1.8.0", "ngboost": "0.6.0", "lifelines": "0.30.3"}[args.package]
    with urlopen(f"https://pypi.org/pypi/{args.package}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    # MSVC 的部分输出文件仍受路径长度限制，生成文件放到较短的独立构建目录。
    workspace = nullcontext(args.work_dir.resolve()) if args.work_dir else tempfile.TemporaryDirectory(prefix="h315-")
    with workspace as directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        archive = root / source["filename"]
        digest = hashlib.sha256()
        if archive.is_file():
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256")
        if digest.hexdigest() != source["digests"]["sha256"]:
            digest = hashlib.sha256()
            with urlopen(source["url"], timeout=60) as response, archive.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
        if digest.hexdigest() != source["digests"]["sha256"]:
            raise ValueError(f"{args.package} 官方源码包 SHA-256 校验失败")
        with tarfile.open(archive) as stream:
            stream.extractall(root, filter="data")
        project = root / source["filename"].removesuffix(".tar.gz")
        if args.package == "catboost":
            # 同步 PEP 517 和 setup_requires，避免构建时重新安装旧版 Cython。
            for filename in ("pyproject.toml", "setup.py"):
                path = project / filename
                content = path.read_text(encoding="utf-8")
                old = "cython ~= 3.0.10"
                if content.count(old) != 1:
                    raise ValueError(f"{filename} 的 Cython 依赖与已验证的源码结构不一致")
                path.write_text(content.replace(old, "cython>=3.3,<3.4"), encoding="utf-8")
            if sys.platform == "win32":
                windows_toolchain(project, environment)
            command = [sys.executable, "setup.py", "bdist_wheel", "--no-widget", "--no-cuda", f"--parallel={args.parallel}", f"--dist-dir={destination}"]
        elif args.package == "scikit-learn":
            command = [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                f"--config-settings=compile-args=-j{args.parallel}",
                f"--config-settings=build-dir={root / 'b'}",
                "--wheel-dir",
                str(destination),
                str(project),
            ]
        elif args.package == "ngboost":
            path = project / "pyproject.toml"
            content = path.read_text(encoding="utf-8")
            old = 'python = ">=3.10, <3.15"'
            if content.count(old) != 1:
                raise ValueError("NGBoost 的 Python 版本声明与已验证的源码结构不一致")
            path.write_text(content.replace(old, 'python = ">=3.10, <3.16"'), encoding="utf-8")
            command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(destination), str(project)]
        else:
            path = project / "reqs/base-requirements.txt"
            content = path.read_text(encoding="utf-8")
            old = "pandas>=2.1,<3.0"
            if content.count(old) != 1:
                raise ValueError("lifelines 的 pandas 依赖与已验证的源码结构不一致")
            new = "pandas>=2.1,<3.0; python_version < '3.15'\npandas>=3.0.6,<4.0; python_version >= '3.15'"
            path.write_text(content.replace(old, new), encoding="utf-8")
            command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(destination), str(project)]
        subprocess.run(
            command,
            cwd=project,
            env=environment,
            check=True,
        )
    print(f"{args.package} Python 3.15 wheel 已输出至 {destination}")


if __name__ == "__main__":
    main()
