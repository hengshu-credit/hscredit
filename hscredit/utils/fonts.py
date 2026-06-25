"""系统字体安装工具.

将 hscredit 包内置字体安装到当前用户的系统字体目录，避免要求管理员权限。
"""

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


FONT_NAME = "阿里妈妈方圆体 VF Medium"
FONT_FILENAME = "hscredit-font.ttf"


def get_bundled_font_path() -> Path:
    """返回 hscredit 包内置字体文件路径."""
    return Path(__file__).resolve().parent.parent / "resources" / "fonts" / "font.ttf"


def _same_file_content(source: Path, destination: Path) -> bool:
    """判断两个字体文件内容是否一致."""
    if not destination.is_file() or source.stat().st_size != destination.stat().st_size:
        return False

    with source.open("rb") as source_file, destination.open("rb") as destination_file:
        while True:
            source_chunk = source_file.read(1024 * 1024)
            destination_chunk = destination_file.read(1024 * 1024)
            if source_chunk != destination_chunk:
                return False
            if not source_chunk:
                return True


def _copy_font(source: Path, destination: Path, force: bool) -> bool:
    """复制字体文件，返回本次是否实际写入."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and _same_file_content(source, destination):
        return False
    shutil.copy2(source, destination)
    return True


def _install_windows_font(source: Path, force: bool) -> Tuple[Path, bool]:
    """将字体安装到 Windows 当前用户字体库."""
    import winreg

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("无法获取 LOCALAPPDATA，不能定位 Windows 用户字体目录。")

    destination = Path(local_app_data) / "Microsoft" / "Windows" / "Fonts" / FONT_FILENAME
    changed = _copy_font(source, destination, force)

    registry_path = r"Software\Microsoft\Windows NT\CurrentVersion\Fonts"
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, registry_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, f"{FONT_NAME} (TrueType)", 0, winreg.REG_SZ, str(destination))

    # 将字体加载到当前 Windows 会话，并通知已打开的应用刷新字体列表。
    # 返回 0 也可能表示字体已在当前会话中加载，因此不将其视为安装失败。
    ctypes.windll.gdi32.AddFontResourceW(str(destination))

    hwnd_broadcast = 0xFFFF
    wm_fontchange = 0x001D
    smto_abortifhung = 0x0002
    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(
        hwnd_broadcast,
        wm_fontchange,
        0,
        0,
        smto_abortifhung,
        1000,
        ctypes.byref(result),
    )
    return destination, changed


def _install_macos_font(source: Path, force: bool) -> Tuple[Path, bool]:
    """将字体安装到 macOS 当前用户字体库."""
    destination = Path.home() / "Library" / "Fonts" / FONT_FILENAME
    return destination, _copy_font(source, destination, force)


def _install_linux_font(source: Path, force: bool) -> Tuple[Path, bool]:
    """将字体安装到 Linux 当前用户字体库."""
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    destination = data_home / "fonts" / FONT_FILENAME
    changed = _copy_font(source, destination, force)

    fc_cache = shutil.which("fc-cache")
    if fc_cache:
        subprocess.run(
            [fc_cache, "-f", str(destination.parent)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    return destination, changed


def install_bundled_font(force: bool = False, system: Optional[str] = None) -> Tuple[Path, bool]:
    """安装 hscredit 内置字体到当前用户字体库.

    :param force: 是否强制覆盖已有字体文件
    :param system: 操作系统名称，仅用于测试或显式覆盖自动识别结果
    :return: ``(安装路径, 是否实际更新字体文件)``
    """
    source = get_bundled_font_path()
    if not source.is_file():
        raise FileNotFoundError(f"未找到内置字体文件：{source}")

    system_name = (system or platform.system()).lower()
    if system_name == "windows":
        return _install_windows_font(source, force)
    if system_name == "darwin":
        return _install_macos_font(source, force)
    if system_name == "linux":
        return _install_linux_font(source, force)
    raise RuntimeError(f"暂不支持在 {system or platform.system()} 系统上自动安装字体。")
