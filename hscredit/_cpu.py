"""无需外部命令的 CPU 拓扑探测及动态并发上限。"""

from functools import lru_cache
from numbers import Integral
from pathlib import Path
import os
import struct
import sys
from typing import Callable, Optional


def _positive_count(value) -> Optional[int]:
    """仅接受正整数核数，未知值不伪装成物理核数。"""
    if isinstance(value, Integral) and not isinstance(value, bool):
        count = int(value)
        return count if count > 0 else None
    return None


def _windows_core_records(data: bytes) -> Optional[int]:
    """按 Windows 变长结构的 Size 前进，一个 ProcessorCore 记录对应一个物理核。"""
    count, offset = 0, 0
    while offset < len(data):
        if len(data) - offset < 8:
            return None
        relationship, size = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            return None
        if relationship == 0:  # RelationProcessorCore
            count += 1
        offset += size
    return _positive_count(count)


def _windows_physical_cores() -> Optional[int]:
    """通过 Win32 API 查询，兼容变长记录和多个处理器组，不依赖 WMIC/PowerShell。"""
    import ctypes

    # WinDLL 只在 Windows 的 ctypes 中提供；其他系统上的模拟测试也使用同一入口。
    query = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
    query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    query.restype = ctypes.c_int
    size = ctypes.c_uint32()
    query(0, None, ctypes.byref(size))
    if not 8 <= size.value <= 16 * 1024 * 1024:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if not query(0, buffer, ctypes.byref(size)) or size.value > len(buffer):
        return None
    return _windows_core_records(buffer.raw[: size.value])


def _linux_physical_cores(root: Path = Path("/sys/devices/system/cpu")) -> Optional[int]:
    """读取 sysfs 的插槽/裸片/簇/核心组合，避免不同硬件层级核心编号重复。"""
    cores = set()
    for cpu in root.glob("cpu[0-9]*"):
        if not cpu.name[3:].isdigit():
            continue
        online = cpu / "online"
        if online.exists() and online.read_text(encoding="ascii").strip() == "0":
            continue
        package = int((cpu / "topology/physical_package_id").read_text(encoding="ascii"))
        core = int((cpu / "topology/core_id").read_text(encoding="ascii"))
        if package < 0 or core < 0:
            return None  # 内核默认 -1 表示拓扑未知。
        hierarchy = []
        for name in ("die_id", "cluster_id"):
            path = cpu / "topology" / name
            hierarchy.append(int(path.read_text(encoding="ascii")) if path.exists() else -1)
        cores.add((package, *hierarchy, core))
    return _positive_count(len(cores))


def _darwin_physical_cores() -> Optional[int]:
    """通过 macOS 的 sysctlbyname 查询物理核，不依赖 PATH 中的 sysctl 命令。"""
    import ctypes

    query = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).sysctlbyname
    query.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    count = ctypes.c_uint32()
    size = ctypes.c_size_t(ctypes.sizeof(count))
    if query(b"hw.physicalcpu", ctypes.byref(count), ctypes.byref(size), None, 0) != 0:
        return None
    if size.value != ctypes.sizeof(count):
        return None
    return _positive_count(count.value)


def _psutil_physical_cores() -> Optional[int]:
    """可选 psutil 2.0+ 的稳定公开接口；不强制安装或升级共享环境。"""
    from packaging.version import Version
    from ._compat import installed_version

    try:
        version = installed_version("psutil")
        if version is None or version < Version("2.0"):
            return None
        import psutil

        return _positive_count(psutil.cpu_count(logical=False))
    except (ImportError, OSError, AttributeError, ValueError, TypeError, RuntimeError):
        return None


@lru_cache(maxsize=8)
def _hardware_cpu_count(platform_name: str, logical_count: Optional[int]) -> Optional[int]:
    """只缓存硬件探测；平台或系统逻辑核数变化时重新查询，不缓存进程配额。"""
    probe = {"win32": _windows_physical_cores, "linux": _linux_physical_cores, "darwin": _darwin_physical_cores}.get(
        platform_name
    )
    if probe is not None:
        try:
            count = _positive_count(probe())
        except (OSError, AttributeError, ValueError, TypeError, RuntimeError):
            count = None
        if count is not None:
            return count
    return _psutil_physical_cores()


def get_hardware_cpu_count() -> Optional[int]:
    """返回硬件物理核数；没有可靠探测结果时返回 None。"""
    try:
        logical = os.cpu_count()
    except (OSError, NotImplementedError):
        logical = None
    return _hardware_cpu_count(sys.platform, logical)


def usable_cpu_count(counter: Callable[[], int]) -> int:
    """用 joblib 的无参接口保留配额/亲和性约束，失败时保守使用单 worker。"""
    configured = os.environ.get("LOKY_MAX_CPU_COUNT")
    try:
        limit = max(1, int(configured)) if configured is not None else None
    except ValueError as exc:
        raise ValueError("LOKY_MAX_CPU_COUNT 必须是整数") from exc
    try:
        available = _positive_count(counter()) or 1
    except (OSError, AttributeError, ValueError, TypeError, RuntimeError):
        available = 1
    return min(available, limit) if limit is not None else available
