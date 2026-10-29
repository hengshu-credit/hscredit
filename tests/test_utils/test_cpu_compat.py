"""CPU 探测、版本适配、动态资源上限及无外部命令的回归。"""

import ctypes
import os
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from hscredit import _cpu, _compat
from hscredit.utils import parallel


@pytest.fixture(autouse=True)
def isolate_cpu_cache(monkeypatch):
    cache = _cpu._hardware_cpu_count
    cache.cache_clear()
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
    yield
    cache.cache_clear()


@pytest.mark.parametrize(
    "version, expected",
    [
        (None, False),
        ("0.17.0", False),
        ("1.0.0", True),
        ("1.4.2", True),
        ("1.6.0", True),
        ("1.7.0.dev0", False),
        ("1.7.0rc1", False),
        ("1.7.0", False),
    ],
)
def test_joblib_version_boundaries(version, expected):
    assert _compat.needs_joblib_cpu_compat(Version(version) if version else None) is expected


def test_adapter_preserves_resource_limits_cache_and_unknown_protocol(monkeypatch):
    def legacy_query():
        raise AssertionError("不能调用旧命令")

    count = Mock(return_value=24)
    monkeypatch.setattr(_cpu, "get_hardware_cpu_count", count)
    context = SimpleNamespace(
        _count_physical_cores=legacy_query,
        cpu_count=lambda: 32,
        physical_cores_cache="not found",
    )
    _compat.install_joblib_cpu_compat(context, Version("1.4.2"))
    installed = context._count_physical_cores
    _compat.install_joblib_cpu_compat(context, Version("1.4.2"))
    assert context._count_physical_cores is installed
    assert context._count_physical_cores() == (24, None)
    context.cpu_count = lambda: 3
    assert context._count_physical_cores() == (3, None)
    count.return_value = None
    assert context._count_physical_cores() == ("not found", None)
    assert context.physical_cores_cache == "not found"


def test_adapter_leaves_unverified_versions_untouched():
    context = SimpleNamespace()
    _compat.install_joblib_cpu_compat(context, Version("1.7.0"))
    assert vars(context) == {}


def test_prepare_dependency_uses_explicit_joblib_version(monkeypatch):
    calls = []
    monkeypatch.setattr(_compat, "installed_version", lambda name: Version("1.4.2"))
    monkeypatch.setattr(_compat, "install_joblib_cpu_compat", lambda context, version: calls.append(version))
    _compat.prepare_dependency("joblib")
    assert calls == [Version("1.4.2")]


def test_windows_variable_size_records_and_processor_groups(monkeypatch):
    records = b"".join(struct.pack("<II", 0, size) + bytes(size - 8) for size in [48, 64, 80])
    calls = []

    class Query:
        def __call__(self, relationship, buffer, size):
            assert relationship == 0
            calls.append(buffer is None)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32)).contents.value = len(records)
            if buffer is None:
                return 0
            ctypes.memmove(buffer, records, len(records))
            return 1

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(GetLogicalProcessorInformationEx=Query()),
        raising=False,
    )
    assert _cpu._windows_physical_cores() == 3
    assert calls == [True, False]


@pytest.mark.parametrize("data", [b"", b"abc", struct.pack("<II", 0, 0), struct.pack("<II", 0, 48) + b"short"])
def test_windows_malformed_records_cannot_create_fake_counts(data):
    assert _cpu._windows_core_records(data) is None


def test_windows_api_denial_returns_unknown(monkeypatch):
    class Query:
        def __call__(self, relationship, buffer, size):
            return 0

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(GetLogicalProcessorInformationEx=Query()),
        raising=False,
    )
    assert _cpu._windows_physical_cores() is None


def test_linux_counts_package_core_pairs_and_ignores_offline_cpu(tmp_path):
    for index, (package, core) in enumerate([(0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0)]):
        topology = tmp_path / f"cpu{index}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(str(package), encoding="ascii")
        (topology / "core_id").write_text(str(core), encoding="ascii")
    (tmp_path / "cpu99").mkdir()
    (tmp_path / "cpu99/online").write_text("0", encoding="ascii")
    assert _cpu._linux_physical_cores(tmp_path) == 3
    (tmp_path / "cpu0/topology/physical_package_id").write_text("-1", encoding="ascii")
    assert _cpu._linux_physical_cores(tmp_path) is None


def test_linux_distinguishes_repeated_core_ids_on_dies_and_clusters(tmp_path):
    for index, (die, cluster) in enumerate([(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1)]):
        topology = tmp_path / f"cpu{index}" / "topology"
        topology.mkdir(parents=True)
        for name, value in {"physical_package_id": 0, "die_id": die, "cluster_id": cluster, "core_id": 0}.items():
            (topology / name).write_text(str(value), encoding="ascii")
    assert _cpu._linux_physical_cores(tmp_path) == 3


def test_macos_uses_read_only_sysctl_api(monkeypatch):
    class Query:
        def __call__(self, name, result, size, new_value, new_size):
            assert name == b"hw.physicalcpu" and new_value is None and new_size == 0
            ctypes.cast(result, ctypes.POINTER(ctypes.c_uint32)).contents.value = 8
            ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t)).contents.value = 4
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(sysctlbyname=Query()))
    assert _cpu._darwin_physical_cores() == 8


@pytest.mark.parametrize(
    "platform, probe",
    [
        ("win32", "_windows_physical_cores"),
        ("linux", "_linux_physical_cores"),
        ("darwin", "_darwin_physical_cores"),
    ],
)
@pytest.mark.parametrize("error", [PermissionError, OSError, NotImplementedError, ValueError])
def test_platform_probe_failure_uses_optional_provider(monkeypatch, platform, probe, error):
    monkeypatch.setattr(_cpu, probe, Mock(side_effect=error("探测不可用")))
    monkeypatch.setattr(_cpu, "_psutil_physical_cores", lambda: 6)
    assert _cpu._hardware_cpu_count(platform, 12) == 6


@pytest.mark.parametrize(
    "version, value, expected",
    [(None, 4, None), ("1.2.0", 4, None), ("2.0.0", 4, 4), ("5.9.0", None, None), ("7.2.0", 0, None)],
)
def test_optional_psutil_uses_version_and_validates_result(monkeypatch, version, value, expected):
    monkeypatch.setattr(_compat, "installed_version", lambda name: Version(version) if version else None)
    probe = Mock(return_value=value)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(cpu_count=probe))
    assert _cpu._psutil_physical_cores() == expected
    if version is None or Version(version) < Version("2.0"):
        probe.assert_not_called()
    else:
        probe.assert_called_once_with(logical=False)


def test_unknown_platform_and_missing_provider_return_none(monkeypatch):
    monkeypatch.setattr(_cpu, "_psutil_physical_cores", lambda: None)
    assert _cpu._hardware_cpu_count("unknown", None) is None


def test_broken_optional_psutil_is_not_required(monkeypatch):
    monkeypatch.setattr(_compat, "installed_version", lambda name: Version("5.9.0"))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(cpu_count=Mock(side_effect=OSError("接口不可用"))))
    assert _cpu._psutil_physical_cores() is None


def test_real_joblib_resource_limit_is_preserved(monkeypatch):
    import joblib
    from joblib.externals.loky.backend import context

    monkeypatch.setattr(context.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(context, "_cpu_count_user", lambda count: 2)
    monkeypatch.setattr(_cpu, "_hardware_cpu_count", lambda *args: 24)
    assert joblib.cpu_count(only_physical_cores=True) == 2
    assert parallel.get_physical_cpu_count() == 2


def test_hardware_cache_does_not_cache_cpu_limits(monkeypatch):
    hardware = Mock(return_value=24)
    monkeypatch.setattr(_cpu, "_windows_physical_cores", hardware)
    monkeypatch.setattr(parallel, "get_hardware_cpu_count", lambda: _cpu._hardware_cpu_count("win32", 32))
    logical = Mock(return_value=32)
    monkeypatch.setattr(parallel, "joblib_cpu_count", logical)
    assert parallel.get_physical_cpu_count() == 24
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    assert parallel.get_physical_cpu_count() == 2
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT")
    logical.return_value = 4  # 进程亲和性或容器配额变小。
    assert parallel.get_physical_cpu_count() == 4
    hardware.assert_called_once_with()
    logical.assert_called_with()


def test_hardware_count_is_refreshed_when_logical_topology_changes(monkeypatch):
    probe = Mock(side_effect=[4, 8])
    monkeypatch.setattr(_cpu, "_windows_physical_cores", probe)
    assert _cpu._hardware_cpu_count("win32", 8) == 4
    assert _cpu._hardware_cpu_count("win32", 16) == 8


@pytest.mark.parametrize("value", [None, 0, -1, True, "invalid"])
def test_unreliable_resource_counter_uses_one_worker(monkeypatch, value):
    monkeypatch.setattr(parallel, "joblib_cpu_count", lambda: value)
    assert parallel.get_physical_cpu_count() == 1


def test_resource_denial_is_conservative_but_bad_explicit_limit_is_not_hidden(monkeypatch):
    monkeypatch.setattr(parallel, "joblib_cpu_count", Mock(side_effect=PermissionError("无法读取配额")))
    monkeypatch.setattr(parallel, "get_hardware_cpu_count", lambda: 64)
    assert parallel.get_physical_cpu_count() == 1
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "不是数字")
    with pytest.raises(ValueError, match="LOKY_MAX_CPU_COUNT 必须是整数"):
        parallel.get_physical_cpu_count()


def test_real_process_has_no_cpu_subprocess_warning_or_global_side_effects():
    code = r"""
import os, subprocess, warnings
import hscredit
import joblib
from hscredit.utils import get_physical_cpu_count
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from joblib.externals.loky.backend import context
from hscredit._cpu import get_hardware_cpu_count

def forbidden(*args, **kwargs):
    raise AssertionError("CPU query must not start subprocesses")

subprocess.Popen = forbidden
environment, filters = dict(os.environ), list(warnings.filters)
old_cache = context.physical_cores_cache
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    available = joblib.cpu_count()
    for value in (get_physical_cpu_count(), joblib.cpu_count(only_physical_cores=True), _openmp_effective_n_threads()):
        assert 1 <= value <= available
    assert not captured, [str(item.message) for item in captured]
assert dict(os.environ) == environment and warnings.filters == filters
assert context.physical_cores_cache == old_cache
print("hardware=", get_hardware_cpu_count(), "available=", available)
"""
    environment = dict(os.environ)
    environment.pop("OMP_NUM_THREADS", None)  # 验证自动探测，而非用户显式 OpenMP 覆盖值。
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, timeout=90, env=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Could not find the number of physical cores" not in result.stderr
    assert "Traceback" not in result.stderr
