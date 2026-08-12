"""系统字体安装、运行时默认字体与命令行测试."""

import copy
import importlib
import sys
from types import SimpleNamespace

import pytest
from matplotlib import pyplot as plt

from hscredit import cli
from hscredit.utils import fonts

init_module = importlib.import_module("hscredit.utils.init")


@pytest.fixture(autouse=True)
def restore_matplotlib_font_settings():
    """隔离 init_setting 测试修改的 matplotlib 全局字体配置."""
    keys = ("font.family", "font.weight", "axes.titleweight", "axes.labelweight")
    original = {key: copy.deepcopy(plt.rcParams[key]) for key in keys}
    yield
    plt.rcParams.update(original)


def test_initialize_bundled_font_selects_brand_after_install(monkeypatch):
    """安装成功后应选择 TTF 的真实字体家族名作为运行时默认字体."""
    monkeypatch.setattr(fonts, "install_bundled_font", lambda: (fonts.get_bundled_font_path(), True))

    assert fonts.initialize_bundled_font() == "Alimama FangYuanTi VF"
    assert fonts.get_default_font_name() == "Alimama FangYuanTi VF"


def test_initialize_bundled_font_falls_back_when_install_is_denied(monkeypatch):
    """权限不足且系统不存在品牌字体时应静默回退到楷体."""
    monkeypatch.setattr(
        fonts,
        "install_bundled_font",
        lambda: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(fonts, "is_font_available", lambda name: False)

    assert fonts.initialize_bundled_font() == "楷体"
    assert fonts.get_default_font_name() == "楷体"


def test_initialize_bundled_font_uses_existing_brand_after_install_failure(monkeypatch):
    """安装失败但系统已有品牌字体时不应错误回退."""
    monkeypatch.setattr(
        fonts,
        "install_bundled_font",
        lambda: (_ for _ in ()).throw(OSError("readonly")),
    )
    monkeypatch.setattr(fonts, "is_font_available", lambda name: True)

    assert fonts.initialize_bundled_font() == "Alimama FangYuanTi VF"
    assert fonts.get_default_font_name() == "Alimama FangYuanTi VF"


def test_initialize_bundled_font_falls_back_when_availability_check_fails(monkeypatch):
    """安装与系统字体探测同时异常时仍不得阻断初始化."""
    monkeypatch.setattr(
        fonts,
        "install_bundled_font",
        lambda: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(
        fonts,
        "is_font_available",
        lambda name: (_ for _ in ()).throw(RuntimeError("broken cache")),
    )

    assert fonts.initialize_bundled_font() == "楷体"
    assert fonts.get_default_font_name() == "楷体"


def test_install_windows_font_uses_current_user_font_directory(monkeypatch, tmp_path):
    """Windows 安装必须仅写入当前用户字体目录并注册真实字体家族名."""
    source = tmp_path / "source.ttf"
    source.write_bytes(b"font-content")
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    registry_calls = []

    class RegistryKey:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_winreg = SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=1,
        REG_SZ=1,
        CreateKeyEx=lambda *args: RegistryKey(),
        SetValueEx=lambda key, name, reserved, value_type, value: registry_calls.append((name, value)),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setattr(
        fonts.ctypes,
        "windll",
        SimpleNamespace(
            gdi32=SimpleNamespace(AddFontResourceW=lambda path: 1),
            user32=SimpleNamespace(SendMessageTimeoutW=lambda *args: 1),
        ),
        raising=False,
    )

    destination, changed = fonts._install_windows_font(source, force=False)

    expected = local_app_data / "Microsoft" / "Windows" / "Fonts" / "hscredit-font.ttf"
    assert (destination, changed) == (expected, True)
    assert destination.read_bytes() == b"font-content"
    assert registry_calls == [("Alimama FangYuanTi VF (TrueType)", str(expected))]


def test_install_windows_font_rejects_failed_session_load(monkeypatch, tmp_path):
    """Windows 未将字体加载进当前会话时不得报告安装成功."""
    source = tmp_path / "source.ttf"
    source.write_bytes(b"font-content")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    class RegistryKey:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "winreg",
        SimpleNamespace(
            HKEY_CURRENT_USER=object(),
            KEY_SET_VALUE=1,
            REG_SZ=1,
            CreateKeyEx=lambda *args: RegistryKey(),
            SetValueEx=lambda *args: None,
        ),
    )
    monkeypatch.setattr(
        fonts.ctypes,
        "windll",
        SimpleNamespace(
            gdi32=SimpleNamespace(AddFontResourceW=lambda path: 0),
            user32=SimpleNamespace(SendMessageTimeoutW=lambda *args: 1),
        ),
        raising=False,
    )

    with pytest.raises(OSError, match="当前 Windows 会话"):
        fonts._install_windows_font(source, force=False)


def test_install_macos_font_uses_current_user_font_directory(monkeypatch, tmp_path):
    """macOS 安装必须写入当前用户 Library/Fonts."""
    source = tmp_path / "source.ttf"
    source.write_bytes(b"font-content")
    monkeypatch.setattr(fonts.Path, "home", classmethod(lambda cls: tmp_path))

    destination, changed = fonts._install_macos_font(source, force=False)

    assert (destination, changed) == (tmp_path / "Library" / "Fonts" / "hscredit-font.ttf", True)
    assert destination.read_bytes() == b"font-content"


def test_install_linux_font_uses_xdg_directory_and_refreshes_cache(monkeypatch, tmp_path):
    """Linux 安装必须使用用户数据目录并刷新字体缓存."""
    source = tmp_path / "source.ttf"
    source.write_bytes(b"font-content")
    data_home = tmp_path / "xdg-data"
    cache_calls = []
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setattr(fonts.shutil, "which", lambda command: "/usr/bin/fc-cache")
    monkeypatch.setattr(fonts.subprocess, "run", lambda args, **kwargs: cache_calls.append((args, kwargs)))

    destination, changed = fonts._install_linux_font(source, force=False)

    expected = data_home / "fonts" / "hscredit-font.ttf"
    assert (destination, changed) == (expected, True)
    assert destination.read_bytes() == b"font-content"
    assert cache_calls[0][0] == ["/usr/bin/fc-cache", "-f", str(expected.parent)]
    assert cache_calls[0][1]["check"] is True


def test_install_linux_font_treats_empty_xdg_data_home_as_unset(monkeypatch, tmp_path):
    """空 XDG_DATA_HOME 必须回退用户默认目录，不能写入当前工作目录."""
    source = tmp_path / "source.ttf"
    source.write_bytes(b"font-content")
    user_home = tmp_path / "home"
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setattr(fonts.Path, "home", classmethod(lambda cls: user_home))
    monkeypatch.setattr(fonts.shutil, "which", lambda command: None)

    destination, changed = fonts._install_linux_font(source, force=False)

    expected = user_home / ".local" / "share" / "fonts" / "hscredit-font.ttf"
    assert (destination, changed) == (expected, True)
    assert not (working_directory / "fonts" / "hscredit-font.ttf").exists()


def test_init_setting_initializes_brand_font_without_changing_return_contract(monkeypatch):
    """init_setting 应执行字体初始化并保持默认返回 None."""
    calls = []
    monkeypatch.setattr(
        init_module,
        "initialize_bundled_font",
        lambda: calls.append(True) or "Alimama FangYuanTi VF",
    )

    assert init_module.init_setting() is None
    assert calls == [True]
    assert plt.rcParams["font.family"] == ["Alimama FangYuanTi VF"]


def test_init_setting_uses_kaiti_after_install_fallback(monkeypatch):
    """字体初始化回退后 matplotlib 也应使用楷体."""
    monkeypatch.setattr(init_module, "initialize_bundled_font", lambda: "楷体")

    assert init_module.init_setting() is None
    assert plt.rcParams["font.family"] == ["楷体"]


def test_init_setting_preserves_explicit_matplotlib_font_family(monkeypatch):
    """显式字体家族只覆盖 matplotlib，不跳过系统字体初始化."""
    calls = []
    monkeypatch.setattr(
        init_module,
        "initialize_bundled_font",
        lambda: calls.append(True) or "Alimama FangYuanTi VF",
    )

    init_module.init_setting(font_path="SimHei")

    assert calls == [True]
    assert plt.rcParams["font.family"] == ["SimHei"]


def test_init_setting_preserves_explicit_ttf_and_logger_return(monkeypatch):
    """显式 TTF 仍按内部家族名加载，logger=True 仍返回日志器."""
    monkeypatch.setattr(init_module, "initialize_bundled_font", lambda: "Alimama FangYuanTi VF")

    logger = init_module.init_setting(
        font_path=fonts.get_bundled_font_path(),
        logger=True,
        name="hscredit-font-test",
    )

    assert logger.name == "hscredit-font-test"
    assert plt.rcParams["font.family"] == ["Alimama FangYuanTi VF"]


def test_init_setting_survives_default_matplotlib_registration_failure(monkeypatch):
    """默认 TTF 的 Matplotlib 即时注册异常不得阻断环境初始化."""
    monkeypatch.setattr(init_module, "initialize_bundled_font", lambda: "Alimama FangYuanTi VF")
    monkeypatch.setattr(
        init_module.font_manager.fontManager,
        "addfont",
        lambda path: (_ for _ in ()).throw(RuntimeError("broken font cache")),
    )

    assert init_module.init_setting() is None
    assert plt.rcParams["font.family"] == ["Alimama FangYuanTi VF"]


def test_install_bundled_font_dispatches_by_system(monkeypatch, tmp_path):
    """应根据操作系统调用对应安装函数."""
    expected = (tmp_path / "font.ttf", True)
    monkeypatch.setattr(fonts, "_install_linux_font", lambda source, force: expected)

    assert fonts.install_bundled_font(system="Linux") == expected


def test_install_bundled_font_rejects_unsupported_system():
    """不支持的系统应返回中文错误."""
    with pytest.raises(RuntimeError, match="暂不支持"):
        fonts.install_bundled_font(system="UnknownOS")


def test_copy_font_is_idempotent(tmp_path):
    """相同字体重复安装时不应重复复制."""
    source = tmp_path / "source.ttf"
    destination = tmp_path / "fonts" / "target.ttf"
    source.write_bytes(b"font-content")

    assert fonts._copy_font(source, destination, force=False) is True
    assert fonts._copy_font(source, destination, force=False) is False
    assert destination.read_bytes() == b"font-content"


def test_cli_init_installs_font(monkeypatch, tmp_path, capsys):
    """init 命令应安装字体并输出安装位置."""
    destination = tmp_path / "hscredit-font.ttf"
    monkeypatch.setattr(cli, "install_bundled_font", lambda force=False: (destination, True))

    assert cli.main(["init"]) == 0
    output = capsys.readouterr().out
    assert "安装完成" in output
    assert str(destination) in output


def test_cli_init_reports_failure(monkeypatch, capsys):
    """字体安装失败时应返回非零退出码和中文错误."""
    monkeypatch.setattr(cli, "install_bundled_font", lambda force=False: (_ for _ in ()).throw(OSError("无权限")))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["init"])

    assert exc_info.value.code == 1
    assert "初始化失败：无权限" in capsys.readouterr().err
