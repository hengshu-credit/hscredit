"""系统字体安装与命令行测试."""

import pytest

from hscredit import cli
from hscredit.utils import fonts


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
