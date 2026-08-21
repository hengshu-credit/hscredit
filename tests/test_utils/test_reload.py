"""模块重载工具测试。"""

import importlib
import sys


def test_reload_is_exported_from_public_namespaces():
    """公开命名空间应统一导出 reload。"""
    from hscredit import reload as root_reload
    from hscredit.utils import reload as utils_reload

    assert root_reload is utils_reload


def test_reload_reimports_module_and_evicts_submodules(tmp_path, monkeypatch):
    """reload 应重新导入主模块，并从缓存中移除其子模块。"""
    from hscredit import reload

    package_name = "hscredit_reload_fixture"
    child_name = f"{package_name}.child"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("marker = object()\n", encoding="utf-8")
    (package_dir / "child.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    try:
        original = importlib.import_module(package_name)
        importlib.import_module(child_name)

        reloaded = reload(package_name)

        assert reloaded is sys.modules[package_name]
        assert reloaded is not original
        assert child_name not in sys.modules
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
