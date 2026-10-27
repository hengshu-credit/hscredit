"""规则表样式的新旧依赖分支与实际 HTML 渲染回归。"""

from types import SimpleNamespace
import warnings

import matplotlib as mpl
import pandas as pd
import pytest
from packaging.version import Version

from hscredit._compat import install_matplotlib_colormap_compat, map_styler, pandas_styler_map_api
from hscredit.utils import style_rule_table


@pytest.mark.parametrize(
    "version, method",
    [("2.0.0", "applymap"), ("2.0.3", "applymap"), ("2.1.0", "map"), ("2.3.3", "map"), ("3.0.6", "map")],
)
def test_styler_map_uses_version_selected_method(monkeypatch, version, method):
    """对象只提供对应版本的接口，验证参数和返回值完整保留。"""
    monkeypatch.setattr(pd, "__version__", version)
    assert pandas_styler_map_api(Version(version)) == method
    received = []
    result = object()

    def apply_style(func, **kwargs):
        received.append((func, kwargs))
        return result

    def color(value, color):
        return f"color: {color}"

    styler = SimpleNamespace(**{method: apply_style})
    assert map_styler(styler, color, subset=["是否叶子"], color="red") is result
    assert received == [(color, {"subset": ["是否叶子"], "color": "red"})]


def test_styler_map_preserves_callback_error():
    def fail(value):
        raise ValueError("样式函数错误")

    styler = map_styler(pd.DataFrame({"字段": [1]}).style, fail)
    with pytest.raises(ValueError, match="样式函数错误"):
        styler.to_html()


@pytest.mark.parametrize("version", ["3.3.0", "3.4.3", "3.5.0", "3.5.3", "3.6.0"])
def test_legacy_colormap_queries_render_real_styler(monkeypatch, version):
    """重建旧版接口缺口，实际渲染默认/命名/对象色图，覆盖延迟执行。"""
    # 保存当前版本的查询器，模拟旧接口，不改动当前环境的发行版。
    current_get_cmap = mpl.colormaps.get_cmap if Version(mpl.__version__) >= Version("3.6.1") else mpl.cm.get_cmap
    monkeypatch.setattr(mpl.cm, "get_cmap", current_get_cmap, raising=False)
    monkeypatch.setattr(mpl, "_hscredit_colormap_compat", False, raising=False)
    if Version(version) < Version("3.5"):
        monkeypatch.delattr(mpl, "colormaps", raising=False)
    else:

        class RegistryWithoutGetCmap:
            def __getitem__(self, name):
                return current_get_cmap(name)

        monkeypatch.setattr(mpl, "colormaps", RegistryWithoutGetCmap(), raising=False)

    install_matplotlib_colormap_compat(mpl, Version(version))
    registry = mpl.colormaps
    install_matplotlib_colormap_compat(mpl, Version(version))
    assert mpl.colormaps is registry
    color = current_get_cmap("RdYlGn")
    for cmap in (None, "RdYlGn", color):
        html = pd.DataFrame({"坏样本率": [0.1, 0.3]}).style.background_gradient(cmap=cmap).to_html()
        assert "background-color:" in html
    # 兼容查询返回副本，调用者不能通过修改色图污染其他表格。
    assert registry.get_cmap(color) is not color


@pytest.mark.parametrize("version", ["3.2.2", "3.6.1", "3.10.1", "3.11.2"])
def test_colormap_compat_does_not_touch_outside_supported_range(version):
    module = SimpleNamespace()
    install_matplotlib_colormap_compat(module, Version(version))
    assert vars(module) == {}


@pytest.mark.parametrize("version, expected", [(None, False), ("3.5.3", True), ("3.6.0", True), ("3.6.1", False)])
def test_prepare_matplotlib_uses_installed_version(monkeypatch, version, expected):
    import hscredit._compat as compat

    installed = []
    value = Version(version) if version else None
    monkeypatch.setattr(compat, "installed_version", lambda name: value)
    monkeypatch.setattr(
        compat, "install_matplotlib_colormap_compat", lambda module, ver: installed.append((module, ver))
    )
    compat.prepare_dependency("matplotlib")
    assert installed == ([(mpl, value)] if expected else [])


def test_style_rule_table_renders_colors_formats_and_missing_values():
    table = pd.DataFrame(
        {
            "节点编号": [0, 1, 2],
            "是否叶子": ["否", "是", "是"],
            "指标含义": ["全部样本", "评分 <= 600", "评分 > 600"],
            "分箱": ["命中"] * 3,
            "样本总数": [10, 4, 6],
            "样本占比": [1.0, 0.4, 0.6],
            "坏样本数": [2, 1, 1],
            "坏样本率": [0.2, 0.25, 1 / 6],
            "LIFT值": [1.0, 1.25, 5 / 6],
            "风险拒绝比": [float("nan"), 0.5, 0.5],
        }
    )
    original = table.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        html = style_rule_table(table, overall_badrate=0.2).to_html()
    pd.testing.assert_frame_equal(table, original)
    assert "#e8f5e9" in html and "#fff8e1" in html
    assert "20.00%" in html and "1.25" in html
    assert ">分箱<" not in html
    assert ">-<" in html
