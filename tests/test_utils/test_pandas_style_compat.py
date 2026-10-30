"""规则表样式的新旧依赖分支与实际 HTML 渲染回归。"""

from types import SimpleNamespace
import pickle
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

from hscredit._compat import install_matplotlib_colormap_compat, map_styler, pandas_styler_map_api
from hscredit.utils import BinTableDisplay, style_bin_table, style_rule_table


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


@pytest.mark.parametrize("multi_level", [False, True])
def test_dataframe_show_renders_complex_values_without_changing_data(monkeypatch, tmp_path, multi_level):
    """列表指标不能计算分位数；实际执行 HTML 格式化以覆盖延迟异常。"""
    table = pd.DataFrame(
        {
            "方法": ["等频", "决策树", "手工", "特殊值", "缺失"],
            "坏样本率序列": [[0.1, 0.2], [0.2], [], (0.3, 0.4), None],
            "LIFT值序列": [np.array([1.0, 2.0]), np.array([]), {"低": 1.2}, {1, 2}, pd.NA],
            "坏样本率": [[0.1, 0.2], np.array([0.3]), 0.25, "文本", np.nan],
            "配置": [{"切分点": [100, 200]}, (1, 2), pd.Series([1, 2]), ["<b>原样文本</b>"], None],
        }
    )
    if multi_level:
        table.columns = pd.MultiIndex.from_product([["比较结果"], table.columns])
    # pandas 2.0 的 assert_frame_equal 不支持 set；序列化快照也覆盖容器内部的改动。
    original = pickle.dumps(table)
    rendered = []
    monkeypatch.setattr("IPython.display.display", lambda obj: rendered.append(obj.to_html()))

    result = table.show()

    assert isinstance(result, BinTableDisplay)
    assert len(rendered) == 1
    html = rendered[0]
    assert "[0.1, 0.2]" in html and "[1. 2.]" in html
    assert "切分点" in html and "25.00%" in html
    assert "&lt;b&gt;原样文本&lt;/b&gt;" in html and "<b>原样文本</b>" not in html
    assert ">-<" in html
    assert "linear-gradient" not in html
    output = tmp_path / "复杂类型.html"
    assert result.export_html(str(output)) is result
    assert "切分点" in output.read_text(encoding="utf-8")
    assert pickle.dumps(table) == original


@pytest.mark.parametrize("multi_level", [False, True])
def test_show_keeps_numeric_bars_formats_and_precision(multi_level):
    table = pd.DataFrame(
        {
            "样本总数": pd.Series([10, 20, pd.NA], dtype="Int64"),
            "坏样本率": pd.Series([0.125, 0.25, pd.NA], dtype="Float64"),
            "LIFT值": [1.25, 2.5, np.nan],
            "分档KS值": [0.1234, 0.2345, np.nan],
        }
    )
    if multi_level:
        table.columns = pd.MultiIndex.from_product([["MOB1>3"], table.columns])
    html = style_bin_table(table, precision={"坏样本率": 3}).to_html()
    assert "12.500%" in html and "12.34%" in html
    assert ">1.25<" in html and "125.00%" not in html
    assert ">10<" in html and ">-<" in html
    assert "linear-gradient" in html
    plain = style_bin_table(
        table, percent_format=False, highlight_bad_rate=False, highlight_lift=False, highlight_ks=False
    ).to_html()
    assert "linear-gradient" not in plain


@pytest.mark.parametrize("styler_factory", [style_bin_table, style_rule_table])
@pytest.mark.parametrize("multi_level", [False, True])
def test_table_header_stripes_and_hover(styler_factory, multi_level):
    table = pd.DataFrame({"指标含义": ["低分", "高分"], "样本总数": [10, 20]})
    if multi_level:
        table.columns = pd.MultiIndex.from_product([["结果"], table.columns])
    html = styler_factory(table).to_html()
    assert "thead th" in html and "#2639E9" in html and "#ffffff" in html
    assert "tbody tr:nth-child(odd)" in html
    assert "tbody tr:nth-child(even)" in html and "#f8f9ff" in html
    assert "tbody tr:hover" in html and "#e8eaff" in html
    assert "tbody tr:hover td" in html and "tbody tr:hover th" in html


def test_show_explicit_columns_override_automatic_formats_and_flags(monkeypatch):
    table = pd.DataFrame({"自定义率": [0.125, 0.5], "坏样本率": [0.25, 0.75], "自定义值": [1.0, 4.0]})
    rendered = []
    monkeypatch.setattr("IPython.display.display", lambda obj: rendered.append(obj))
    table.show(
        percent_cols=["自定义率"],
        condition_cols=["自定义值"],
        percent_format=False,
        highlight_bad_rate=False,
        condition_color="123456",
        theme_color="#345678",
        precision={"自定义率": 3},
    )
    styler = rendered[0]
    html = styler.to_html()
    assert "12.500%" in html and "25.00%" not in html
    assert "#345678" in html and "#123456" in html
    styles = styler._compute().ctx
    assert any("linear-gradient" in value for _, value in styles[1, 2])
    assert not any("linear-gradient" in value for _, value in styles[1, 1])


def test_show_empty_lists_disable_automatic_formats():
    table = pd.DataFrame({"坏样本率": [0.1, 0.2], "LIFT值": [1.0, 2.0]})
    html = style_bin_table(table, percent_cols=[], condition_cols=[], color_cols=[]).to_html()
    assert "10.00%" not in html and "linear-gradient" not in html


@pytest.mark.parametrize("multi_level", [False, True])
def test_show_color_blocks_override_bars_and_resolve_column_colors(multi_level):
    table = pd.DataFrame({"坏样本率": [0.0, 0.2, 1.0], "得分": [0.0, 1.0, 2.0]})
    if multi_level:
        table.columns = pd.MultiIndex.from_product([["结果"], table.columns])
    score = ("结果", "得分") if multi_level else "得分"
    colors = {"坏样本率": ["00FF00", "FFFF00", "FF0000"], "得分": "111111", score: "123456"}
    styler = style_bin_table(
        table,
        condition_cols=["坏样本率", score],
        color_cols=["坏样本率"],
        condition_color=colors,
    )
    html = styler.to_html()
    styles = styler._compute().ctx
    for row, expected in enumerate(["#00ff00", "#ffff00", "#ff0000"]):
        assert dict(styles[row, 0])["background-color"] == expected
        assert not any("linear-gradient" in value for _, value in styles[row, 0])
    assert "#123456" in html


def test_show_full_tuple_selector_and_color_priority():
    table = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=pd.MultiIndex.from_tuples([("甲", "率"), ("乙", "率")]))
    styler = style_bin_table(
        table,
        percent_cols=[("甲", "率")],
        condition_cols=["率"],
        condition_color={"率": "111111", "甲": "222222", ("甲", "率"): "333333"},
    )
    html = styler.to_html()
    assert "10.00%" in html and "20.00%" not in html
    styles = styler._compute().ctx
    assert "#333333" in str(styles[1, 0]) and "#111111" in str(styles[1, 1])


def test_show_explicit_columns_survive_compact_mode_and_complex_values():
    table = pd.DataFrame({"自定义率": [0.1, 0.2], "列表": [[1, 2], [3]], "隐藏列": [1, 2]})
    html = style_bin_table(
        table,
        compact=True,
        percent_cols=["自定义率", "列表"],
        condition_cols=["列表"],
        color_cols=["列表"],
        fill=False,
    ).to_html()
    assert "10.00%" in html and "[1, 2]" in html and "隐藏列" not in html
    assert "linear-gradient" not in html
    assert "tbody tr:nth-child(even)" not in html


@pytest.mark.parametrize("value", ["112233", ["000000", "112233"], {"得分": "112233"}])
def test_show_condition_color_forms(value):
    table = pd.DataFrame({"得分": [0, 1, 2]})
    html = style_bin_table(table, condition_cols=["得分"], condition_color=value).to_html()
    assert "#112233" in html


def test_show_color_blocks_preserve_missing_and_constant_columns():
    table = pd.DataFrame({"常量": [1.0, 1.0, np.nan], "缺失": [np.nan] * 3})
    styler = style_bin_table(table, color_cols=["常量", "缺失"], condition_color="123456")
    assert ">-<" in styler.to_html()
    styles = styler._compute().ctx
    assert dict(styles[0, 0])["background-color"] == dict(styles[1, 0])["background-color"]
    assert "background-color" not in dict(styles[2, 0])
    assert "background-color" not in dict(styles[0, 1])


def test_show_two_color_scale_and_default_color_fallback():
    table = pd.DataFrame({"色阶": [-2.0, 0.0, 2.0], "数据条": [1, 2, 3]})
    styler = style_bin_table(
        table,
        color_cols=["色阶"],
        condition_cols=["数据条"],
        condition_color={"色阶": ["FF0000", "0000FF"]},
    )
    html = styler.to_html()
    styles = styler._compute().ctx
    assert dict(styles[0, 0])["background-color"] == "#ff0000"
    assert dict(styles[2, 0])["background-color"] == "#0000ff"
    assert "#F76E6C" in html


@pytest.mark.parametrize("kwargs", [{"theme_color": "invalid"}, {"condition_cols": ["得分"], "condition_color": "red"}])
def test_show_invalid_colors_report_chinese_error(kwargs):
    with pytest.raises(ValueError, match="六位十六进制"):
        style_bin_table(pd.DataFrame({"得分": [0.0, 1.0]}), **kwargs).to_html()
