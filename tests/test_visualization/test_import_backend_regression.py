import subprocess
import sys
import textwrap

from hscredit.core.viz import utils as viz_utils


def test_import_hscredit_does_not_change_matplotlib_backend():
    code = textwrap.dedent(
        """
        import matplotlib
        matplotlib.use('svg')
        import matplotlib.pyplot as plt

        before = matplotlib.get_backend().lower()

        import hscredit  # noqa: F401

        after = matplotlib.get_backend().lower()
        print(before)
        print(after)
        assert after == before, (before, after)
        """
    )

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_create_subplots_keeps_ratio_options_compatible_with_matplotlib_3_5(monkeypatch):
    """旧版 pyplot.subplots 不接收独立 ratio 参数，必须通过 gridspec_kw 传递。"""
    captured = []
    sentinel = object()

    def legacy_subplots(
        nrows=1,
        ncols=1,
        *,
        sharex=False,
        sharey=False,
        squeeze=True,
        subplot_kw=None,
        gridspec_kw=None,
        **figure_kwargs,
    ):
        if "width_ratios" in figure_kwargs or "height_ratios" in figure_kwargs:
            raise AttributeError("Figure has no property 'width_ratios'")
        captured.append((gridspec_kw, figure_kwargs))
        return sentinel, None

    monkeypatch.setattr(viz_utils.plt, "subplots", legacy_subplots)

    figure, _ = viz_utils._create_subplots(figsize=(10, 5))
    ratio_figure, _ = viz_utils._create_subplots(
        1,
        2,
        width_ratios=[2, 1],
        height_ratios=[1],
        gridspec_kw={"wspace": 0.2},
    )

    assert figure is sentinel
    assert ratio_figure is sentinel
    assert captured == [
        (None, {"figsize": (10, 5)}),
        ({"wspace": 0.2, "width_ratios": [2, 1], "height_ratios": [1]}, {}),
    ]
