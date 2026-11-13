import subprocess
import sys
import textwrap


def test_notebook_progress_updates_installed_widget(monkeypatch):
    from types import SimpleNamespace

    import IPython
    from ipywidgets import FloatProgress
    from hscredit._progress import create_progress_bar

    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(config={"IPKernelApp": {}}))
    with create_progress_bar(total=3) as bar:
        bar.update(2)
        bar.refresh()
        widget = next(child for child in bar.container.children if isinstance(child, FloatProgress))
        assert (widget.value, widget.max) == (2, 3)


def test_notebook_without_widgets_imports_and_reports_text_progress():
    code = textwrap.dedent("""
        import io
        import sys
        import warnings
        from contextlib import redirect_stderr
        from types import SimpleNamespace
        import IPython
        from tqdm import TqdmWarning

        IPython.get_ipython = lambda: SimpleNamespace(config={"IPKernelApp": {}})
        sys.modules["ipywidgets"] = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", TqdmWarning)
            import hscredit
            from hscredit.database import QueryStream
            assert "tqdm.notebook" not in sys.modules

            import pandas as pd
            data = pd.Series([3, 7])
            output = io.StringIO()
            with redirect_stderr(output):
                result = data.hscredit(n_jobs=1, bar=True).apply(lambda value: value + 1)
            assert result.tolist() == [4, 8]
            assert "2/2" in output.getvalue()
        assert not any(issubclass(w.category, TqdmWarning) for w in caught), caught
        """)
    subprocess.run([sys.executable, "-X", "utf8", "-c", code], check=True)


def test_star_import_exposes_key_top_level_apis():
    code = textwrap.dedent("""
        namespace = {}
        exec('from hscredit import *', namespace)

        expected = [
            'Rule',
            'LogisticRegression',
            'ScoreCard',
            'OptimalBinning',
            'WOEEncoder',
            'feature_bin_stats',
            'feature_efficiency_analysis',
            'ruleset_analysis',
            'germancredit',
            'hscredit_dataset',
            'bin_plot',
            'data_info',
            'seed_everything',
            'ExcelWriter',
            'ks',
            'approval_badrate_tradeoff',
            'population_profile',
            'BalancedFocalLoss',
            'ExpectedProfitLoss',
            'RankingAUCProxyLoss',
            'KSFocusedLoss',
            'TopKBadCaptureLoss',
            'AmountWeightedLoss',
            'ExpectedValueLoss',
            'NGBoostLossAdapter',
            'Database',
            'WriteResult',
            'DatabaseCapabilityError',
            'register_adapter',
        ]

        missing = [name for name in expected if name not in namespace]
        assert not missing, missing

        assert '_build_overdue_labels' not in namespace
        assert namespace['Rule'].__module__ == 'hscredit.core.rules.rule'
        assert namespace['feature_summary'].__module__.startswith('hscredit.core.eda')
        assert namespace['Database'].__module__ == 'hscredit.database.client'
        """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_info_does_not_list_implemented_metrics_as_pending():
    code = textwrap.dedent("""
        import hscredit

        hscredit.info()
        """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "待实现模块" not in result.stdout
    assert "core.encoding" not in result.stdout
    assert "core.metrics: 指标计算" not in result.stdout


def test_core_aggregates_public_subpackage_apis():
    code = textwrap.dedent("""
        import hscredit.core as core
        from hscredit.core import binning, metrics, selectors

        missing = {}
        for module_name, module in {
            'binning': binning,
            'metrics': metrics,
            'selectors': selectors,
        }.items():
            current = [name for name in module.__all__ if name not in core.__all__]
            if current:
                missing[module_name] = current

        assert not missing, missing
        assert core.CPSATBinning is binning.CPSATBinning
        assert core.OptimalBinning2D is binning.OptimalBinning2D
        assert core.StabilityAwareSelector is selectors.StabilityAwareSelector
        assert core.selection_report is selectors.selection_report
        assert core.compute_bin_stats is metrics.compute_bin_stats
        """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
