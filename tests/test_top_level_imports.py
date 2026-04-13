import subprocess
import sys
import textwrap


def test_star_import_exposes_key_top_level_apis():
    code = textwrap.dedent(
        """
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
            'bin_plot',
            'data_info',
            'seed_everything',
            'ExcelWriter',
            'ks',
            'approval_badrate_tradeoff',
            'population_profile',
        ]

        missing = [name for name in expected if name not in namespace]
        assert not missing, missing

        assert '_build_overdue_labels' not in namespace
        assert namespace['Rule'].__module__ == 'hscredit.core.rules.rule'
        assert namespace['feature_summary'].__module__.startswith('hscredit.core.eda')
        """
    )

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr