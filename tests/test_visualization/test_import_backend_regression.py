import subprocess
import sys
import textwrap


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