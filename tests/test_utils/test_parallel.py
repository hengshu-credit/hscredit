"""并行任务参数测试."""

from unittest.mock import patch

from hscredit.core.models import RandomForestRiskModel
from hscredit.utils import resolve_n_jobs


def test_resolve_n_jobs_minus_one_reserves_one_cpu():
    with patch("hscredit.utils.parallel.get_physical_cpu_count", return_value=8):
        assert resolve_n_jobs(-1) == 7


def test_resolve_n_jobs_minus_one_keeps_single_core_usable():
    with patch("hscredit.utils.parallel.get_physical_cpu_count", return_value=1):
        assert resolve_n_jobs(-1) == 1


def test_resolve_n_jobs_handles_unknown_cpu_count():
    with patch("hscredit.utils.parallel.get_physical_cpu_count", return_value=1):
        assert resolve_n_jobs(-1) == 1


def test_resolve_n_jobs_preserves_explicit_values():
    assert resolve_n_jobs(None) is None
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(4) == 4


def test_risk_model_uses_resolved_cpu_count():
    with patch("hscredit.utils.parallel.get_physical_cpu_count", return_value=12):
        model = RandomForestRiskModel(n_jobs=-1)
    assert model.n_jobs == 10
