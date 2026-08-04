"""并行运行时配置测试。"""

import pytest
from unittest.mock import patch

from hscredit.exceptions import ValidationError
from hscredit.utils import get_physical_cpu_count, resolve_n_jobs, validate_parallel_config


def test_physical_cpu_count_falls_back_to_joblib_without_keyword_support():
    """旧版 joblib 不支持物理核参数时仍应使用其 CPU 查询结果。"""
    def cpu_count(*args, **kwargs):
        if kwargs:
            raise TypeError("unexpected keyword argument")
        return 4

    with patch("hscredit.utils.parallel.joblib_cpu_count", side_effect=cpu_count):
        with patch("hscredit.utils.parallel.os.cpu_count", return_value=1):
            assert get_physical_cpu_count() == 4


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [(1, 1), (2, 1), (8, 7), (16, 13)],
)
def test_auto_workers_use_eighty_percent_and_leave_one_cpu(cpus, expected):
    """自动并行度应保留至少一个 CPU。"""
    assert resolve_n_jobs(-1, cpu_count=cpus) == expected


@pytest.mark.parametrize(
    ("value", "cpus", "expected"),
    [(1, 8, 1), (1.0, 8, 1), (2.0, 8, 2), (0.25, 8, 2), (0.26, 8, 3)],
)
def test_explicit_worker_forms(value, cpus, expected):
    """显式并行度支持正整数和 CPU 比例。"""
    assert resolve_n_jobs(value, cpu_count=cpus) == expected


def test_task_count_caps_workers():
    """任务数不足时不创建空闲工作进程。"""
    assert resolve_n_jobs(-1, task_count=2, cpu_count=16) == 2


@pytest.mark.parametrize(
    ("cpus", "available_budget", "expected"),
    [(8, 8, 7), (16, 8, 8)],
)
def test_available_budget_caps_automatic_workers(cpus, available_budget, expected):
    """嵌套预算只能收紧自动工作数，不能绕过保守 CPU 限制。"""
    assert resolve_n_jobs(-1, cpu_count=cpus, available_budget=available_budget) == expected


@pytest.mark.parametrize("available_budget", [True, 0, -1, 1.5, "2"])
def test_invalid_available_budget_raises_chinese_validation_error(available_budget):
    """嵌套预算必须为正整数。"""
    with pytest.raises(ValidationError, match="available_budget"):
        resolve_n_jobs(-1, cpu_count=8, available_budget=available_budget)


def test_serial_n_jobs_still_validates_available_budget():
    """兼容的串行模式也不能绕过嵌套预算校验。"""
    with pytest.raises(ValidationError, match="available_budget"):
        resolve_n_jobs(None, available_budget=0)


def test_large_integral_n_jobs_preserves_exact_worker_count():
    """大整数工作数不能因浮点转换而丢失精度。"""
    large_n_jobs = 2**53 + 1
    assert resolve_n_jobs(large_n_jobs, cpu_count=8) == large_n_jobs


@pytest.mark.parametrize("value", [True, 0, -2, 1.5, "2", object()])
def test_invalid_n_jobs_raises_chinese_validation_error(value):
    """无效并行度应抛出统一校验异常。"""
    with pytest.raises(ValidationError, match="n_jobs"):
        resolve_n_jobs(value, cpu_count=8)


def test_parallel_config_rejects_duplicate_worker_and_backend_sources():
    """并行配置不能重复声明工作数或后端。"""
    with pytest.raises(ValidationError, match="n_jobs"):
        validate_parallel_config(None, {"n_jobs": 2})
    with pytest.raises(ValidationError, match="backend"):
        validate_parallel_config("loky", {"backend": "threading"})


def test_parallel_config_preserves_supported_joblib_values():
    """校验后的配置应保留允许值且不修改调用方对象。"""
    source = {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}
    assert validate_parallel_config("loky", source) == source
    assert source == {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}


def test_parallel_config_rejects_non_string_keys_with_chinese_validation_error():
    """混合类型的未知配置键也必须返回统一中文校验错误。"""
    with pytest.raises(ValidationError, match="parallel_config"):
        validate_parallel_config(None, {"unknown": 1, 1: 2})
