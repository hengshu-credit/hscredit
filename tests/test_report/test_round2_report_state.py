"""报告同名章节重跑不能混用旧内容、状态和诊断。"""
import pandas as pd
import pytest
from hscredit.report.result import ReportResult, ReportGenerationError


def test_success_failure_success_is_one_coherent_snapshot():
    result = ReportResult()
    result.plan('章节')
    result.run('章节', lambda: pd.DataFrame({'值': [1, 2]}))
    result.run('章节', lambda: 1 / 0)
    assert '章节' not in result
    assert result.sections['章节'].output_rows is None
    assert not result.complete
    result.success('章节', pd.DataFrame({'值': [3]}))
    assert result.complete and result.sections['章节'].reason == ''
    assert result.sections['章节'].output_rows == 1
    assert result['章节']['值'].tolist() == [3]


def test_skip_and_replan_clear_prior_output():
    result = ReportResult()
    result.plan('章节')
    result.success('章节', pd.DataFrame({'值': [1]}))
    result.skip('章节', '本次不适用')
    assert '章节' not in result
    result.plan('章节')
    assert result.sections['章节'].reason == ''
    assert not result.complete


def test_strict_retry_exception_never_retains_old_table():
    result = ReportResult(mode='strict')
    result.plan('章节')
    result.success('章节', pd.DataFrame({'值': [1]}))
    with pytest.raises(ReportGenerationError):
        result.run('章节', lambda: None)
    assert '章节' not in result
