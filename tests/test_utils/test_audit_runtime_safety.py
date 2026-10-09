"""导入副作用与制品覆盖安全。"""
import warnings
from unittest.mock import patch
import pytest
from hscredit.utils import init_setting
from hscredit.core.binning import OptimalBinning
from hscredit.skills_runtime.bootstrap import ensure_environment
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.skills_runtime.io import InputResolver


def test_seed_zero_is_honored_without_hiding_warnings():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        with patch('hscredit.utils.random.seed_everything') as seed:
            init_setting(seed=0)
            seed.assert_called_once_with(0, freeze_torch=False)
        warnings.warn('审计警告可见', UserWarning)
    assert any('审计警告可见' in str(w.message) for w in captured)


def test_failed_generic_artifact_save_preserves_previous_file(tmp_path):
    target = tmp_path / 'old.joblib'
    target.write_bytes(b'previous successful artifact')
    binner = OptimalBinning(n_jobs=1)
    with patch('hscredit.utils.serialization.save_pickle', side_effect=RuntimeError('模拟失败')):
        with pytest.raises(RuntimeError):
            binner.save_artifact(target)
    assert target.read_bytes() == b'previous successful artifact'


def test_isolated_environment_respects_install_missing_false(tmp_path):
    config = {'skill': 'hsbin', 'hscredit': {'repository': 'https://invalid.local/repo', 'ref': 'main'}}
    request = {'operation': 'feature_bin_stats', 'environment': {'mode': 'isolated', 'install_missing': False}}
    with patch('hscredit.skills_runtime.bootstrap.install_requirement') as install:
        with pytest.raises(SkillExecutionError, match='禁止安装'):
            ensure_environment(config, request, cache_root=tmp_path)
    install.assert_not_called()


def test_csv_limits_refuse_truncation_and_preserve_column_order(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('a,b\n1,2\n3,4\n', encoding='utf-8')
    spec = {'kind': 'file', 'path': str(path), 'columns': ['b', 'a']}
    result = InputResolver().resolve(spec)
    assert result.columns.tolist() == ['b', 'a']
    with pytest.raises(SkillExecutionError, match='不静默截断'):
        InputResolver().resolve({**spec, 'max_rows': 1})
