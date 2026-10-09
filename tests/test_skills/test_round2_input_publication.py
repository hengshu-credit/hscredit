"""对象引用限额和发布失败的故障注入回归。"""
import json
from pathlib import Path
import warnings
import pandas as pd
import pytest
from hscredit.skills_runtime.artifacts import ArtifactTransaction
from hscredit.skills_runtime.io import InputResolver
from hscredit.skills_runtime.objects import ObjectRegistry
from hscredit.skills_runtime.errors import SkillExecutionError


def test_object_projection_and_limits_match_files():
    frame = pd.DataFrame({'a': [1, 2, 3], 'extra': ['data'] * 3})
    resolver = InputResolver(ObjectRegistry({'d': frame}))
    assert resolver.resolve({'kind': 'object_ref', 'ref': 'd'}) is frame
    projected = resolver.resolve({'kind': 'object_ref', 'ref': 'd', 'columns': ['a']})
    assert projected.columns.tolist() == ['a']
    for option in ('max_rows', 'max_frame_bytes'):
        with pytest.raises(SkillExecutionError):
            resolver.resolve({'kind': 'object_ref', 'ref': 'd', option: 1})
    with pytest.raises(SkillExecutionError):
        resolver.resolve({'kind': 'object_ref', 'ref': 'd', 'max_file_bytes': 1})


def test_lock_write_failure_releases_owned_lock(tmp_path, monkeypatch):
    original_open = Path.open
    class BrokenWrite:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def write(self, value):
            raise OSError('模拟磁盘满')
        def __getattr__(self, name):
            return getattr(self.handle, name)
    def fail_write(path, mode='r', *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        return BrokenWrite(handle) if mode == 'x' else handle
    monkeypatch.setattr(Path, 'open', fail_write)
    tx = ArtifactTransaction({'directory': str(tmp_path), 'name': 'lock'})
    with pytest.raises((OSError, SkillExecutionError)):
        with tx:
            pass
    assert not tx._lock_path.exists()
    assert not tx._owns_lock


def test_cleanup_warning_as_error_preserves_primary_failure(tmp_path, monkeypatch):
    tx = ArtifactTransaction({'directory': str(tmp_path), 'name': 'cleanup'})
    def fail():
        raise OSError('模拟清理失败')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError, match='原始错误'):
            with tx:
                monkeypatch.setattr(tx, '_cleanup', fail)
                raise ValueError('原始错误')
    assert not tx._lock_path.exists()
    assert tx.cleanup_errors


def test_publication_state_and_owner_metadata(tmp_path):
    tx = ArtifactTransaction({'directory': str(tmp_path), 'name': 'owner'})
    with tx:
        owner = json.loads(tx._lock_path.read_text(encoding='utf-8'))
        assert owner['pid'] > 0 and owner['owner_token']
        tx.publish(tx.write_json('value.json', {'value': 1}))
        assert not tx.committed
    assert tx.committed and tx.manifest_path.is_file()
