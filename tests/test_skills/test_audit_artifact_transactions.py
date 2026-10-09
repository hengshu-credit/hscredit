"""多制品发布、严格JSON与制品版本校验。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hscredit.skills_runtime.artifacts import ArtifactTransaction, summarize_dataframe
from hscredit.skills_runtime.io import InputResolver
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.utils.io import save_pickle


def test_publish_does_not_expose_files_before_success(tmp_path):
    with pytest.raises(RuntimeError):
        with ArtifactTransaction({"directory": str(tmp_path), "name": "report"}) as tx:
            first = tx.write_json("one.json", {"a": 1})
            item = tx.publish(first)
            assert not Path(item['path']).exists()
            raise RuntimeError("第二产物失败")
    assert not list(tmp_path.glob('*.manifest.json'))
    assert not list(tmp_path.glob('report-*'))


def test_versioned_commit_preserves_previous_artifacts(tmp_path):
    with ArtifactTransaction({"directory": str(tmp_path), "name": "report"}) as tx:
        item = tx.publish(tx.write_json("one.json", {"a": 1}))
    original = Path(item['path'])
    assert json.loads(original.read_text()) == {"a": 1}
    manifest_before = tx.manifest_path.read_bytes()
    with pytest.raises(RuntimeError):
        with ArtifactTransaction({"directory": str(tmp_path), "name": "report", "overwrite": True}) as tx2:
            tx2.publish(tx2.write_json("one.json", {"a": 2}))
            raise RuntimeError("后续失败")
    assert tx.manifest_path.read_bytes() == manifest_before
    assert json.loads(original.read_text()) == {"a": 1}


def test_summary_uses_strict_json():
    summary = summarize_dataframe(pd.DataFrame({"值": [np.nan, np.inf, -np.inf, 1.]}))
    json.dumps(summary, allow_nan=False)
    assert summary['preview'][0]['值'] is None


def test_skill_resolver_rejects_future_protocol(tmp_path):
    path = tmp_path / 'future.joblib'
    save_pickle({'format': 'hscredit-artifact', 'version': 999, 'object': {}}, path)
    with pytest.raises(SkillExecutionError, match='版本'):
        InputResolver().resolve({'kind': 'file', 'path': str(path), 'trusted': True})


def test_cleanup_failure_does_not_leave_publication_lock(tmp_path, monkeypatch):
    tx = ArtifactTransaction({'directory': str(tmp_path), 'name': 'cleanup'})
    def fail_cleanup():
        raise OSError('模拟清理失败')
    with pytest.warns(RuntimeWarning, match='清理失败'):
        with tx:
            monkeypatch.setattr(tx, '_cleanup', fail_cleanup)
    assert not tx._lock_path.exists()


def test_dataframe_pickle_obeys_table_limits(tmp_path):
    path = tmp_path / 'data.joblib'
    save_pickle(pd.DataFrame({'x': [1, 2, 3]}), path)
    with pytest.raises(SkillExecutionError, match='max_rows'):
        InputResolver().resolve({'kind': 'file', 'path': str(path), 'trusted': True, 'max_rows': 1})
