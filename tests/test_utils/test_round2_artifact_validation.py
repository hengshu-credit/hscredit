"""可信制品的元数据、兼容政策与读取前摘要核验。"""
import hashlib
from unittest.mock import patch
import pytest
from hscredit.exceptions import SerializationError
from hscredit.utils.serialization import ArtifactSerializableMixin
from hscredit.utils.io import save_pickle, load_pickle
from hscredit.skills_runtime.io import InputResolver
from hscredit.skills_runtime.errors import SkillExecutionError


class ExampleArtifact(ArtifactSerializableMixin):
    def __init__(self):
        self.feature_names_in_ = ['字段']


@pytest.mark.parametrize('key,value', [('class', 'wrong.Class'), ('features', ['其他字段']), ('dependencies', {'numpy': '999.0'})])
def test_strict_metadata_rejects_inconsistent_envelope(tmp_path, key, value):
    path = tmp_path / 'sample.joblib'
    ExampleArtifact().save_artifact(path)
    payload = load_pickle(path)
    payload[key] = value
    save_pickle(payload, path)
    with pytest.raises(SerializationError, match='元数据'):
        ExampleArtifact.load_artifact(path, metadata_policy='strict')
    with pytest.warns(UserWarning, match='元数据'):
        restored = ExampleArtifact.load_artifact(path, metadata_policy='warn')
    assert restored.feature_names_in_ == ['字段']


def test_digest_and_trust_are_checked_before_unpickling(tmp_path):
    path = tmp_path / 'bad.joblib'
    path.write_bytes(b'not a pickle')
    with patch('hscredit.utils.serialization.load_pickle') as loader:
        with pytest.raises(SerializationError, match='摘要'):
            ExampleArtifact.load_artifact(path, expected_sha256='0' * 64)
        with pytest.raises(SerializationError, match='可信'):
            ExampleArtifact.load_artifact(path, trusted=False)
    loader.assert_not_called()


def test_skill_and_direct_api_share_validation(tmp_path):
    path = tmp_path / 'sample.joblib'
    ExampleArtifact().save_artifact(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = InputResolver().resolve({'kind': 'file', 'path': str(path), 'trusted': True,
                                      'metadata_policy': 'strict', 'sha256': digest})
    assert isinstance(result, ExampleArtifact)
    with pytest.raises(SkillExecutionError, match='摘要'):
        InputResolver().resolve({'kind': 'file', 'path': str(path), 'trusted': True, 'sha256': '0' * 64})


@pytest.mark.parametrize('engine', ['joblib', 'pickle', 'cloudpickle', 'dill'])
@pytest.mark.parametrize('suffix', ['', '.gz', '.bz2', '.xz'])
def test_verified_snapshot_is_loaded_even_if_original_path_changes(tmp_path, engine, suffix):
    if engine in {'cloudpickle', 'dill'}:
        pytest.importorskip(engine)
    path = tmp_path / ('sample.' + engine + suffix)
    original = ExampleArtifact()
    original.save_artifact(path, engine=engine)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def replace_path_before_load(snapshot, **kwargs):
        replacement = ExampleArtifact()
        replacement.feature_names_in_ = ['不能加载的替换对象']
        replacement.save_artifact(path, engine=engine)
        return load_pickle(snapshot, **kwargs)

    with patch('hscredit.utils.serialization.load_pickle', side_effect=replace_path_before_load):
        restored = ExampleArtifact.load_artifact(path, engine=engine, expected_sha256=digest)
    assert restored.feature_names_in_ == ['字段']
