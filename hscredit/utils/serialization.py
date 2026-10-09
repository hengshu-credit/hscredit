"""统一制品序列化协议.

为模型、评分卡、编码器和分箱器提供一致的 ``save_artifact`` /
``load_artifact`` 接口。原有的规则 JSON、映射 JSON 和模型原生格式接口继续保留，
统一制品接口用于完整对象的可靠往返保存。
"""

from pathlib import Path
from uuid import uuid4
from importlib.metadata import version, PackageNotFoundError
import hashlib
import warnings
import tempfile
from typing import Any, Dict, Optional, Type, TypeVar, Union

from ..exceptions import SerializationError
from .io import load_pickle, save_pickle


T = TypeVar("T", bound="ArtifactSerializableMixin")

ARTIFACT_FORMAT = "hscredit-artifact"
ARTIFACT_VERSION = 1


def _validate_digest(expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        char not in '0123456789abcdefABCDEF' for char in expected_sha256
    ):
        raise SerializationError("制品SHA256摘要必须为64位十六进制字符串")


def verify_artifact_digest(file, expected_sha256=None):
    """仅核对当前文件摘要；需反序列化时使用load_verified_pickle以避免路径替换竞态。"""
    if expected_sha256 is None:
        return
    _validate_digest(expected_sha256)
    digest = hashlib.sha256()
    with Path(file).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise SerializationError("制品摘要不匹配，已拒绝反序列化")


def load_verified_pickle(file, *, expected_sha256=None, **kwargs):
    """校验与反序列化读取同一个私有快照；分块复制避免模型大小级别的额外内存。"""
    if expected_sha256 is None:
        return load_pickle(file, **kwargs)
    _validate_digest(expected_sha256)
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(mode='w+b') as snapshot:
        with Path(file).open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                snapshot.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise SerializationError("制品摘要不匹配，已拒绝反序列化")
        snapshot.seek(0)
        return load_pickle(snapshot, source_name=str(file), **kwargs)


def unpack_artifact(payload, *, expected_type=None, metadata_policy='warn'):
    """验证可信对象的协议和声明；warn/strict/ignore只控制可选元数据政策。"""
    if metadata_policy not in {'warn', 'strict', 'ignore'}:
        raise SerializationError("metadata_policy 必须为 warn、strict 或 ignore")
    envelope = isinstance(payload, dict) and payload.get('format') == ARTIFACT_FORMAT
    if envelope:
        if payload.get('version') != ARTIFACT_VERSION or isinstance(payload.get('version'), bool):
            raise SerializationError(f"不支持的制品协议版本 {payload.get('version')}，当前支持版本为 {ARTIFACT_VERSION}")
        if 'object' not in payload:
            raise SerializationError("制品内容不完整，缺少 object 字段")
        obj = payload['object']
    else:
        obj = payload
    if expected_type is not None and not isinstance(obj, expected_type):
        raise SerializationError(f"制品对象类型为 {type(obj).__name__}，不能作为 {expected_type.__name__} 加载")
    if metadata_policy == 'ignore':
        return obj
    issues = []
    if not envelope:
        if metadata_policy == 'strict':
            issues.append('旧裸对象缺少可验证的制品元数据')
    else:
        actual_class = f'{type(obj).__module__}.{type(obj).__qualname__}'
        if 'class' not in payload and metadata_policy == 'strict':
            issues.append('缺少class声明')
        elif payload.get('class', actual_class) != actual_class:
            issues.append(f"class声明与实际类型不一致: {payload.get('class')} != {actual_class}")
        actual_kind = getattr(obj, 'artifact_kind', None)
        if actual_kind is not None and 'kind' in payload and payload['kind'] != actual_kind:
            issues.append('制品类别声明与实际对象不一致')
        if 'features' in payload and payload['features'] is not None:
            actual_features = getattr(obj, 'feature_names_in_', None)
            import pandas as pd
            try:
                equal = actual_features is not None and pd.Index(payload['features']).equals(pd.Index(actual_features))
            except (TypeError, ValueError):
                equal = False
            if not equal:
                issues.append('features声明与实际字段不一致')
        dependencies = payload.get('dependencies', {})
        if not isinstance(dependencies, dict):
            issues.append('依赖版本声明不是字典')
        else:
            for package, declared_version in dependencies.items():
                try:
                    installed = version(str(package))
                except PackageNotFoundError:
                    installed = '未安装'
                if str(declared_version) != installed:
                    issues.append(f'依赖{package}的声明版本{declared_version}与当前{installed}不同')
    if issues:
        message = '制品元数据不一致: ' + '；'.join(issues)
        if metadata_policy == 'strict':
            raise SerializationError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return obj


class ArtifactSerializableMixin:
    """hscredit 完整对象制品序列化混入类.

    子类无需实现额外方法即可获得统一持久化能力。制品中同时记录对象类型、
    制品类别和协议版本，加载时会校验目标类型，避免误加载其他对象。
    """

    artifact_kind = "通用制品"

    def get_artifact_metadata(self) -> Dict[str, Any]:
        """返回不包含对象本体的制品元数据."""
        dependencies = {}
        for package in ('hscredit', 'numpy', 'pandas', 'scikit-learn'):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                continue
        return {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "kind": self.artifact_kind,
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "dependencies": dependencies,
            "history_policy": getattr(self, 'history_policy', 'unspecified'),
            "features": list(getattr(self, 'feature_names_in_', [])) if getattr(self, 'feature_names_in_', None) is not None else None,
        }

    def save_artifact(
        self,
        file: Union[str, Path],
        engine: str = "joblib",
        compression: Optional[str] = None,
        **kwargs,
    ) -> str:
        """保存完整 hscredit 对象.

        :param file: 输出文件路径
        :param engine: joblib、pickle、dill 或 cloudpickle
        :param compression: 可选压缩格式
        :param kwargs: 传递给 :func:`hscredit.utils.save_pickle`
        :return: 保存后的文件路径
        """
        path = Path(file)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            **self.get_artifact_metadata(),
            "object": self,
        }
        temporary = path.with_name(f'.{uuid4().hex}.{path.name}')
        try:
            save_pickle(payload, temporary, engine=engine, compression=compression, **kwargs)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)

    @classmethod
    def load_artifact(
        cls: Type[T],
        file: Union[str, Path],
        engine: str = "auto",
        compression: Optional[str] = None,
        *,
        metadata_policy: str = 'warn',
        trusted: bool = True,
        expected_sha256: Optional[str] = None,
        **kwargs,
    ) -> T:
        """加载并校验完整 hscredit 对象.

        为兼容旧文件，也接受直接保存、未包装制品元数据的对象。
        此接口仅用于可信pickle/joblib。trusted=False在读取前拒绝；默认True保留旧调用。
        metadata_policy控制元数据不一致的warn/strict/ignore行为；strict拒绝无元数据裸对象。
        expected_sha256可用于读取前核对可信来源的摘要，不会使不可信pickle变安全。
        """
        if trusted is not True:
            raise SerializationError("只能反序列化可信来源的制品，请显式确认 trusted=True")
        if metadata_policy not in {'warn', 'strict', 'ignore'}:
            raise SerializationError("metadata_policy 必须为 warn、strict 或 ignore")
        payload = load_verified_pickle(
            file,
            expected_sha256=expected_sha256,
            engine=engine,
            compression=compression,
            **kwargs,
        )

        return unpack_artifact(payload, expected_type=cls, metadata_policy=metadata_policy)

