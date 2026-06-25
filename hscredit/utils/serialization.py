"""统一制品序列化协议.

为模型、评分卡、编码器和分箱器提供一致的 ``save_artifact`` /
``load_artifact`` 接口。原有的规则 JSON、映射 JSON 和模型原生格式接口继续保留，
统一制品接口用于完整对象的可靠往返保存。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from ..exceptions import SerializationError
from .io import load_pickle, save_pickle


T = TypeVar("T", bound="ArtifactSerializableMixin")

ARTIFACT_FORMAT = "hscredit-artifact"
ARTIFACT_VERSION = 1


class ArtifactSerializableMixin:
    """hscredit 完整对象制品序列化混入类.

    子类无需实现额外方法即可获得统一持久化能力。制品中同时记录对象类型、
    制品类别和协议版本，加载时会校验目标类型，避免误加载其他对象。
    """

    artifact_kind = "通用制品"

    def get_artifact_metadata(self) -> Dict[str, Any]:
        """返回不包含对象本体的制品元数据."""
        return {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "kind": self.artifact_kind,
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
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
        return save_pickle(
            payload,
            path,
            engine=engine,
            compression=compression,
            **kwargs,
        )

    @classmethod
    def load_artifact(
        cls: Type[T],
        file: Union[str, Path],
        engine: str = "auto",
        compression: Optional[str] = None,
        **kwargs,
    ) -> T:
        """加载并校验完整 hscredit 对象.

        为兼容旧文件，也接受直接保存、未包装制品元数据的对象。
        """
        payload = load_pickle(
            file,
            engine=engine,
            compression=compression,
            **kwargs,
        )

        if isinstance(payload, dict) and payload.get("format") == ARTIFACT_FORMAT:
            version = payload.get("version")
            if version != ARTIFACT_VERSION:
                raise SerializationError(
                    f"不支持的制品协议版本 {version}，当前支持版本为 {ARTIFACT_VERSION}"
                )
            if "object" not in payload:
                raise SerializationError("制品内容不完整，缺少 object 字段")
            obj = payload["object"]
        else:
            obj = payload

        if not isinstance(obj, cls):
            raise SerializationError(
                f"制品对象类型为 {type(obj).__name__}，不能作为 {cls.__name__} 加载"
            )
        return obj

