"""Agent Skills 文件与对象输入解析。"""

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from ..utils.io import load_pickle
from ..utils.serialization import ARTIFACT_FORMAT
from .errors import SkillExecutionError
from .objects import ObjectRegistry


_ARTIFACT_SUFFIXES = {".joblib", ".pkl", ".pickle", ".dill", ".cloudpickle"}


class InputResolver:
    """把受控输入描述解析为 DataFrame 或 Python 对象。"""

    def __init__(self, objects: Optional[ObjectRegistry] = None) -> None:
        self.objects = objects or ObjectRegistry()

    def resolve(self, spec: Mapping[str, Any]) -> Any:
        """解析文件或同进程对象引用。"""
        if not isinstance(spec, Mapping):
            raise SkillExecutionError(code="SCHEMA_INVALID", message="输入源必须是 JSON 对象")
        kind = spec.get("kind")
        if kind == "object_ref":
            ref = spec.get("ref")
            if not isinstance(ref, str) or not ref:
                raise SkillExecutionError(code="SCHEMA_INVALID", message="object_ref 缺少非空 ref", field="ref")
            return self.objects.resolve(ref)
        if kind != "file":
            raise SkillExecutionError(
                code="SCHEMA_INVALID",
                message=f"不支持的输入源类型“{kind}”",
                field="kind",
            )
        return self._resolve_file(spec)

    def _resolve_file(self, spec: Mapping[str, Any]) -> Any:
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise SkillExecutionError(code="SCHEMA_INVALID", message="文件输入缺少非空 path", field="path")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SkillExecutionError(code="INPUT_NOT_FOUND", message=f"输入文件不存在：{path}", field="path")

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(
                    path,
                    encoding=spec.get("encoding", "utf-8"),
                    sep=spec.get("separator", spec.get("sep", ",")),
                )
            if suffix == ".xlsx":
                return pd.read_excel(path, sheet_name=spec.get("sheet_name", 0))
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix in _ARTIFACT_SUFFIXES:
                return self._load_artifact(path, spec)
        except ImportError as exc:
            raise SkillExecutionError(
                code="DEPENDENCY_MISSING",
                message=f"读取“{path.name}”缺少可选依赖：{exc}",
                field="path",
                cause=exc,
            ) from exc
        except (KeyError, ValueError) as exc:
            if suffix == ".xlsx" and spec.get("sheet_name") is not None:
                raise SkillExecutionError(
                    code="INPUT_NOT_FOUND",
                    message=f"Excel 文件“{path.name}”中未找到工作表“{spec.get('sheet_name')}”",
                    field="sheet_name",
                    cause=exc,
                ) from exc
            raise SkillExecutionError(
                code="INPUT_FORMAT_UNSUPPORTED",
                message=f"无法读取输入文件“{path.name}”：{exc}",
                field="path",
                cause=exc,
            ) from exc

        raise SkillExecutionError(
            code="INPUT_FORMAT_UNSUPPORTED",
            message=f"不支持的输入文件格式“{suffix or path.name}”",
            field="path",
        )

    @staticmethod
    def _load_artifact(path: Path, spec: Mapping[str, Any]) -> Any:
        if spec.get("trusted") is not True:
            raise SkillExecutionError(
                code="ARTIFACT_UNTRUSTED",
                message="加载 pickle/joblib 制品可能执行代码，必须显式设置 trusted=true",
                field="trusted",
            )
        try:
            payload = load_pickle(path, engine=spec.get("engine", "auto"))
        except Exception as exc:
            raise SkillExecutionError(
                code="INPUT_FORMAT_UNSUPPORTED",
                message=f"无法加载 hscredit 制品“{path.name}”：{exc}",
                field="path",
                cause=exc,
            ) from exc
        if isinstance(payload, dict) and payload.get("format") == ARTIFACT_FORMAT:
            if "object" not in payload:
                raise SkillExecutionError(
                    code="INPUT_FORMAT_UNSUPPORTED",
                    message=f"hscredit 制品“{path.name}”缺少 object 字段",
                    field="path",
                )
            return payload["object"]
        return payload
