"""Agent Skills 文件与对象输入解析。"""

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from ..utils.serialization import unpack_artifact, load_verified_pickle
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
            if spec.get('max_file_bytes') is not None:
                raise SkillExecutionError(code="SCHEMA_INVALID", message="对象引用不支持文件字节限制 max_file_bytes", field="max_file_bytes")
            return self._checked_table(self.objects.resolve(ref), spec)
        if kind != "file":
            raise SkillExecutionError(
                code="SCHEMA_INVALID",
                message=f"不支持的输入源类型“{kind}”",
                field="kind",
            )
        return self._resolve_file(spec)

    @staticmethod
    def _table_options(spec):
        limits = {}
        for option in ('max_rows', 'max_file_bytes', 'max_frame_bytes'):
            limit = spec.get(option)
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
                raise SkillExecutionError(code="SCHEMA_INVALID", message=f"{option} 必须为正整数", field=option)
            limits[option] = limit
        columns = spec.get('columns')
        if columns is not None and (not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns)):
            raise SkillExecutionError(code="SCHEMA_INVALID", message="columns 必须是不重复的非空字段名列表", field="columns")
        return columns, limits

    @classmethod
    def _checked_table(cls, value, spec):
        columns, limits = cls._table_options(spec)
        if not isinstance(value, pd.DataFrame):
            if columns is not None or limits['max_rows'] is not None or limits['max_frame_bytes'] is not None:
                raise SkillExecutionError(code="SCHEMA_INVALID", message="非DataFrame对象不支持字段、行数或表格内存限制", field="inputs")
            return value
        if limits['max_rows'] is not None and len(value) > limits['max_rows']:
            raise SkillExecutionError(code="INPUT_LIMIT_EXCEEDED", message="输入行数超过 max_rows 限制，不静默截断", field="max_rows")
        if columns is not None:
            if not value.columns.is_unique:
                raise SkillExecutionError(code="SCHEMA_INVALID", message="投影输入不能包含重复列名", field="columns")
            missing = [column for column in columns if column not in value.columns]
            if missing:
                raise SkillExecutionError(code="COLUMN_MISSING", message=f"输入缺少投影字段: {missing}", field="columns")
            value = value.loc[:, columns]
        if limits['max_frame_bytes'] is not None and int(value.memory_usage(deep=True).sum()) > limits['max_frame_bytes']:
            raise SkillExecutionError(code="INPUT_LIMIT_EXCEEDED", message="投影后数据超过 max_frame_bytes 限制", field="max_frame_bytes")
        return value

    def _resolve_file(self, spec: Mapping[str, Any]) -> Any:
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise SkillExecutionError(code="SCHEMA_INVALID", message="文件输入缺少非空 path", field="path")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SkillExecutionError(code="INPUT_NOT_FOUND", message=f"输入文件不存在：{path}", field="path")
        suffix = path.suffix.lower()
        columns, limits = self._table_options(spec)
        if limits['max_file_bytes'] is not None and path.stat().st_size > limits['max_file_bytes']:
            raise SkillExecutionError(code="INPUT_LIMIT_EXCEEDED", message="输入文件超过 max_file_bytes 限制", field="max_file_bytes")

        def checked(frame):
            return self._checked_table(frame, spec)

        nrows = limits['max_rows'] + 1 if limits['max_rows'] is not None else None
        try:
            if suffix == ".csv":
                return checked(pd.read_csv(
                    path,
                    encoding=spec.get("encoding", "utf-8"),
                    sep=spec.get("separator", spec.get("sep", ",")),
                    usecols=columns, nrows=nrows,
                ))
            if suffix == ".xlsx":
                return checked(pd.read_excel(path, sheet_name=spec.get("sheet_name", 0), usecols=columns, nrows=nrows))
            if suffix == ".parquet":
                if limits['max_rows'] is not None:
                    from pyarrow.parquet import ParquetFile
                    if ParquetFile(path).metadata.num_rows > limits['max_rows']:
                        raise SkillExecutionError(code="INPUT_LIMIT_EXCEEDED", message="Parquet元数据行数超过 max_rows 限制", field="max_rows")
                return checked(pd.read_parquet(path, columns=columns))
            if suffix in _ARTIFACT_SUFFIXES:
                loaded = self._load_artifact(path, spec)
                return checked(loaded)
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
            policy = spec.get('metadata_policy', 'warn')
            if policy not in {'warn', 'strict', 'ignore'}:
                raise ValueError("metadata_policy 必须为 warn、strict 或 ignore")
            payload = load_verified_pickle(path, expected_sha256=spec.get('sha256'), engine=spec.get("engine", "auto"))
            return unpack_artifact(payload, metadata_policy=policy)
        except Exception as exc:
            raise SkillExecutionError(
                code="INPUT_FORMAT_UNSUPPORTED",
                message=f"无法加载 hscredit 制品“{path.name}”：{exc}",
                field="path",
                cause=exc,
            ) from exc
