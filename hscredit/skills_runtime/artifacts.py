"""Agent Skills 事务制品和紧凑结果摘要。"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .contracts import OutputSpec
from .errors import SkillExecutionError


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return str(value)


def _column_label(column: Any) -> Union[str, List[str]]:
    if isinstance(column, tuple):
        return [str(part) for part in column]
    return str(column)


def _preview_key(column: Any) -> str:
    if isinstance(column, tuple):
        return " / ".join(str(part) for part in column)
    return str(column)


def summarize_dataframe(frame: pd.DataFrame, preview_rows: int = 10) -> Dict[str, Any]:
    """返回适合 Agent 上下文的行列信息和受限预览。"""
    limit = max(0, int(preview_rows))
    preview = []
    for _, row in frame.head(limit).iterrows():
        preview.append({_preview_key(column): _json_value(row[column]) for column in frame.columns})
    return {
        "rows": int(len(frame)),
        "columns": [_column_label(column) for column in frame.columns],
        "preview": preview,
    }


class ArtifactTransaction:
    """在目标目录内暂存并原子发布本次操作的制品。"""

    def __init__(self, output: Union[OutputSpec, Mapping[str, Any]]) -> None:
        if isinstance(output, OutputSpec):
            directory = output.directory
            name = output.name
            overwrite = output.overwrite
        else:
            directory = output.get("directory", ".")
            name = output.get("name", "artifact")
            overwrite = output.get("overwrite", False)
        self.output_dir = Path(directory).expanduser().resolve()
        self.name = str(name)
        self.overwrite = bool(overwrite)
        self.staging_dir: Optional[Path] = None
        self.artifacts: List[Dict[str, Any]] = []

    def __enter__(self) -> "ArtifactTransaction":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".hscredit-skill-", dir=self.output_dir)).resolve()
        if staging.parent != self.output_dir:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="临时制品目录不在目标输出目录内")
        self.staging_dir = staging
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._cleanup()

    def stage_path(self, relative_name: str) -> Path:
        """返回严格位于本次临时目录内的暂存路径。"""
        if self.staging_dir is None:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="制品事务尚未开始")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"非法制品相对路径：{relative_name}")
        staged = (self.staging_dir / relative).resolve()
        if self.staging_dir not in staged.parents:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"制品路径越过临时目录：{relative_name}")
        staged.parent.mkdir(parents=True, exist_ok=True)
        return staged

    def publish(
        self,
        staged: Union[str, Path],
        final_name: Optional[str] = None,
        *,
        artifact_type: str = "file",
    ) -> Dict[str, Any]:
        """发布一个完整制品并返回 manifest 项。"""
        if self.staging_dir is None:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="制品事务尚未开始")
        staged_path = Path(staged).resolve()
        if self.staging_dir not in staged_path.parents or not staged_path.is_file():
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"暂存制品不存在或越界：{staged_path}")
        name = final_name or staged_path.name
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"非法目标制品路径：{name}")
        destination = (self.output_dir / relative).resolve()
        if self.output_dir not in destination.parents:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"目标制品路径越过输出目录：{name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not self.overwrite:
            raise SkillExecutionError(
                code="ARTIFACT_EXISTS",
                message=f"目标制品已存在且未允许覆盖：{destination}",
                field="output.overwrite",
            )
        try:
            staged_path.replace(destination)
        except OSError as exc:
            raise SkillExecutionError(
                code="ARTIFACT_WRITE_FAILED",
                message=f"无法发布制品“{destination.name}”：{exc}",
                cause=exc,
            ) from exc
        item = {"type": artifact_type, "path": str(destination)}
        self.artifacts.append(item)
        return item

    def write_json(self, relative_name: str, value: Any) -> Path:
        """在暂存区写入 UTF-8 JSON。"""
        path = self.stage_path(relative_name)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_value),
            encoding="utf-8",
        )
        return path

    def _cleanup(self) -> None:
        if self.staging_dir is None:
            return
        staging = self.staging_dir.resolve()
        self.staging_dir = None
        if staging.parent != self.output_dir or not staging.name.startswith(".hscredit-skill-"):
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"拒绝清理未验证目录：{staging}")
        if staging.exists():
            shutil.rmtree(staging)
