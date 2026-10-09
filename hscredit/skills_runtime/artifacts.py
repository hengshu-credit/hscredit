"""Agent Skills 事务制品和紧凑结果摘要。"""

import json
import shutil
import tempfile
import hashlib
import math
import warnings
import os
import socket
import logging
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .contracts import OutputSpec
from .errors import SkillExecutionError

logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
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
    """暂存全部制品，以不可变版本目录和原子完成清单发布。

    消费者应使用返回的path或manifest，不猜测根目录文件名。覆盖只更新完成清单，
    不改动上次版本。进程崩溃最多留下无清单引用的版本目录，不会发布半完成清单。
    """

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
        if not self.name or Path(self.name).name != self.name or any(c in self.name for c in '/\\') or self.name in {'.', '..'}:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="制品名称必须是非空文件名，不能包含目录")
        self.overwrite = bool(overwrite)
        self.staging_dir: Optional[Path] = None
        self.artifacts: List[Dict[str, Any]] = []
        self.run_dir = self.output_dir / f"{self.name}-{uuid4().hex}"
        self.manifest_path = self.output_dir / f"{self.name}.manifest.json"
        self._pending = []
        self._lock_path = self.output_dir / f".{self.name}.publish.lock"
        self._owns_lock = False
        self._lock_identity = None
        self.owner_token = uuid4().hex
        self.cleanup_errors = []
        self.committed = False

    def __enter__(self) -> "ArtifactTransaction":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            lock = self._lock_path.open('x', encoding='utf-8')
        except FileExistsError as exc:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="同名制品正在发布，或上次中断后需检查发布锁") from exc
        try:
            # open('x')成功即取得所有权，不能等write/flush/close完成后才记账。
            self._owns_lock = True
            with lock:
                identity = os.fstat(lock.fileno())
                self._lock_identity = (identity.st_dev, identity.st_ino)
                lock.write(json.dumps({'pid': os.getpid(), 'host': socket.gethostname(),
                    'created_at': datetime.now(timezone.utc).isoformat(), 'owner_token': self.owner_token,
                    'version_directory': str(self.run_dir)}, ensure_ascii=False))
                lock.flush()
        except BaseException:
            self._release_lock()
            raise
        if self.manifest_path.exists() and not self.overwrite:
            self._release_lock()
            raise SkillExecutionError(code="ARTIFACT_EXISTS", message="同名制品清单已存在且未允许覆盖", field="output.overwrite")
        try:
            staging = Path(tempfile.mkdtemp(prefix=".hscredit-skill-", dir=self.output_dir)).resolve()
        except BaseException:
            self._release_lock()
            raise
        if staging.parent != self.output_dir:
            self._release_lock()
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="临时制品目录不在目标输出目录内")
        self.staging_dir = staging
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                self._commit()
        finally:
            try:
                self._cleanup()
            except Exception as cleanup_error:
                self._cleanup_diagnostic('制品暂存清理失败', cleanup_error)
            finally:
                self._release_lock()

    def _cleanup_diagnostic(self, stage, error):
        """次要清理错误可观测，但-Werror不能改变业务异常或已提交状态。"""
        message = f"{stage}，不改变已提交清单或原始异常: {type(error).__name__}: {error}"
        self.cleanup_errors.append({'阶段': stage, '错误': str(error), '已提交': self.committed})
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        except RuntimeWarning:
            try:
                logger.warning(message)
            except Exception as log_error:
                self.cleanup_errors.append({'阶段': '清理诊断日志', '错误': str(log_error), '已提交': self.committed})

    def _release_lock(self):
        if not self._owns_lock:
            return
        try:
            if self._lock_path.exists() and self._lock_identity is not None:
                current = self._lock_path.stat()
                if (current.st_dev, current.st_ino) != self._lock_identity:
                    self._owns_lock = False
                    self._cleanup_diagnostic('发布锁已被替换，不删除其他任务的锁', OSError(str(self._lock_path)))
                    return
            self._lock_path.unlink(missing_ok=True)
            self._owns_lock = False
        except OSError as error:
            self._cleanup_diagnostic('释放发布锁失败', error)

    @classmethod
    def inspect_lock(cls, directory, name):
        """只读查看发布锁，不根据PID猜测存活、不自动删除锁或版本目录。"""
        transaction = cls({'directory': directory, 'name': name})
        path = transaction._lock_path
        result = {'存在': path.exists(), '路径': str(path)}
        if result['存在']:
            try:
                with path.open('r', encoding='utf-8') as handle:
                    content = handle.read(16385)
                if len(content) > 16384:
                    raise ValueError('锁文件超出诊断大小限制')
                result['所有者'] = json.loads(content)
            except (OSError, ValueError) as error:
                result['诊断'] = str(error)
        return result

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
        """登记制品；成功退出上下文时统一发布，退出前最终路径不可见。"""
        if self.staging_dir is None:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message="制品事务尚未开始")
        staged_path = Path(staged).resolve()
        if self.staging_dir not in staged_path.parents or not staged_path.is_file():
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"暂存制品不存在或越界：{staged_path}")
        name = final_name or staged_path.name
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"非法目标制品路径：{name}")
        destination = (self.run_dir / relative).resolve()
        if self.run_dir not in destination.parents:
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"目标制品路径越过输出目录：{name}")
        if (self.output_dir / relative).exists() and not self.overwrite:
            raise SkillExecutionError(
                code="ARTIFACT_EXISTS",
                message=f"目标制品已存在且未允许覆盖：{destination}",
                field="output.overwrite",
            )
        if any(existing[1] == relative for existing in self._pending):
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"制品名称重复: {name}")
        self._pending.append((staged_path, relative))
        item = {"type": artifact_type, "path": str(destination)}
        self.artifacts.append(item)
        return item

    def write_json(self, relative_name: str, value: Any) -> Path:
        """在暂存区写入 UTF-8 JSON。"""
        path = self.stage_path(relative_name)
        path.write_text(
            json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return path

    def _commit(self):
        """完成目录一次改名，最后原子更新清单；异常保留旧清单。"""
        if not self._pending:
            return
        payload = self.staging_dir / ('.bundle-' + uuid4().hex)
        payload.mkdir()
        entries = []
        try:
            for (source, relative), item in zip(self._pending, self.artifacts):
                if not source.is_file():
                    raise OSError(f"暂存制品丢失: {source.name}")
                target = payload / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                digest = hashlib.sha256()
                with target.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                entries.append({**item, 'sha256': digest.hexdigest(), 'bytes': target.stat().st_size})
            manifest = self.staging_dir / '.completed.json'
            manifest.write_text(json.dumps({'状态': '完成', '协议版本': 1, '版本目录': str(self.run_dir),
                                            '制品': entries}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            payload.replace(self.run_dir)
            manifest.replace(self.manifest_path)
            self.committed = True
        except Exception as exc:
            # 目录是本次随机生成的私有版本，失败时不可覆盖/删除旧版本。
            if self.run_dir.parent == self.output_dir and self.run_dir.exists():
                try:
                    shutil.rmtree(self.run_dir)
                except OSError as cleanup_error:
                    self._cleanup_diagnostic('清理未发布版本失败', cleanup_error)
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"制品整批发布失败: {exc}", cause=exc) from exc

    def _cleanup(self) -> None:
        if self.staging_dir is None:
            return
        staging = self.staging_dir.resolve()
        self.staging_dir = None
        if staging.parent != self.output_dir or not staging.name.startswith(".hscredit-skill-"):
            raise SkillExecutionError(code="ARTIFACT_WRITE_FAILED", message=f"拒绝清理未验证目录：{staging}")
        if staging.exists():
            shutil.rmtree(staging)
