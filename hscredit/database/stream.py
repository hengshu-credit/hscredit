"""可中断流式查询。

管理查询资源生命周期、分块 DataFrame、进度显示以及中断后的部分结果合并。
"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from ..exceptions import StateError
from .exceptions import DatabaseQueryError
from .types import StreamState


class QueryStream:
    """有状态的 DataFrame 分块查询迭代器。"""

    def __init__(
        self,
        resource: Any,
        *,
        chunksize: int,
        retain: bool = True,
        total_rows: Optional[int] = None,
        progress: bool = False,
    ):
        self.resource = resource
        self.chunksize = chunksize
        self.retain = retain
        self.total_rows = total_rows
        self.progress = progress
        self.state = StreamState.RUNNING
        self.rows_read = 0
        self.interrupted_at: Optional[str] = None
        self.interrupt_reason: Optional[str] = None
        self._chunks: List[pd.DataFrame] = []
        self._columns = list(getattr(resource, "columns", ()))
        self._resource_closed = False
        self._progress_bar = tqdm(total=total_rows, desc="流式读取", unit="行") if progress else None

    def __iter__(self) -> "QueryStream":
        return self

    @staticmethod
    def _is_empty_batch(batch: Any) -> bool:
        if batch is None:
            return True
        try:
            return len(batch) == 0
        except TypeError:
            return False

    def _to_frame(self, batch: Any) -> pd.DataFrame:
        if isinstance(batch, pd.DataFrame):
            return batch
        rows: Sequence[Any] = list(batch)
        return pd.DataFrame.from_records(rows, columns=self._columns)

    def __next__(self) -> pd.DataFrame:
        if self.state is not StreamState.RUNNING:
            raise StopIteration
        try:
            batch = self.resource.fetchmany(self.chunksize)
        except KeyboardInterrupt:
            self._interrupt("KeyboardInterrupt")
            raise StopIteration
        except Exception as exc:
            self.state = StreamState.FAILED
            self.interrupt_reason = str(exc)
            self._close_resource()
            raise DatabaseQueryError("流式读取失败") from exc

        if self._is_empty_batch(batch):
            self.state = StreamState.COMPLETED
            self._close_resource()
            raise StopIteration

        frame = self._to_frame(batch)
        self.rows_read += len(frame)
        if self.retain:
            self._chunks.append(frame)
        if self._progress_bar is not None:
            self._progress_bar.update(len(frame))
        return frame

    def _close_resource(self) -> None:
        if self._resource_closed:
            return
        try:
            self.resource.close()
        finally:
            self._resource_closed = True
            if self._progress_bar is not None:
                self._progress_bar.close()

    def _interrupt(self, reason: str) -> None:
        if self.state is not StreamState.RUNNING:
            return
        self.state = StreamState.INTERRUPTED
        self.interrupted_at = datetime.now(timezone.utc).isoformat()
        self.interrupt_reason = reason
        self._close_resource()

    def stop(self, reason: str = "用户主动停止") -> None:
        """安全停止读取，并保留当前已经读取的数据。"""

        self._interrupt(reason)

    def close(self) -> None:
        """关闭查询资源。"""

        if self.state is StreamState.RUNNING:
            self.state = StreamState.CLOSED
            self.interrupt_reason = "查询流已关闭"
        self._close_resource()

    def to_dataframe(self) -> pd.DataFrame:
        """合并已经保留的分块并附加完成状态。"""

        if not self.retain:
            raise StateError("retain=False时无法合并已消费的数据")
        if self._chunks:
            frame = pd.concat(self._chunks, axis=0, ignore_index=True, copy=False)
        else:
            frame = pd.DataFrame(columns=self._columns)

        frame.attrs.update(
            {
                "completed": self.state is StreamState.COMPLETED,
                "rows_read": self.rows_read,
                "total_rows": self.total_rows,
                "state": self.state.value,
                "interrupted_at": self.interrupted_at,
                "interrupt_reason": self.interrupt_reason,
            }
        )
        return frame

    def __enter__(self) -> "QueryStream":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_value, traceback
        if exc_type is KeyboardInterrupt:
            self._interrupt("KeyboardInterrupt")
        else:
            self.close()
        return False


__all__ = ["QueryStream"]
