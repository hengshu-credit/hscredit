"""可中断流式查询。

管理查询资源生命周期、分块 DataFrame、进度显示以及中断后的部分结果合并。
"""

from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from ..exceptions import StateError
from .exceptions import DatabaseQueryError, database_error_from
from .types import StreamState, validate_result_type


class QueryStream:
    """可中断并可合并已读数据的分块查询迭代器。

    QueryStream 由 :meth:`Database.stream_query <hscredit.database.client.Database.stream_query>`
    创建。迭代期间只持有当前数据库资源；``retain=True`` 时额外保留已经返回的标准化
    DataFrame 分块，用于停止或中断后合并结果。

    **参数**

    resource
        实现 ``fetchmany(size)``、``close()`` 和 ``columns`` 的适配器查询资源。
    chunksize : int
        每次请求的最大行数。
    retain : bool, default=True
        是否保留已读取分块。
    total_rows : int, optional
        进度条总行数。
    progress : bool, default=False
        是否显示 tqdm 进度条。
    result : {"dataframe", "records", "rows"}, default="dataframe"
        每次迭代和最终合并使用的结果类型。
    defaults : mapping, optional
        JSON 投影字段在数据库返回 ``NULL`` 时使用的默认值。
    sql : str, optional
        当前流实际执行的 SQL，用于失败异常上下文。
    params : Any, optional
        SQL 绑定参数，仅作为失败异常属性保存。

    **属性**

    state : StreamState
        当前运行、完成、中断、失败或关闭状态。
    rows_read : int
        已经读取并返回的累计行数。
    interrupt_reason : str, optional
        主动停止、中断或失败原因。

    **参考样例**

    >>> stream = db.stream_query("SELECT id FROM events", chunksize=1000)
    >>> first = next(stream)
    >>> stream.stop("仅抽取首批")
    >>> partial = stream.to_dataframe()
    """

    def __init__(
        self,
        resource: Any,
        *,
        chunksize: int,
        retain: bool = True,
        total_rows: Optional[int] = None,
        progress: bool = False,
        result: str = "dataframe",
        defaults: Optional[Mapping[str, Any]] = None,
        sql: Optional[str] = None,
        params: Any = None,
    ):
        validate_result_type(result)
        self.resource = resource
        self.chunksize = chunksize
        self.retain = retain
        self.total_rows = total_rows
        self.progress = progress
        self.result = result
        self.defaults = dict(defaults or {})
        self.sql = sql
        self.params = params
        self.state = StreamState.RUNNING
        self.rows_read = 0
        self.interrupted_at: Optional[str] = None
        self.interrupt_reason: Optional[str] = None
        self._chunks: List[pd.DataFrame] = []
        self._columns = list(getattr(resource, "columns", ()))
        self._resource_closed = False
        self._close_callbacks: List[Callable[[], Any]] = []
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

    @staticmethod
    def _is_sql_null(value: Any) -> bool:
        return value is None or value is pd.NA

    def _to_frame(self, batch: Any) -> tuple[pd.DataFrame, Mapping[str, pd.Series]]:
        if isinstance(batch, pd.DataFrame):
            null_masks = {
                column: batch[column].map(self._is_sql_null)
                for column, default in self.defaults.items()
                if default is not None and column in batch.columns
            }
            return batch, null_masks
        rows: Sequence[Any] = list(batch)
        frame = pd.DataFrame.from_records(rows, columns=self._columns)
        null_masks = {}
        for column, default in self.defaults.items():
            if default is None or column not in self._columns:
                continue
            position = self._columns.index(column)
            values = (row.get(column) if isinstance(row, Mapping) else row[position] for row in rows)
            null_masks[column] = pd.Series(
                [self._is_sql_null(value) for value in values],
                index=frame.index,
                dtype=bool,
            )
        return frame, null_masks

    def _apply_defaults(
        self,
        frame: pd.DataFrame,
        null_masks: Mapping[str, pd.Series],
    ) -> pd.DataFrame:
        for column, default in self.defaults.items():
            if default is None or column not in frame.columns:
                continue
            missing = null_masks.get(column)
            if missing is None:
                continue
            if not missing.any():
                continue
            frame[column] = frame[column].astype(object)
            column_position = frame.columns.get_loc(column)
            for row_position, is_missing in enumerate(missing.to_numpy(dtype=bool)):
                if is_missing:
                    frame.iat[row_position, column_position] = deepcopy(default)
        return frame

    def _format_frame(self, frame: pd.DataFrame) -> Any:
        if self.result == "dataframe":
            return frame
        if self.result == "records":
            return frame.to_dict("records")
        return list(frame.itertuples(index=False, name=None))

    def __next__(self) -> Any:
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
            cleanup_error = self._close_after_failure()
            raise database_error_from(
                DatabaseQueryError,
                "流式读取失败",
                cause=exc,
                sql=self.sql,
                params=self.params,
                cleanup_error=cleanup_error,
            )

        if self._is_empty_batch(batch):
            self.state = StreamState.COMPLETED
            self._close_resource()
            raise StopIteration

        try:
            frame, null_masks = self._to_frame(batch)
            frame = self._apply_defaults(frame, null_masks)
            self.rows_read += len(frame)
            if self.retain:
                self._chunks.append(frame)
            if self._progress_bar is not None:
                self._progress_bar.update(len(frame))
            return self._format_frame(frame)
        except KeyboardInterrupt:
            self._interrupt("KeyboardInterrupt")
            raise StopIteration
        except Exception as exc:
            self.state = StreamState.FAILED
            self.interrupt_reason = str(exc)
            cleanup_error = self._close_after_failure()
            raise database_error_from(
                DatabaseQueryError,
                "流式读取失败",
                cause=exc,
                sql=self.sql,
                params=self.params,
                cleanup_error=cleanup_error,
            )

    def _close_after_failure(self) -> Optional[BaseException]:
        try:
            self._close_resource()
        except Exception as exc:
            return exc
        return None

    def _close_resource(self) -> None:
        if self._resource_closed:
            return
        try:
            self.resource.close()
        finally:
            self._resource_closed = True
            if self._progress_bar is not None:
                self._progress_bar.close()
            callbacks = self._close_callbacks
            self._close_callbacks = []
            callback_error = None
            for callback in callbacks:
                try:
                    callback()
                except BaseException as exc:
                    if callback_error is None:
                        callback_error = exc
            if callback_error is not None:
                raise callback_error

    def _add_close_callback(self, callback: Callable[[], Any]) -> None:
        """注册在查询资源关闭后执行一次的内部清理回调。"""

        if not callable(callback):
            raise TypeError("callback 必须可调用")
        if self._resource_closed:
            callback()
            return
        self._close_callbacks.append(callback)

    def _interrupt(self, reason: str) -> None:
        if self.state is not StreamState.RUNNING:
            return
        self.state = StreamState.INTERRUPTED
        self.interrupted_at = datetime.now(timezone.utc).isoformat()
        self.interrupt_reason = reason
        self._close_resource()

    def stop(self, reason: str = "用户主动停止") -> None:
        """安全停止读取并关闭底层查询资源。

        :param reason: 写入流状态和最终 DataFrame 属性的停止原因。
        :return: ``None``。已经读取的数据可继续通过 :meth:`to_result` 获取。
        """

        self._interrupt(reason)

    def close(self) -> None:
        """关闭查询资源但不把状态标记为主动中断。

        尚在运行时状态会变为 ``closed``；重复调用不会重复关闭游标或连接。
        """

        if self.state is StreamState.RUNNING:
            self.state = StreamState.CLOSED
            self.interrupt_reason = "查询流已关闭"
        self._close_resource()

    def to_dataframe(self) -> pd.DataFrame:
        """把已保留分块合并为 DataFrame，并附加读取状态属性。

        :return: 连续索引的合并 DataFrame。``attrs`` 包含 ``completed``、``rows_read``、
            ``total_rows``、``state``、``interrupted_at`` 和 ``interrupt_reason``。
        :rtype: pandas.DataFrame
        :raises StateError: 创建流时设置了 ``retain=False``。
        """

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

    def to_records(self) -> List[Mapping[str, Any]]:
        """把已保留分块合并为记录字典列表。

        :return: 每行一个字段名到原始值映射的列表。
        :raises StateError: 创建流时设置了 ``retain=False``。
        """

        return self.to_dataframe().to_dict("records")

    def to_rows(self) -> List[Any]:
        """把已保留分块合并为原始行元组列表。

        :return: 字段顺序与查询结果一致的行元组列表。
        :raises StateError: 创建流时设置了 ``retain=False``。
        """

        frame = self.to_dataframe()
        return list(frame.itertuples(index=False, name=None))

    def to_result(self) -> Any:
        """按创建流时的 ``result`` 配置返回合并结果。

        :return: DataFrame、``list[dict]`` 或行元组列表。
        :raises StateError: 创建流时设置了 ``retain=False``。
        """

        if self.result == "dataframe":
            return self.to_dataframe()
        if self.result == "records":
            return self.to_records()
        return self.to_rows()

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
