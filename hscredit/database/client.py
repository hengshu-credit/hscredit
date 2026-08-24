"""数据库公共门面。"""

from pathlib import Path
from itertools import chain
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..exceptions import InputValidationError, StateError, ValidationError
from .exceptions import (
    DatabaseCapabilityError,
    DatabaseConnectionError,
    DatabaseMetadataError,
    DatabaseWriteError,
)
from .metadata import MetadataInspection, metadata_frame, parse_targets
from .json_projection import normalize_json_projection
from .registry import canonical_adapter_name, get_adapter_class
from .stream import QueryStream
from .types import PoolOptions, WRITE_MODES, WriteResult, validate_result_type
from .writing import BatchWriteResult, iter_write_batches, split_qualified_name


class Database:
    """统一数据库连接与操作门面。

    **参数**

    database_type : str
        已注册数据库类型或别名。
    pool_options : mapping or PoolOptions, optional
        连接池配置。
    adapter_options : mapping, optional
        后端专有配置。
    ``**connect_kwargs``
        直接传递给后端驱动的连接参数。

    **属性**

    adapter
        当前数据库适配器实例。

    **参考样例**

    >>> with Database("mysql", host="127.0.0.1", database="risk") as db:
    ...     rows = db.query("SELECT 1", result="rows")
    """

    def __init__(
        self,
        database_type: str,
        *,
        pool_options: Optional[Mapping[str, Any]] = None,
        adapter_options: Optional[Mapping[str, Any]] = None,
        **connect_kwargs: Any,
    ):
        self.database_type = canonical_adapter_name(database_type)
        self.adapter_options = dict(adapter_options or {})
        self._closed = False

        adapter_class = get_adapter_class(self.database_type)
        pool_options_class = getattr(adapter_class, "pool_options_class", PoolOptions)
        self.pool_options = pool_options_class.from_mapping(pool_options)
        try:
            self.adapter = adapter_class(
                connect_kwargs=dict(connect_kwargs),
                pool_options=self.pool_options,
                adapter_options=self.adapter_options,
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                f"初始化 {self.database_type} 数据库适配器失败，请检查连接参数和可选依赖"
            ) from exc

    @property
    def closed(self) -> bool:
        """数据库门面是否已经关闭。

        :return: 已调用 :meth:`close` 时为 ``True``，否则为 ``False``。
        :rtype: bool
        """

        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateError("数据库连接已经关闭")

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        """一次性执行查询并返回全部结果。

        大结果集建议改用 :meth:`stream_query` 或 :meth:`read_query`，避免一次性占用过多内存。

        :param sql: 由当前数据库执行的查询 SQL。
        :param params: 按底层驱动参数风格绑定的 SQL 参数，默认为 ``None``。
        :param result: 结果类型，支持 ``dataframe``、``records`` 和 ``rows``。
        :return: ``dataframe`` 返回 DataFrame；``records`` 返回记录字典列表；``rows`` 返回原始行列表。
        :raises StateError: 数据库门面已经关闭。
        :raises ValidationError: ``result`` 不是支持的结果类型。
        :raises DatabaseQueryError: 数据库执行查询失败。

        **参考样例**

        >>> frame = db.query("SELECT id FROM events WHERE id > %s", params=(10,))
        >>> records = db.query("SELECT id FROM events", result="records")
        """

        self._ensure_open()
        return self.adapter.query(sql, params=params, result=result)

    def execute(self, sql: str, params: Any = None) -> Any:
        """执行单条 DDL 或 DML SQL。

        :param sql: 要执行的 SQL。
        :param params: 按底层驱动参数风格绑定的 SQL 参数，默认为 ``None``。
        :return: 适配器报告的影响行数或原生执行结果。
        :raises StateError: 数据库门面已经关闭。
        :raises DatabaseQueryError: SQL 执行失败。
        """

        self._ensure_open()
        return self.adapter.execute(sql, params=params)

    def executemany(self, sql: str, values: Any) -> Any:
        """使用多组绑定值批量执行同一条 SQL。

        :param sql: 带驱动占位符的 SQL。
        :param values: 每行一组参数的可迭代对象。
        :return: 适配器报告的累计影响行数或原生执行结果。
        :raises StateError: 数据库门面已经关闭。
        :raises DatabaseQueryError: 批量执行失败。
        """

        self._ensure_open()
        return self.adapter.executemany(sql, values)

    @property
    def native_client(self) -> Any:
        """返回 Redis 或 MongoDB 适配器持有的原生客户端。"""

        self._ensure_open()
        client = getattr(self.adapter, "client", None)
        if client is None:
            raise DatabaseCapabilityError(f"数据库 {self.database_type} 没有原生客户端接口")
        return client

    def _call_nosql(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        method = getattr(self.adapter, method_name, None)
        if not callable(method):
            raise DatabaseCapabilityError(f"数据库 {self.database_type} 不支持 NoSQL 方法 {method_name}")
        return method(*args, **kwargs)

    def read_one(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """读取单个 Redis key 或 MongoDB 文档。"""

        return self._call_nosql("read_one", resource, selector, **options)

    def read_many(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """批量读取 Redis keys 或 MongoDB 文档。"""

        return self._call_nosql("read_many", resource, selector, **options)

    def read(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """根据输入形态自适应执行单条或批量读取。"""

        return self._call_nosql("read", resource, selector, **options)

    def write_one(self, resource: Any, data: Any, **options: Any) -> Any:
        """写入单个 Redis key 或 MongoDB 文档。"""

        return self._call_nosql("write_one", resource, data, **options)

    def write_many(self, resource: Any, data: Any = None, **options: Any) -> Any:
        """批量写入 Redis key-value 或 MongoDB 文档。"""

        return self._call_nosql("write_many", resource, data, **options)

    def write(self, resource: Any, data: Any = None, **options: Any) -> Any:
        """根据输入形态自适应执行单条或批量写入。"""

        return self._call_nosql("write", resource, data, **options)

    def delete_one(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """删除单个 Redis key 或首个匹配 MongoDB 文档。"""

        return self._call_nosql("delete_one", resource, selector, **options)

    def delete_many(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """批量删除 Redis keys 或 MongoDB 文档。"""

        return self._call_nosql("delete_many", resource, selector, **options)

    def delete(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        """根据输入形态与 ``many`` 选项自适应执行删除。"""

        return self._call_nosql("delete", resource, selector, **options)

    def exists(self, resource: Any, selector: Any = None, **options: Any) -> bool:
        """判断 Redis key 或 MongoDB 匹配文档是否存在。"""

        return bool(self._call_nosql("exists", resource, selector, **options))

    @staticmethod
    def _validate_stream_options(
        chunksize: int,
        progress: bool,
        retain: bool,
        total_rows: Optional[int],
        result: str,
        count_total: bool,
        count_sql: Optional[str],
    ) -> None:
        if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
            raise ValidationError("chunksize 必须是正整数")
        if not isinstance(progress, bool):
            raise ValidationError("progress 必须是布尔值")
        if not isinstance(retain, bool):
            raise ValidationError("retain 必须是布尔值")
        if total_rows is not None:
            if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 0:
                raise ValidationError("total_rows 必须是非负整数或 None")
        if not isinstance(count_total, bool):
            raise ValidationError("count_total 必须是布尔值")
        if count_sql is not None and (not isinstance(count_sql, str) or not count_sql.strip()):
            raise ValidationError("count_sql 必须是非空字符串或 None")
        if not progress and (count_total or count_sql is not None):
            raise ValidationError("count_total 或 count_sql 仅能在 progress=True 时使用")
        if total_rows is not None and (count_total or count_sql is not None):
            raise ValidationError("total_rows 不能与 count_total 或 count_sql 同时使用")
        validate_result_type(result)

    def stream_query(
        self,
        sql: str,
        params: Any = None,
        *,
        chunksize: int = 50_000,
        progress: bool = False,
        retain: bool = True,
        count_total: bool = False,
        count_sql: Optional[str] = None,
        total_rows: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        json_fields: Optional[Mapping[str, Mapping[str, Any]]] = None,
        result: str = "dataframe",
    ) -> QueryStream:
        """打开可中断的分块查询流。

        当 ``json_fields`` 不为空时，适配器把 JSON 路径提取下推到数据库，只传输
        ``columns`` 和指定的 JSON 子字段，不返回原始大 JSON。JSON 字段定义格式为
        ``{源字段: {输出字段: 路径或(路径, 默认值)}}``。

        :param sql: 原始查询 SQL。JSON 投影会把该查询包装为子查询。
        :param params: 原始 SQL 的绑定参数，默认为 ``None``。
        :param chunksize: 每次向 DB-API 流式游标请求的最大行数，默认 50000。
        :param progress: 是否显示读取进度，默认 ``False``。未知总数时显示累计行数、速度和耗时。
        :param retain: 是否保留已经产生的分块，默认 ``True``。设为 ``False`` 后不能合并历史数据。
        :param count_total: 是否为进度条自动执行 ``COUNT(1)``，默认 ``False``。
        :param count_sql: 为进度条显式指定的统计 SQL；提供后会执行该 SQL。
        :param total_rows: 已知总行数；提供后不执行统计 SQL。
        :param columns: JSON 投影时原样保留的普通输出字段；不能包含 JSON 源字段。
        :param json_fields: JSON 源字段、输出字段、JSONPath 和可选默认值的嵌套映射。
        :param result: 每个分块的结果类型，支持 ``dataframe``、``records`` 和 ``rows``。
        :return: 可迭代、可主动停止并可合并已读数据的查询流。
        :rtype: QueryStream
        :raises ValidationError: 分块、进度、结果类型或 JSON 投影参数无效。
        :raises DatabaseCapabilityError: 当前适配器不支持 JSON 字段投影。
        :raises DatabaseQueryError: 统计查询或打开流式查询失败。

        **参考样例**

        >>> stream = db.stream_query(
        ...     "SELECT id, huge_json FROM user_profile",
        ...     columns=["id"],
        ...     json_fields={
        ...         "huge_json": {
        ...             "city": ("$.address.city", "未知"),
        ...             "customer_id": "$.customer.id",
        ...         }
        ...     },
        ...     result="records",
        ... )
        >>> for records in stream:
        ...     consume(records)
        """

        self._ensure_open()
        self._validate_stream_options(
            chunksize,
            progress,
            retain,
            total_rows,
            result,
            count_total,
            count_sql,
        )
        projection = normalize_json_projection(columns, json_fields)

        resolved_total = total_rows
        if progress and resolved_total is None and (count_total or count_sql is not None):
            resolved_count_sql = count_sql or self.adapter.build_count_sql(sql)
            resolved_total = self.adapter.count_rows(resolved_count_sql, params=params)

        projected_sql = (
            self.adapter.build_json_projection_sql(
                sql,
                columns=columns,
                json_fields=json_fields,
            )
            if projection is not None
            else sql
        )
        resource = self.adapter.open_stream(projected_sql, params=params)
        return QueryStream(
            resource,
            chunksize=chunksize,
            retain=retain,
            total_rows=resolved_total,
            progress=progress,
            result=result,
            defaults=projection.defaults if projection is not None else None,
        )

    def read_query(
        self,
        sql: str,
        params: Any = None,
        *,
        chunksize: int = 50_000,
        progress: bool = False,
        count_total: bool = False,
        count_sql: Optional[str] = None,
        total_rows: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        json_fields: Optional[Mapping[str, Mapping[str, Any]]] = None,
        result: str = "dataframe",
    ) -> Any:
        """消费完整查询流，并在中断后直接返回已经读取的数据。

        参数语义与 :meth:`stream_query` 一致，但本方法自动消费所有分块。发生
        ``KeyboardInterrupt`` 时会关闭底层资源，并按照 ``result`` 返回当前已合并数据。

        :param sql: 原始查询 SQL。
        :param params: 原始 SQL 的绑定参数，默认为 ``None``。
        :param chunksize: 每次请求的最大行数，默认 50000。
        :param progress: 是否显示进度，默认 ``False``。
        :param count_total: 是否自动执行 ``COUNT(1)`` 获取进度条总数，默认 ``False``。
        :param count_sql: 为进度条显式指定的统计 SQL。
        :param total_rows: 已知总行数。
        :param columns: JSON 投影时原样保留的普通输出字段；不能包含 JSON 源字段。
        :param json_fields: JSON 子字段投影映射。
        :param result: 返回类型，支持 ``dataframe``、``records`` 和 ``rows``。
        :return: 完整数据或中断前已读取的部分数据。
        :raises ValidationError: 查询或投影参数无效。
        :raises DatabaseQueryError: 流式查询失败。

        **参考样例**

        >>> frame = db.read_query("SELECT * FROM events", progress=True)
        >>> rows = db.read_query("SELECT id FROM events", result="rows")
        """

        stream = self.stream_query(
            sql,
            params=params,
            chunksize=chunksize,
            progress=progress,
            count_total=count_total,
            retain=True,
            count_sql=count_sql,
            total_rows=total_rows,
            columns=columns,
            json_fields=json_fields,
            result=result,
        )
        try:
            for _ in stream:
                pass
        except KeyboardInterrupt:
            stream.stop("KeyboardInterrupt")
        finally:
            if stream.state.value == "running":
                stream.close()
        return stream.to_result()

    def export_schema(
        self,
        targets: Optional[Sequence[str]] = None,
        *,
        output: Optional[Any] = None,
        excel_params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """读取数据库表结构并生成中文字段元数据宽表。

        :param targets: 可选数据库或表目标，例如 ``risk``、``risk.events``；默认扫描适配器可见范围。
        :param output: 可选 ``.xlsx`` 输出路径。提供时通过 ``dataframe2excel`` 导出。
        :param excel_params: 传递给 ``dataframe2excel`` 的附加参数。
        :return: 中文列名的表和字段信息 DataFrame；数据库原始元数据值保持不变。
        :rtype: pandas.DataFrame
        :raises ValidationError: 目标、输出扩展名或 Excel 参数无效。
        :raises DatabaseMetadataError: 元数据读取、目标匹配或 Excel 导出失败。
        """

        self._ensure_open()
        parsed_targets = parse_targets(targets)

        output_path = None
        if output is not None:
            output_path = Path(output)
            if output_path.suffix.lower() != ".xlsx":
                raise ValidationError("数据库表结构导出仅支持 .xlsx 文件")
        if excel_params is not None and not isinstance(excel_params, Mapping):
            raise ValidationError("excel_params 必须是映射或 None")

        try:
            inspection = self.adapter.inspect_schema(parsed_targets)
        except DatabaseMetadataError:
            raise
        except Exception as exc:
            raise DatabaseMetadataError("读取数据库表结构失败") from exc
        if not isinstance(inspection, MetadataInspection):
            raise DatabaseMetadataError("数据库适配器 inspect_schema() 必须返回 MetadataInspection")

        inspection = MetadataInspection(
            rows=list(inspection.rows),
            errors=list(inspection.errors),
        )
        if parsed_targets:
            missing_exact = []
            for target in parsed_targets:
                if len(target.parts) < 2:
                    continue
                expected = tuple(part.casefold() for part in target.parts)
                matched = False
                for row in inspection.rows:
                    candidate = tuple(
                        str(value).casefold()
                        for value in (
                            row.get("catalog"),
                            row.get("database"),
                            row.get("schema"),
                            row.get("table_name"),
                        )
                        if value not in (None, "")
                    )
                    if len(candidate) >= len(expected) and candidate[-len(expected) :] == expected:
                        matched = True
                        break
                if not matched:
                    missing_exact.append(target.raw)
            if missing_exact:
                raise DatabaseMetadataError(f"未找到精确指定的数据库表或无访问权限: {missing_exact}")

        frame = metadata_frame(inspection)
        if output_path is not None:
            from ..excel import dataframe2excel

            params = {
                "sheet_name": "表结构",
                "title": "数据库表结构",
                "index": False,
                "decimal": None,
                "auto_filter": True,
                "auto_width": True,
            }
            params.update(dict(excel_params or {}))
            try:
                dataframe2excel(frame, output_path, **params)
            except Exception as exc:
                raise DatabaseMetadataError(f"数据库表结构 Excel 导出失败: {output_path.name}") from exc
        return frame

    @staticmethod
    def _validate_dialect_options(
        dialect_options: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if dialect_options is None:
            return {}
        if not isinstance(dialect_options, Mapping):
            raise ValidationError("dialect_options 必须是映射或 None")
        return dict(dialect_options)

    @staticmethod
    def _validate_key_columns(
        key_columns: Optional[Sequence[str]],
        data_columns: Sequence[Any],
    ) -> Optional[Sequence[str]]:
        if key_columns is None:
            return None
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        normalized = tuple(key_columns)
        if not normalized or any(not isinstance(column, str) or not column for column in normalized):
            raise ValidationError("key_columns 必须是非空字段名序列")
        missing = [column for column in normalized if column not in data_columns]
        if missing:
            raise InputValidationError(f"写入数据缺少主键字段: {missing}")
        return normalized

    def create_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """根据 DataFrame 字段和后端方言参数创建表。

        :param data: 用于推断字段结构的非空列 DataFrame。
        :param table_name: 表名或 ``数据库名.表名`` 等限定名。
        :param dialect_options: 后端专有建表参数，例如键、引擎、分区、字段类型和注释。
        :return: 适配器返回的已执行 DDL 或原生结果。
        :raises InputValidationError: ``data`` 不是带字段的 DataFrame。
        :raises ValidationError: 表名或方言参数无效。
        :raises DatabaseQueryError: 建表 SQL 执行失败。
        """

        self._ensure_open()
        split_qualified_name(table_name)
        if not isinstance(data, pd.DataFrame) or len(data.columns) == 0:
            raise InputValidationError("create_table 的 data 必须是包含字段的 DataFrame")
        options = self._validate_dialect_options(dialect_options)
        return self.adapter.create_table(
            data,
            table_name,
            dialect_options=options,
        )

    @staticmethod
    def _add_optional_count(current: Optional[int], value: Optional[int]) -> Optional[int]:
        if current is None or value is None:
            return None
        return current + value

    def stream_write(
        self,
        data: Any,
        table_name: str,
        *,
        mode: str = "a",
        batch_size: int = 10_000,
        key_columns: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> WriteResult:
        """把 DataFrame、DataFrame 分块或行记录迭代器流式写入目标表。

        :param data: DataFrame、DataFrame 分块、记录字典或位置行的可迭代对象。
        :param table_name: 目标表限定名。
        :param mode: 写入模式。``a`` 追加且主键重复不覆盖；``r`` 追加且主键重复覆盖；
            ``o`` 保留表结构并清空重写；``d`` 删除并按首批数据重建表后写入。
        :param batch_size: 每个写入批次的最大行数，默认 10000。
        :param key_columns: ``a`` 或 ``r`` 使用的显式键字段；未提供时由支持的适配器读取表元数据。
        :param columns: 位置行字段名，或用于校验 DataFrame/记录字段顺序。
        :param dialect_options: 传递给目标适配器的建表和写入选项。
        :return: 完成状态、接收/插入/更新/跳过行数及已提交批次数。
        :rtype: WriteResult
        :raises InputValidationError: 数据为空、批次字段不一致或缺少键字段。
        :raises ValidationError: 模式、批次大小、表名或方言参数无效。
        :raises DatabaseCapabilityError: 目标数据库或表不支持指定模式。
        :raises DatabaseWriteError: 准备、批次写入或收尾失败；异常的 ``result`` 保留部分统计。
        """

        self._ensure_open()
        if mode not in WRITE_MODES:
            raise ValidationError(f"mode 只支持 {sorted(WRITE_MODES)}，收到 {mode!r}")
        split_qualified_name(table_name)
        options = self._validate_dialect_options(dialect_options)
        batches = iter_write_batches(
            data,
            batch_size=batch_size,
            columns=columns,
        )
        try:
            first_batch = next(batches)
        except StopIteration as exc:
            raise InputValidationError("写入数据没有可用的有效数据行") from exc
        if len(first_batch) == 0 or len(first_batch.columns) == 0:
            raise InputValidationError("写入数据没有可用的有效数据行")

        resolved_keys = self._validate_key_columns(
            key_columns,
            first_batch.columns,
        )
        if mode in {"a", "r"}:
            resolved_keys = self.adapter.resolve_key_columns(
                table_name,
                resolved_keys,
                first_batch,
                dialect_options=options,
            )
            resolved_keys = self._validate_key_columns(
                resolved_keys,
                first_batch.columns,
            )
        result = WriteResult(
            mode=mode,
            completed=False,
            rows_inserted=0,
            rows_updated=0,
            rows_skipped=0,
        )
        try:
            self.adapter.prepare_write(
                table_name,
                mode,
                first_batch,
                key_columns=resolved_keys,
                dialect_options=options,
            )
        except (DatabaseCapabilityError, ValidationError, InputValidationError):
            raise
        except Exception as exc:
            result.failed_batch = 0
            raise DatabaseWriteError(
                f"写入目标表 {table_name!r} 的准备操作失败",
                result=result,
            ) from exc

        for batch_index, batch in enumerate(chain((first_batch,), batches), start=1):
            result.rows_received += len(batch)
            try:
                batch_result = self.adapter.write_batch(
                    table_name,
                    batch,
                    mode,
                    batch_index,
                    key_columns=resolved_keys,
                    dialect_options=options,
                )
            except Exception as exc:
                result.failed_batch = batch_index
                raise DatabaseWriteError(
                    f"写入目标表 {table_name!r} 的第 {batch_index} 批数据失败",
                    result=result,
                ) from exc
            if batch_result is None:
                batch_result = BatchWriteResult()
            if not isinstance(batch_result, BatchWriteResult):
                raise DatabaseWriteError(
                    "数据库适配器 write_batch() 必须返回 BatchWriteResult",
                    result=result,
                )
            result.rows_inserted = self._add_optional_count(
                result.rows_inserted,
                batch_result.inserted,
            )
            result.rows_updated = self._add_optional_count(
                result.rows_updated,
                batch_result.updated,
            )
            result.rows_skipped = self._add_optional_count(
                result.rows_skipped,
                batch_result.skipped,
            )
            result.batches_committed += 1

        try:
            self.adapter.finish_write(
                table_name,
                mode,
                result,
                dialect_options=options,
            )
        except Exception as exc:
            result.failed_batch = result.batches_committed + 1
            raise DatabaseWriteError(
                f"完成目标表 {table_name!r} 的写入收尾失败",
                result=result,
            ) from exc
        result.completed = True
        return result

    def close(self) -> None:
        """关闭连接池或原生客户端；重复调用不产生副作用。

        关闭后所有查询和写入方法都会抛出 :class:`~hscredit.exceptions.StateError`。
        """

        if self._closed:
            return
        try:
            self.adapter.close()
        finally:
            self._closed = True

    def __enter__(self) -> "Database":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Database(database_type={self.database_type!r}, closed={self.closed})"


__all__ = ["Database"]
