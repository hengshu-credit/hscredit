"""数据库适配器基础契约。"""

from typing import Any, Dict, Mapping, Optional, Sequence

from ...exceptions import StateError, ValidationError
from ..exceptions import DatabaseCapabilityError
from ..types import DatabaseCapabilities, PoolOptions


class BaseDatabaseAdapter:
    """数据库适配器公共基类。

    **参数**

    connect_kwargs : dict
        传递给底层数据库驱动的连接参数。
    pool_options : PoolOptions
        已校验的连接池配置。
    adapter_options : dict
        仅由适配器解释的方言或原生通道参数。
    """

    database_type = "base"
    identifier_quote = '"'
    capabilities = DatabaseCapabilities()

    def __init__(
        self,
        *,
        connect_kwargs: Mapping[str, Any],
        pool_options: PoolOptions,
        adapter_options: Optional[Mapping[str, Any]] = None,
    ):
        self.connect_kwargs: Dict[str, Any] = dict(connect_kwargs)
        self.pool_options = pool_options
        self.adapter_options: Dict[str, Any] = dict(adapter_options or {})
        self._closed = False

    @property
    def closed(self) -> bool:
        """适配器是否已关闭。"""

        return self._closed

    def ensure_open(self) -> None:
        """确保适配器仍可使用。"""

        if self._closed:
            raise StateError("数据库适配器已经关闭")

    def capabilities_for_table(
        self,
        table_name: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        """返回目标表可保证的能力。"""

        del table_name, table_metadata
        return self.capabilities

    def require_write_mode(
        self,
        table_name: str,
        mode: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        """校验目标表是否支持指定写入模式。"""

        capabilities = self.capabilities_for_table(table_name, table_metadata)
        if mode not in capabilities.write_modes:
            available = sorted(capabilities.write_modes)
            raise DatabaseCapabilityError(
                f"数据库 {self.database_type} 的目标表 {table_name!r} 不支持写入模式 {mode!r}，"
                f"当前可用模式: {available}"
            )
        return capabilities

    def build_count_sql(self, sql: str) -> str:
        """生成通用子查询计数 SQL。"""

        query = sql.rstrip().removesuffix(";").rstrip()
        return f"SELECT COUNT(1) FROM ({query}) hscredit_count"

    def quote_identifier(self, identifier: str) -> str:
        """按当前数据库方言引用单个标识符。"""

        if not isinstance(identifier, str) or not identifier:
            raise ValidationError("数据库标识符必须是非空字符串")
        quote = self.identifier_quote
        escaped = identifier.replace(quote, quote + quote)
        return f"{quote}{escaped}{quote}"

    def quote_qualified_name(self, name: str) -> str:
        """逐段引用数据库对象限定名。"""

        from ..writing import split_qualified_name

        return ".".join(self.quote_identifier(part) for part in split_qualified_name(name))

    def create_table(
        self,
        data: Any,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """创建目标表。"""

        raise NotImplementedError

    def resolve_key_columns(
        self,
        table_name: str,
        key_columns: Optional[Sequence[str]],
        first_batch: Any,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Sequence[str]]:
        """解析显式或数据库元数据中的主键字段。"""

        del table_name, first_batch, dialect_options
        return tuple(key_columns) if key_columns is not None else None

    def prepare_write(
        self,
        table_name: str,
        mode: str,
        first_batch: Any,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """在首个批次写入前校验并准备目标表。"""

        del first_batch, key_columns, dialect_options
        self.require_write_mode(table_name, mode)

    def write_batch(
        self,
        table_name: str,
        batch: Any,
        mode: str,
        batch_index: int,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """写入一个已经校验的 DataFrame 批次。"""

        raise NotImplementedError

    def finish_write(
        self,
        table_name: str,
        mode: str,
        result: Any,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """完成适配器专有的写入收尾。"""

        del table_name, mode, result, dialect_options

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        """执行查询。"""

        raise NotImplementedError

    def close(self) -> None:
        """关闭适配器持有的资源。"""

        self._closed = True


__all__ = ["BaseDatabaseAdapter"]
