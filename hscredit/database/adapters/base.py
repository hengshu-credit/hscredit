"""数据库适配器基础契约。"""

from typing import Any, Dict, Mapping, Optional, Sequence

from ...exceptions import StateError, ValidationError
from ..exceptions import DatabaseCapabilityError
from ..json_projection import normalize_json_projection
from ..types import DatabaseCapabilities, PoolOptions


class BaseDatabaseAdapter:
    """数据库适配器公共基类。

    **参数**

    :param connect_kwargs: 传递给底层数据库驱动的连接参数。
    :type connect_kwargs: dict
    :param pool_options: 已校验的连接池配置。
    :type pool_options: PoolOptions
    :param adapter_options: 仅由适配器解释的方言或原生通道参数。
    :type adapter_options: dict

    **属性**

    database_type : str
        注册表使用的数据库类型名称。
    capabilities : DatabaseCapabilities
        适配器默认保证的事务、读取、写入与元数据能力。

    **参考样例**

    自定义后端至少实现查询、流资源和所需写入方法；支持 JSON 字段投影时覆盖
    :meth:`json_extract_expression` ，公共 SQL 包装由  :meth:`build_json_projection_sql`  完成。
    """

    database_type = "base"
    identifier_quote = '"'
    is_nosql = False
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
        """返回目标表可保证的能力。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param table_metadata: 已取得的表元数据；None 时由后端决定是否需要查询。 默认 ``None``。
        """

        del table_name, table_metadata
        return self.capabilities

    def require_write_mode(
        self,
        table_name: str,
        mode: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        """校验目标表是否支持指定写入模式。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param mode: 写入模式：a 追加且重复键不覆盖，r 追加且重复键覆盖，o 保留结构清空重写，d 删除并重建；仅允许后端可保证的模式。
        :param table_metadata: 已取得的表元数据；None 时由后端决定是否需要查询。 默认 ``None``。
        """

        capabilities = self.capabilities_for_table(table_name, table_metadata)
        if mode not in capabilities.write_modes:
            available = sorted(capabilities.write_modes)
            raise DatabaseCapabilityError(
                f"数据库 {self.database_type} 的目标表 {table_name!r} 不支持写入模式 {mode!r}，"
                f"当前可用模式: {available}"
            )
        return capabilities

    def build_count_sql(self, sql: str) -> str:
        """生成通用子查询计数 SQL。

        :param sql: 待执行或改写的 SQL 文本，不在此参数中拼入绑定值。
        """

        query = sql.rstrip().removesuffix(";").rstrip()
        return f"SELECT COUNT(1) FROM ({query}) hscredit_count"

    def quote_identifier(self, identifier: str) -> str:
        """按当前数据库方言引用单个标识符。

        :param identifier: 单个数据库标识符，不含限定名分隔点。
        """

        if not isinstance(identifier, str) or not identifier:
            raise ValidationError("数据库标识符必须是非空字符串")
        quote = self.identifier_quote
        escaped = identifier.replace(quote, quote + quote)
        return f"{quote}{escaped}{quote}"

    def quote_qualified_name(self, name: str) -> str:
        """逐段引用数据库对象限定名。

        :param name: 由点分隔的多级数据库对象名称。
        """

        from ..writing import split_qualified_name

        return ".".join(self.quote_identifier(part) for part in split_qualified_name(name))

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """生成从单个 JSON 字段提取路径值的 SQL 表达式。

        适配器只负责返回表达式，不添加输出别名。 ``path`` 已经由公共层完成安全校验。

        :param column_sql: 已按当前方言引用的 JSON 源字段表达式。
        :type column_sql: str
        :param path: 以 ``$`` 开头的 JSONPath。
        :type path: str
        :return: 返回标量或嵌套 JSON 原始值的 SQL 表达式。
        :raises DatabaseCapabilityError: 适配器未实现 JSON 字段投影。
        """

        del column_sql, path
        raise DatabaseCapabilityError(f"数据库 {self.database_type} 不支持 JSON 字段投影")

    def build_json_projection_sql(
        self,
        sql: str,
        *,
        columns: Optional[Sequence[str]] = None,
        json_fields: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> str:
        """把原查询包装为仅返回普通字段和指定 JSON 子字段的 SQL。

        :param sql: 输出中必须包含 ``columns`` 和所有 JSON 源字段的原始查询。
        :type sql: str
        :param columns: 默认值为 ``None`` 。原样保留的普通字段名序列；不能包含 ``json_fields`` 的源字段
        :type columns: Optional[Sequence[str]]
        :param json_fields: ``{源字段: {输出字段: 路径或(路径, 默认值)}}`` 映射。
            默认值由 QueryStream 在分块返回前处理，不写入 SQL。

        :type json_fields: Optional[Mapping[str, Mapping[str, Any]]]
        :return: 仅选择目标字段的后端方言 SQL；没有投影时返回原 SQL。
        :raises ValidationError: 字段名、JSONPath、默认值简写或输出名重复无效。
        :raises DatabaseCapabilityError: 当前适配器没有 JSON 路径提取实现。
        """

        projection = normalize_json_projection(columns, json_fields)
        if projection is None:
            return sql

        source_alias = "hscredit_json_source"
        quoted_alias = self.quote_identifier(source_alias)
        select_items = [f"{quoted_alias}.{self.quote_identifier(column)}" for column in projection.columns]
        for field in projection.fields:
            source_sql = f"{quoted_alias}.{self.quote_identifier(field.source_column)}"
            expression = self.json_extract_expression(source_sql, field.path)
            select_items.append(f"{expression} AS {self.quote_identifier(field.output_column)}")

        query = sql.rstrip().removesuffix(";").rstrip()
        return f"SELECT {', '.join(select_items)} FROM ({query}) {quoted_alias}"

    def create_table(
        self,
        data: Any,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """创建目标表。

        :param data: 用于推断建表结构的首批 DataFrame。
        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        """

        raise NotImplementedError

    def ensure_table(
        self,
        data: Any,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
        exists: Optional[bool] = None,
    ) -> Any:
        """目标表不存在时根据首批数据创建表，并容忍并发建表竞态。

        :param data: 用于推断建表结构的首批 DataFrame。
        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        :param exists: 调用方已知的目标存在状态；None 时由适配器检查。 默认 ``None``。
        """

        known_exists = self.table_exists(table_name) if exists is None else exists
        if known_exists is True:
            return None

        options = dict(dialect_options or {})
        options["if_not_exists"] = True
        try:
            return self.create_table(data, table_name, dialect_options=options)
        except Exception as exc:
            if self.is_table_already_exists_error(exc):
                try:
                    concurrent_table_exists = self.table_exists(table_name) is True
                except Exception:
                    concurrent_table_exists = False
                if concurrent_table_exists:
                    return None
            raise

    def is_table_already_exists_error(self, exc: BaseException) -> bool:
        """判断建表异常是否明确表示目标表已存在。

        :param exc: 捕获到的驱动异常，用于识别目标表已存在。
        """

        del exc
        return False

    def table_exists(self, table_name: str) -> Optional[bool]:
        """只读判断目标表是否存在；无法通用判断时返回 ``None`` 。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        """

        del table_name
        return None

    def validate_write(
        self,
        table_name: str,
        mode: str,
        first_batch: Any,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
        table_exists: Optional[bool] = None,
    ) -> None:
        """在任何建表、清空或删除动作之前校验写入能力。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param mode: 写入模式：a 追加且重复键不覆盖，r 追加且重复键覆盖，o 保留结构清空重写，d 删除并重建；仅允许后端可保证的模式。
        :param first_batch: 首批输入 DataFrame，用于验证模式、字段和键。
        :param key_columns: 用于去重或更新匹配的键字段名序列。 默认 ``None``。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        :param table_exists: 调用方已查询的目标表存在状态；None 表示未知。 默认 ``None``。
        """

        del first_batch, key_columns, dialect_options, table_exists
        self.require_write_mode(table_name, mode)

    def resolve_key_columns(
        self,
        table_name: str,
        key_columns: Optional[Sequence[str]],
        first_batch: Any,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Sequence[str]]:
        """解析显式或数据库元数据中的主键字段。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param key_columns: 用于去重或更新匹配的键字段名序列。
        :param first_batch: 首批输入 DataFrame，用于验证模式、字段和键。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        """

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
        """在首个批次写入前校验并准备目标表。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param mode: 写入模式：a 追加且重复键不覆盖，r 追加且重复键覆盖，o 保留结构清空重写，d 删除并重建；仅允许后端可保证的模式。
        :param first_batch: 首批输入 DataFrame，用于验证模式、字段和键。
        :param key_columns: 用于去重或更新匹配的键字段名序列。 默认 ``None``。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        """

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
        """写入一个已经校验的 DataFrame 批次。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param batch: 本次待写入的 DataFrame。
        :param mode: 写入模式：a 追加且重复键不覆盖，r 追加且重复键覆盖，o 保留结构清空重写，d 删除并重建；仅允许后端可保证的模式。
        :param batch_index: 本次写入批次编号，供结果和失败定位使用。
        :param key_columns: 用于去重或更新匹配的键字段名序列。 默认 ``None``。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        """

        raise NotImplementedError

    def finish_write(
        self,
        table_name: str,
        mode: str,
        result: Any,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """完成适配器专有的写入收尾。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        :param mode: 写入模式：a 追加且重复键不覆盖，r 追加且重复键覆盖，o 保留结构清空重写，d 删除并重建；仅允许后端可保证的模式。
        :param result: 累计写入结果对象，供适配器补充收尾统计。
        :param dialect_options: 后端专有建表、分区或写入配置映射。 默认 ``None``。
        """

        del table_name, mode, result, dialect_options

    def atomic_replace_session(self, table_name: str):
        """显式暂存/原子替换会话；默认后端不承诺该能力。

        :param table_name: 目标表限定名，由适配器按后端规则引用。
        """
        raise DatabaseCapabilityError(f"数据库 {self.database_type} 不支持暂存表原子替换")

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        """执行查询。

        :param sql: 待执行或改写的 SQL 文本，不在此参数中拼入绑定值。
        :param params: 驱动绑定参数；None 表示无参数。 默认 ``None``。
        :param result: 结果形式：``'dataframe'``、``'records'`` 或 ``'rows'``。 默认 ``'dataframe'``。
        """

        raise NotImplementedError

    def close(self) -> None:
        """关闭适配器持有的资源。"""

        self._closed = True


__all__ = ["BaseDatabaseAdapter"]
