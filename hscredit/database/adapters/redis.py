"""Redis 原生连接池和统一 NoSQL 读写适配器。"""

from collections.abc import Mapping
from typing import Any, List, Optional

from ...exceptions import DependencyError, ValidationError
from ..exceptions import DatabaseQueryError, DatabaseWriteError
from ..types import DatabaseCapabilities, NoSQLWriteResult, RedisPoolOptions
from .base import BaseDatabaseAdapter


class RedisAdapter(BaseDatabaseAdapter):
    """基于 redis-py 的 Redis 连接池适配器。"""

    database_type = "redis"
    pool_options_class = RedisPoolOptions
    capabilities = DatabaseCapabilities(
        transactions=False,
        streaming_read=False,
        native_bulk_write=True,
        metadata_export=False,
        write_modes=frozenset(),
    )

    def __init__(self, *, connect_kwargs, pool_options, adapter_options=None):
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        redis_module = self.load_redis_module()
        pool_class = (
            redis_module.BlockingConnectionPool
            if self.pool_options.blocking
            else redis_module.ConnectionPool
        )
        connection_options = dict(self.connect_kwargs)
        url = connection_options.pop("url", None)
        pool_kwargs = {**connection_options, **self.pool_options.to_redis_kwargs()}
        self.pool = (
            pool_class.from_url(url, **pool_kwargs)
            if url is not None
            else pool_class(**pool_kwargs)
        )
        self.client = redis_module.Redis(connection_pool=self.pool)

    def load_redis_module(self) -> Any:
        """按需加载 redis-py。"""

        try:
            import redis
        except ImportError as exc:
            raise DependencyError("缺少 Redis 可选依赖，请安装: pip install hscredit[db-redis]") from exc
        return redis

    @staticmethod
    def _validate_key(key: Any) -> Any:
        if not isinstance(key, (str, bytes)) or not key:
            raise ValidationError("Redis key 必须是非空字符串或 bytes")
        return key

    @classmethod
    def _validate_keys(cls, keys: Any) -> List[Any]:
        if isinstance(keys, (str, bytes)):
            raise ValidationError("Redis 批量 keys 必须是非字符串可迭代对象")
        try:
            materialized = list(keys)
        except TypeError as exc:
            raise ValidationError("Redis 批量 keys 必须是可迭代对象") from exc
        if not materialized:
            raise ValidationError("Redis 批量 keys 不能为空")
        return [cls._validate_key(key) for key in materialized]

    @staticmethod
    def _reject_selector(selector: Any) -> None:
        if selector is not None:
            raise ValidationError("Redis 读取和删除不支持 selector")

    def read_one(self, key: Any, selector: Any = None, **options: Any) -> Any:
        self.ensure_open()
        self._reject_selector(selector)
        self._validate_key(key)
        try:
            return self.client.get(key, **options)
        except Exception as exc:
            raise DatabaseQueryError(f"读取 Redis key {key!r} 失败") from exc

    def read_many(self, keys: Any, selector: Any = None, **options: Any) -> List[Any]:
        self.ensure_open()
        self._reject_selector(selector)
        materialized = self._validate_keys(keys)
        try:
            return list(self.client.mget(materialized, **options))
        except Exception as exc:
            raise DatabaseQueryError("批量读取 Redis keys 失败") from exc

    def read(self, resource: Any, selector: Any = None, **options: Any) -> Any:
        if isinstance(resource, (str, bytes)):
            return self.read_one(resource, selector, **options)
        return self.read_many(resource, selector, **options)

    def write_one(
        self,
        key: Any,
        value: Any,
        *,
        ttl: Optional[int] = None,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        self._validate_key(key)
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0):
            raise ValidationError("Redis ttl 必须是正整数或 None")
        set_options = dict(options)
        if ttl is not None:
            set_options["ex"] = ttl
        try:
            written = bool(self.client.set(key, value, **set_options))
        except Exception as exc:
            raise DatabaseWriteError(f"写入 Redis key {key!r} 失败") from exc
        return NoSQLWriteResult(
            operation="write",
            acknowledged=written,
            affected_count=int(written),
            identifiers=(key,),
        )

    def write_many(
        self,
        mapping: Any,
        data: Any = None,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        if data is not None:
            raise ValidationError("Redis write_many 只接受一个 key-value 映射")
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValidationError("Redis 批量写入映射不能为空")
        materialized = dict(mapping)
        for key in materialized:
            self._validate_key(key)
        try:
            written = bool(self.client.mset(materialized, **options))
        except Exception as exc:
            raise DatabaseWriteError("批量写入 Redis keys 失败") from exc
        return NoSQLWriteResult(
            operation="write",
            acknowledged=written,
            affected_count=len(materialized) if written else 0,
            identifiers=tuple(materialized),
        )

    def write(self, resource: Any, data: Any = None, **options: Any) -> NoSQLWriteResult:
        if isinstance(resource, Mapping) and data is None:
            return self.write_many(resource, **options)
        return self.write_one(resource, data, **options)

    def delete_one(self, key: Any, selector: Any = None, **options: Any) -> NoSQLWriteResult:
        self.ensure_open()
        self._reject_selector(selector)
        self._validate_key(key)
        try:
            deleted = int(self.client.delete(key, **options))
        except Exception as exc:
            raise DatabaseWriteError(f"删除 Redis key {key!r} 失败") from exc
        return NoSQLWriteResult(
            operation="delete",
            acknowledged=True,
            affected_count=deleted,
            identifiers=(key,),
        )

    def delete_many(self, keys: Any, selector: Any = None, **options: Any) -> NoSQLWriteResult:
        self.ensure_open()
        self._reject_selector(selector)
        materialized = self._validate_keys(keys)
        try:
            deleted = int(self.client.delete(*materialized, **options))
        except Exception as exc:
            raise DatabaseWriteError("批量删除 Redis keys 失败") from exc
        return NoSQLWriteResult(
            operation="delete",
            acknowledged=True,
            affected_count=deleted,
            identifiers=tuple(materialized),
        )

    def delete(self, resource: Any, selector: Any = None, **options: Any) -> NoSQLWriteResult:
        if isinstance(resource, (str, bytes)):
            return self.delete_one(resource, selector, **options)
        return self.delete_many(resource, selector, **options)

    def exists(self, key: Any, selector: Any = None, **options: Any) -> bool:
        self.ensure_open()
        self._reject_selector(selector)
        self._validate_key(key)
        try:
            return bool(self.client.exists(key, **options))
        except Exception as exc:
            raise DatabaseQueryError(f"检查 Redis key {key!r} 是否存在失败") from exc

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        del sql, params, result
        raise DatabaseQueryError("Redis 不支持 SQL query，请使用 read/read_one/read_many")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.client.close()
        finally:
            try:
                self.pool.disconnect()
            finally:
                super().close()


__all__ = ["RedisAdapter"]
