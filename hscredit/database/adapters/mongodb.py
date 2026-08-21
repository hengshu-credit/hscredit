"""MongoDB 原生连接池和统一 NoSQL 读写适配器。"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from ...exceptions import DependencyError, ValidationError
from ..exceptions import DatabaseQueryError, DatabaseWriteError
from ..types import DatabaseCapabilities, MongoPoolOptions, NoSQLWriteResult
from .base import BaseDatabaseAdapter


class MongoDBAdapter(BaseDatabaseAdapter):
    """基于 PyMongo ``MongoClient`` 内建连接池的 MongoDB 适配器。"""

    database_type = "mongodb"
    pool_options_class = MongoPoolOptions
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
        pymongo = self.load_pymongo_module()
        client_options = dict(self.connect_kwargs)
        uri = client_options.pop("uri", None)
        database_name = client_options.pop("database", None)
        pool_kwargs = self.pool_options.to_mongo_kwargs()
        self.client = (
            pymongo.MongoClient(uri, **client_options, **pool_kwargs)
            if uri is not None
            else pymongo.MongoClient(**client_options, **pool_kwargs)
        )
        if database_name is None:
            try:
                self.database = self.client.get_default_database()
            except Exception as exc:
                self.client.close()
                raise ValidationError("MongoDB 必须通过 database 参数或 URI 指定数据库") from exc
        else:
            if not isinstance(database_name, str) or not database_name.strip():
                self.client.close()
                raise ValidationError("MongoDB database 必须是非空字符串")
            self.database = self.client[database_name]

    def load_pymongo_module(self) -> Any:
        """按需加载 PyMongo。"""

        try:
            import pymongo
        except ImportError as exc:
            raise DependencyError("缺少 MongoDB 可选依赖，请安装: pip install hscredit[db-mongodb]") from exc
        return pymongo

    @staticmethod
    def _validate_collection_name(collection_name: Any) -> str:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValidationError("MongoDB 集合名必须是非空字符串")
        return collection_name

    @staticmethod
    def _normalize_selector(selector: Any) -> Dict[str, Any]:
        if selector is None:
            return {}
        if not isinstance(selector, Mapping):
            raise ValidationError("MongoDB selector 必须是映射或 None")
        return dict(selector)

    @staticmethod
    def _validate_document(document: Any) -> Dict[str, Any]:
        if not isinstance(document, Mapping) or not document:
            raise ValidationError("MongoDB 文档必须是非空映射")
        return dict(document)

    @classmethod
    def _validate_documents(cls, documents: Any) -> List[Dict[str, Any]]:
        if isinstance(documents, (str, bytes, Mapping)):
            raise ValidationError("MongoDB 文档序列必须是非映射可迭代对象")
        try:
            materialized = list(documents)
        except TypeError as exc:
            raise ValidationError("MongoDB 文档序列必须是可迭代对象") from exc
        if not materialized:
            raise ValidationError("MongoDB 文档序列不能为空")
        return [cls._validate_document(document) for document in materialized]

    def _collection(self, collection_name: Any) -> Any:
        return self.database[self._validate_collection_name(collection_name)]

    def read_one(
        self,
        collection_name: Any,
        selector: Any = None,
        *,
        projection: Any = None,
        **options: Any,
    ) -> Any:
        self.ensure_open()
        collection = self._collection(collection_name)
        query = self._normalize_selector(selector)
        try:
            return collection.find_one(query, projection, **options)
        except Exception as exc:
            raise DatabaseQueryError(f"读取 MongoDB 集合 {collection_name!r} 的单个文档失败") from exc

    def read_many(
        self,
        collection_name: Any,
        selector: Any = None,
        *,
        projection: Any = None,
        sort: Any = None,
        skip: int = 0,
        limit: int = 0,
        **options: Any,
    ) -> List[Any]:
        self.ensure_open()
        collection = self._collection(collection_name)
        query = self._normalize_selector(selector)
        for name, value in (("skip", skip), ("limit", limit)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"MongoDB {name} 必须是非负整数")
        try:
            cursor = collection.find(query, projection, **options)
            if sort is not None:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as exc:
            raise DatabaseQueryError(f"批量读取 MongoDB 集合 {collection_name!r} 失败") from exc

    def read(
        self,
        collection_name: Any,
        selector: Any = None,
        *,
        many: Optional[bool] = None,
        **options: Any,
    ) -> Any:
        if many is not None and not isinstance(many, bool):
            raise ValidationError("MongoDB many 必须是布尔值或 None")
        limit = options.get("limit", 0)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValidationError("MongoDB limit 必须是非负整数")
        use_one = many is False or (many is None and limit == 1)
        if use_one:
            options.pop("limit", None)
            return self.read_one(collection_name, selector, **options)
        return self.read_many(collection_name, selector, **options)

    @staticmethod
    def _insert_result(result: Any, identifiers: Any) -> NoSQLWriteResult:
        ids = tuple(identifiers)
        acknowledged = bool(result.acknowledged)
        return NoSQLWriteResult(
            operation="write",
            acknowledged=acknowledged,
            affected_count=len(ids) if acknowledged else 0,
            identifiers=ids,
        )

    @staticmethod
    def _update_result(result: Any) -> NoSQLWriteResult:
        upserted_id = getattr(result, "upserted_id", None)
        modified_count = int(result.modified_count)
        return NoSQLWriteResult(
            operation="write",
            acknowledged=bool(result.acknowledged),
            affected_count=modified_count + int(upserted_id is not None),
            matched_count=int(result.matched_count),
            modified_count=modified_count,
            identifiers=() if upserted_id is None else (upserted_id,),
        )

    def write_one(
        self,
        collection_name: Any,
        data: Any,
        *,
        selector: Any = None,
        mode: str = "insert",
        upsert: bool = False,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        collection = self._collection(collection_name)
        document = self._validate_document(data)
        if mode not in {"insert", "update", "replace"}:
            raise ValidationError("MongoDB write mode 只支持 insert、update 或 replace")
        if not isinstance(upsert, bool):
            raise ValidationError("MongoDB upsert 必须是布尔值")
        query = self._normalize_selector(selector)
        if mode != "insert" and not query:
            raise ValidationError(f"MongoDB {mode} 写入必须提供非空 selector")
        try:
            if mode == "insert":
                if selector is not None:
                    raise ValidationError("MongoDB insert 写入不接受 selector")
                result = collection.insert_one(document, **options)
                return self._insert_result(result, (result.inserted_id,))
            if mode == "replace":
                result = collection.replace_one(query, document, upsert=upsert, **options)
            else:
                result = collection.update_one(query, document, upsert=upsert, **options)
            return self._update_result(result)
        except ValidationError:
            raise
        except Exception as exc:
            raise DatabaseWriteError(f"写入 MongoDB 集合 {collection_name!r} 的单个文档失败") from exc

    def write_many(
        self,
        collection_name: Any,
        data: Any,
        *,
        selector: Any = None,
        mode: str = "insert",
        upsert: bool = False,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        collection = self._collection(collection_name)
        if mode not in {"insert", "update"}:
            raise ValidationError("MongoDB 批量 write mode 只支持 insert 或 update")
        if not isinstance(upsert, bool):
            raise ValidationError("MongoDB upsert 必须是布尔值")
        try:
            if mode == "insert":
                if selector is not None:
                    raise ValidationError("MongoDB 批量 insert 写入不接受 selector")
                documents = self._validate_documents(data)
                result = collection.insert_many(documents, **options)
                return self._insert_result(result, result.inserted_ids)
            query = self._normalize_selector(selector)
            if not query:
                raise ValidationError("MongoDB 批量 update 写入必须提供非空 selector")
            update = self._validate_document(data)
            result = collection.update_many(query, update, upsert=upsert, **options)
            return self._update_result(result)
        except ValidationError:
            raise
        except Exception as exc:
            raise DatabaseWriteError(f"批量写入 MongoDB 集合 {collection_name!r} 失败") from exc

    def write(self, collection_name: Any, data: Any, **options: Any) -> NoSQLWriteResult:
        many = options.pop("many", None)
        if many is not None and not isinstance(many, bool):
            raise ValidationError("MongoDB many 必须是布尔值或 None")
        use_many = many is True or (many is None and not isinstance(data, Mapping))
        if use_many:
            return self.write_many(collection_name, data, **options)
        return self.write_one(collection_name, data, **options)

    @staticmethod
    def _delete_result(result: Any) -> NoSQLWriteResult:
        return NoSQLWriteResult(
            operation="delete",
            acknowledged=bool(result.acknowledged),
            affected_count=int(result.deleted_count),
        )

    def delete_one(
        self,
        collection_name: Any,
        selector: Any = None,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        collection = self._collection(collection_name)
        query = self._normalize_selector(selector)
        try:
            return self._delete_result(collection.delete_one(query, **options))
        except Exception as exc:
            raise DatabaseWriteError(f"删除 MongoDB 集合 {collection_name!r} 的单个文档失败") from exc

    def delete_many(
        self,
        collection_name: Any,
        selector: Any = None,
        *,
        allow_all: bool = False,
        **options: Any,
    ) -> NoSQLWriteResult:
        self.ensure_open()
        collection = self._collection(collection_name)
        query = self._normalize_selector(selector)
        if not isinstance(allow_all, bool):
            raise ValidationError("MongoDB allow_all 必须是布尔值")
        if not query and not allow_all:
            raise ValidationError("MongoDB 批量删除空 selector 时必须显式设置 allow_all=True")
        try:
            return self._delete_result(collection.delete_many(query, **options))
        except Exception as exc:
            raise DatabaseWriteError(f"批量删除 MongoDB 集合 {collection_name!r} 的文档失败") from exc

    def delete(
        self,
        collection_name: Any,
        selector: Any = None,
        *,
        many: bool = False,
        **options: Any,
    ) -> NoSQLWriteResult:
        if not isinstance(many, bool):
            raise ValidationError("MongoDB many 必须是布尔值")
        if many:
            return self.delete_many(collection_name, selector, **options)
        return self.delete_one(collection_name, selector, **options)

    def exists(self, collection_name: Any, selector: Any = None, **options: Any) -> bool:
        projection = options.pop("projection", {"_id": 1})
        return self.read_one(
            collection_name,
            selector,
            projection=projection,
            **options,
        ) is not None

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        del sql, params, result
        raise DatabaseQueryError("MongoDB 不支持 SQL query，请使用 read/read_one/read_many")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.client.close()
        finally:
            super().close()


__all__ = ["MongoDBAdapter"]
