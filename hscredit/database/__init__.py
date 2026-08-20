"""可扩展数据库读写模块。

数据库驱动均为可选依赖，本模块的公共契约可在未安装任何驱动时安全导入。
"""

from .exceptions import (
    DatabaseCapabilityError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseMetadataError,
    DatabaseQueryError,
    DatabaseWriteError,
)
from .types import (
    WRITE_MODES,
    DatabaseCapabilities,
    PoolOptions,
    StreamState,
    WriteResult,
)
from .client import Database
from .registry import available_adapters, get_adapter_class, register_adapter
from .stream import QueryStream
from .metadata import METADATA_COLUMNS_ZH, MetadataInspection, QualifiedTarget
from .writing import BatchWriteResult, iter_write_batches

__all__ = [
    "WRITE_MODES",
    "StreamState",
    "PoolOptions",
    "DatabaseCapabilities",
    "WriteResult",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseWriteError",
    "DatabaseMetadataError",
    "DatabaseCapabilityError",
    "Database",
    "register_adapter",
    "get_adapter_class",
    "available_adapters",
    "QueryStream",
    "METADATA_COLUMNS_ZH",
    "QualifiedTarget",
    "MetadataInspection",
    "BatchWriteResult",
    "iter_write_batches",
]
