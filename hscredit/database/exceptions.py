"""数据库模块异常。

所有异常均兼容 hscredit 统一异常体系，并为调用方保留底层驱动异常链。
"""

from typing import Optional

from ..exceptions import HSCreditError
from .types import WriteResult


class DatabaseError(HSCreditError):
    """数据库模块基础异常。"""


class DatabaseConnectionError(DatabaseError):
    """数据库连接或连接池操作失败。"""


class DatabaseQueryError(DatabaseError):
    """数据库查询或 SQL 执行失败。"""


class DatabaseWriteError(DatabaseError):
    """数据库写入失败，并可携带部分写入结果。"""

    def __init__(self, message: str, *, result: Optional[WriteResult] = None):
        super().__init__(message)
        self.result = result


class DatabaseMetadataError(DatabaseError):
    """数据库元数据读取或导出失败。"""


class DatabaseCapabilityError(DatabaseError):
    """数据库或目标表不支持所请求的能力。"""


__all__ = [
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseWriteError",
    "DatabaseMetadataError",
    "DatabaseCapabilityError",
]
