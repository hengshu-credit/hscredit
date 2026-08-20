"""数据库适配器。

内置适配器由注册表按需导入，导入本包不会加载任何数据库驱动。
"""

from .base import BaseDatabaseAdapter

__all__ = ["BaseDatabaseAdapter"]
