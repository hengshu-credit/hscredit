"""数据库模块异常。

所有异常均兼容 hscredit 统一异常体系，并为调用方保留 SQL、绑定参数和底层驱动异常链。
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any, Iterator, Optional, Tuple, Type

from ..exceptions import HSCreditError
from .types import WriteResult

_REDACTED_DRIVER_MESSAGE = "[已脱敏：数据库错误包含绑定参数]"


class DatabaseDriverError(RuntimeError):
    """仅用于 traceback 的脱敏数据库驱动异常代理。"""


class DatabaseError(HSCreditError):
    """数据库模块基础异常，并提供可见且可编程读取的执行上下文。

    **参数**

    message : str
        面向用户的中文错误概要。
    sql : str, optional
        实际交给数据库执行的 SQL。
    params : Any, optional
        SQL 绑定参数；只作为属性保存，不进入异常文本。
    driver_error : BaseException, optional
        底层数据库驱动返回的原始异常。
    cleanup_error : BaseException, optional
        主操作失败后发生的资源清理异常。

    **属性**

    sql : str, optional
        当前异常链中最接近底层的 SQL。
    params : Any, optional
        对应 SQL 的绑定参数。
    driver_error : BaseException, optional
        当前异常链中最底层的数据库驱动异常。
    cleanup_error : BaseException, optional
        关闭游标、连接或查询流时发生的次要异常。

    **参考样例**

    >>> try:
    ...     db.execute("DELETE FROM events WHERE id=%s", params=(1,))
    ... except DatabaseError as exc:
    ...     print(exc)          # 显示 SQL 和数据库原始错误，但不显示绑定参数
    ...     print(exc.params)   # 确有排查需要时显式查看参数
    """

    def __init__(
        self,
        message: str,
        *,
        sql: Optional[str] = None,
        params: Any = None,
        driver_error: Optional[BaseException] = None,
        cleanup_error: Optional[BaseException] = None,
    ):
        self.message = str(message)
        self._sql = sql
        self._params = params
        self._driver_error = driver_error
        self._cleanup_error = cleanup_error
        super().__init__(self.message)

    def _cause_chain(self) -> Iterator[BaseException]:
        current = self.__cause__
        seen = {id(self)}
        while isinstance(current, BaseException) and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.__cause__

    @property
    def sql(self) -> Optional[str]:
        """当前异常链中实际执行的 SQL。"""

        if self._sql is not None:
            return self._sql
        for error in self._cause_chain():
            if isinstance(error, DatabaseError) and error._sql is not None:
                return error._sql
        return None

    @property
    def params(self) -> Any:
        """SQL 绑定参数；为避免敏感信息泄露，不进入异常文本。"""

        if self._params is not None:
            return self._params
        for error in self._cause_chain():
            if isinstance(error, DatabaseError) and error._params is not None:
                return error._params
        return None

    @property
    def driver_error(self) -> Optional[BaseException]:
        """当前异常链中最底层的数据库驱动异常。"""

        if self._driver_error is not None:
            return self._resolve_driver_error(self._driver_error)
        for error in self._cause_chain():
            resolved = self._resolve_driver_error(error)
            if resolved is not None:
                return resolved
        return None

    @classmethod
    def _resolve_driver_error(
        cls,
        error: Optional[BaseException],
        seen: Optional[set] = None,
    ) -> Optional[BaseException]:
        if error is None:
            return None
        visited = set() if seen is None else seen
        if id(error) in visited:
            return None
        visited.add(id(error))
        if not isinstance(error, DatabaseError):
            return error
        if error._driver_error is not None:
            return cls._resolve_driver_error(error._driver_error, visited)
        for cause in error._cause_chain():
            resolved = cls._resolve_driver_error(cause, visited)
            if resolved is not None:
                return resolved
        return error

    @property
    def cleanup_error(self) -> Optional[BaseException]:
        """主操作失败后发生的资源清理异常。"""

        if self._cleanup_error is not None:
            return self._cleanup_error
        for error in self._cause_chain():
            if isinstance(error, DatabaseError) and error._cleanup_error is not None:
                return error._cleanup_error
        return None

    @staticmethod
    def _parameter_tokens(params: Any) -> Tuple[set, bool]:
        tokens = set()
        visited = set()
        uncertain = False

        def visit(value: Any) -> None:
            nonlocal uncertain
            if value is None:
                return
            if isinstance(value, str):
                if value:
                    tokens.add(value)
                return
            if isinstance(value, bytes):
                uncertain = True
                return
            if isinstance(value, Mapping):
                if id(value) in visited:
                    return
                visited.add(id(value))
                try:
                    for item in value.values():
                        visit(item)
                except Exception:
                    uncertain = True
                return
            if isinstance(value, (list, tuple, set, frozenset)):
                if id(value) in visited:
                    return
                visited.add(id(value))
                try:
                    for item in value:
                        visit(item)
                except Exception:
                    uncertain = True
                return
            if isinstance(value, bool):
                return
            if isinstance(value, Iterable):
                uncertain = True
                return
            try:
                rendered = (str(value), repr(value))
            except Exception:
                uncertain = True
                return
            tokens.update(text for text in rendered if text)

        visit(params)
        return tokens, uncertain

    @classmethod
    def _visible_error_detail(cls, error: BaseException, params: Any) -> str:
        detail = error.message if isinstance(error, DatabaseError) else str(error)
        tokens, uncertain = cls._parameter_tokens(params)
        if uncertain:
            return _REDACTED_DRIVER_MESSAGE
        for token in sorted(tokens, key=len, reverse=True):
            contains_token = token in detail if len(token) >= 3 else bool(
                re.search(rf"(?<!\w){re.escape(token)}(?!\w)", detail)
            )
            if contains_token:
                return _REDACTED_DRIVER_MESSAGE
        return detail

    def __str__(self) -> str:
        parts = [self.message]
        sql = self.sql
        if sql is not None:
            parts.append(f"执行SQL:\n{sql}")
        driver_error = self.driver_error
        if driver_error is not None and driver_error is not self:
            detail = self._visible_error_detail(driver_error, self.params)
            parts.append(f"数据库错误: {type(driver_error).__name__}: {detail}")
        cleanup_error = self.cleanup_error
        if cleanup_error is not None:
            detail = self._visible_error_detail(cleanup_error, self.params)
            parts.append(f"资源清理错误: {type(cleanup_error).__name__}: {detail}")
        return "\n".join(parts)


class DatabaseConnectionError(DatabaseError):
    """数据库连接或连接池操作失败。"""


class DatabaseQueryError(DatabaseError):
    """数据库查询或 SQL 执行失败。"""


class DatabaseWriteError(DatabaseError):
    """数据库写入失败，并可携带部分写入结果。

    除 :class:`DatabaseError` 的 SQL 上下文属性外，``result`` 保存失败前已经提交的批次和行数统计。
    """

    def __init__(
        self,
        message: str,
        *,
        result: Optional[WriteResult] = None,
        sql: Optional[str] = None,
        params: Any = None,
        driver_error: Optional[BaseException] = None,
        cleanup_error: Optional[BaseException] = None,
    ):
        super().__init__(
            message,
            sql=sql,
            params=params,
            driver_error=driver_error,
            cleanup_error=cleanup_error,
        )
        self.result = result


def database_error_from(
    error_type: Type[DatabaseError],
    message: str,
    *,
    cause: BaseException,
    sql: Optional[str] = None,
    params: Any = None,
    cleanup_error: Optional[BaseException] = None,
    **kwargs: Any,
) -> DatabaseError:
    """构造保留原始异常属性、但 traceback 仅链接脱敏代理的数据库异常。"""

    driver_error = DatabaseError._resolve_driver_error(cause) or cause
    error = error_type(
        message,
        sql=sql,
        params=params,
        driver_error=driver_error,
        cleanup_error=cleanup_error,
        **kwargs,
    )
    detail = DatabaseError._visible_error_detail(driver_error, params)
    error.__cause__ = DatabaseDriverError(f"{type(driver_error).__name__}: {detail}")
    error.__suppress_context__ = True
    return error


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
