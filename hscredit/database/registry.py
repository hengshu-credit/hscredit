"""数据库适配器注册表。

内置适配器以导入字符串保存，仅在创建对应连接时加载模块。
"""

import importlib
from typing import Dict, Iterable, Tuple, Type, Union

from ..exceptions import ValidationError
from .adapters.base import BaseDatabaseAdapter

AdapterEntry = Union[str, Type[BaseDatabaseAdapter]]


_BUILTIN_ADAPTERS: Dict[str, AdapterEntry] = {
    "mysql": "hscredit.database.adapters.mysql:MySQLAdapter",
    "hive": "hscredit.database.adapters.hive:HiveAdapter",
    "impala": "hscredit.database.adapters.impala:ImpalaAdapter",
    "oracle": "hscredit.database.adapters.oracle:OracleAdapter",
    "starrocks": "hscredit.database.adapters.starrocks:StarRocksAdapter",
    "clickhouse": "hscredit.database.adapters.clickhouse:ClickHouseAdapter",
    "maxcompute": "hscredit.database.adapters.maxcompute:MaxComputeAdapter",
}

_ADAPTERS: Dict[str, AdapterEntry] = dict(_BUILTIN_ADAPTERS)
_ALIASES: Dict[str, str] = {
    "mariadb": "mysql",
    "odps": "maxcompute",
    "max_computer": "maxcompute",
    "maxcomputer": "maxcompute",
}
_REGISTERED_ALIASES: Dict[str, Tuple[str, ...]] = {}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("数据库类型必须是非空字符串")
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_adapter_name(name: str) -> str:
    """解析数据库类型或别名的规范名称。"""

    normalized = _normalize_name(name)
    return _ALIASES.get(normalized, normalized)


def register_adapter(
    name: str,
    adapter_class: Type[BaseDatabaseAdapter],
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> None:
    """注册自定义数据库适配器。"""

    canonical = _normalize_name(name)
    if not isinstance(adapter_class, type):
        raise ValidationError("adapter_class 必须是适配器类")
    if canonical in _ADAPTERS and not replace:
        raise ValidationError(f"数据库适配器 {canonical!r} 已经注册")

    normalized_aliases = tuple(_normalize_name(alias) for alias in aliases)
    for alias in normalized_aliases:
        owner = _ALIASES.get(alias)
        if owner is not None and owner != canonical and not replace:
            raise ValidationError(f"数据库适配器别名 {alias!r} 已经注册给 {owner!r}")

    if replace:
        for alias in _REGISTERED_ALIASES.get(canonical, ()):
            if _ALIASES.get(alias) == canonical:
                _ALIASES.pop(alias, None)

    _ADAPTERS[canonical] = adapter_class
    _REGISTERED_ALIASES[canonical] = normalized_aliases
    for alias in normalized_aliases:
        _ALIASES[alias] = canonical


def _load_entry(canonical: str, entry: AdapterEntry) -> Type[BaseDatabaseAdapter]:
    if isinstance(entry, type):
        return entry
    module_name, separator, class_name = entry.partition(":")
    if not separator:
        raise ValidationError(f"数据库适配器导入路径无效: {entry!r}")
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    if not isinstance(adapter_class, type):
        raise ValidationError(f"数据库适配器 {canonical!r} 不是类")
    _ADAPTERS[canonical] = adapter_class
    return adapter_class


def get_adapter_class(name: str) -> Type[BaseDatabaseAdapter]:
    """按数据库类型获取适配器类。"""

    canonical = canonical_adapter_name(name)
    entry = _ADAPTERS.get(canonical)
    if entry is None:
        available = ", ".join(available_adapters())
        raise ValidationError(f"不支持数据库类型 {name!r}，支持的数据库类型: {available}")
    return _load_entry(canonical, entry)


def available_adapters() -> Tuple[str, ...]:
    """返回已注册的规范数据库类型。"""

    return tuple(sorted(_ADAPTERS))


__all__ = [
    "register_adapter",
    "get_adapter_class",
    "available_adapters",
    "canonical_adapter_name",
]
