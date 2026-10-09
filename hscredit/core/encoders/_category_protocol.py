"""编码器共享类别哨兵与无执行能力的类型化 JSON 协议。"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CategoryToken:
    """内部控制键，与所有业务字符串严格分离。"""

    kind: str


MISSING = CategoryToken("missing")
UNKNOWN = CategoryToken("unknown")
OTHER = CategoryToken("other")
INFREQUENT = CategoryToken("infrequent")


def pack(value):
    """以记录而非 JSON 对象键保存类型；不保存可执行对象。"""
    if isinstance(value, CategoryToken):
        return {"type": "token", "value": value.kind}
    if value is pd.NA:
        return {"type": "pd.NA"}
    if value is pd.NaT:
        return {"type": "pd.NaT"}
    if value is None:
        return {"type": "none"}
    if isinstance(value, np.generic):
        if value.dtype.kind == "S":
            raise ValueError("映射 JSON 不支持 numpy bytes 类别，请使用字符串或完整制品")
        return {"type": "numpy", "dtype": value.dtype.str, "value": str(value)}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, complex):
        return {"type": "complex", "real": pack(value.real), "imag": pack(value.imag)}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta", "value": str(value.value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {"type": "duration", "value": value.total_seconds()}
    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "index": pack(value.index.tolist()),
            "values": pack(value.tolist()),
            "name": pack(value.name),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "array",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": pack(value.ravel().tolist()),
        }
    if isinstance(value, dict):
        for key in value:
            if isinstance(key, (Decimal, complex, np.complexfloating)) and pd.isna(key):
                raise ValueError("Decimal/complex NaN类别键可能依赖对象身份，不能保证JSON往返；请先转为稳定的类别标识")
        return {"type": "dict", "items": [[pack(key), pack(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [pack(item) for item in value]}
    raise ValueError(f"映射 JSON 不支持类型 {type(value).__name__}，完整对象请使用 save_artifact")


def unpack(record):
    """只允许固定的无执行类型，不按模块/类名动态导入。"""
    if not isinstance(record, dict) or "type" not in record:
        raise ValueError("编码映射包含无效类型记录")
    kind = record["type"]
    if kind == "token":
        if record.get("value") not in {"missing", "unknown", "other", "infrequent"}:
            raise ValueError("编码映射包含未知内部类别")
        return CategoryToken(record["value"])
    if kind == "none":
        return None
    if kind == "pd.NA":
        return pd.NA
    if kind == "pd.NaT":
        return pd.NaT
    if kind == "bool":
        if not isinstance(record.get("value"), bool):
            raise ValueError("映射布尔类型无效")
        return record["value"]
    if kind == "str":
        if not isinstance(record.get("value"), str):
            raise ValueError("映射字符串类型无效")
        return record["value"]
    if kind == "int":
        return int(record["value"])
    if kind == "float":
        return float.fromhex(record["value"])
    if kind == "numpy":
        dtype = np.dtype(record["dtype"])
        if dtype.kind not in "biufcMmU":
            raise ValueError("映射 numpy 标量类型无效")
        if dtype.kind == "b":
            if record["value"] not in {"True", "False"}:
                raise ValueError("映射 numpy 布尔类型无效")
            return np.bool_(record["value"] == "True")
        return np.asarray(record["value"], dtype=dtype)[()]
    if kind == "decimal":
        return Decimal(record["value"])
    if kind == "complex":
        return complex(unpack(record["real"]), unpack(record["imag"]))
    if kind == "timestamp":
        return pd.Timestamp(record["value"])
    if kind == "timedelta":
        return pd.Timedelta(int(record["value"]), unit="ns")
    if kind == "datetime":
        return datetime.fromisoformat(record["value"])
    if kind == "date":
        return date.fromisoformat(record["value"])
    if kind == "duration":
        return timedelta(seconds=record["value"])
    if kind in {"list", "tuple"}:
        values = [unpack(item) for item in record["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        result = {}
        for key_record, value_record in record["items"]:
            key = unpack(key_record)
            if key in result:
                raise ValueError("编码映射包含重复键")
            result[key] = unpack(value_record)
        return result
    if kind == "series":
        return pd.Series(
            unpack(record["values"]), index=unpack(record["index"]), name=unpack(record["name"]), dtype=record["dtype"]
        )
    if kind == "array":
        return np.asarray(unpack(record["values"]), dtype=record["dtype"]).reshape(record["shape"])
    raise ValueError(f"编码映射包含不支持的类型记录: {kind}")
