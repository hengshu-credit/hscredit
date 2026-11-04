"""保留表头语义的严格 JSON 数据转换，不把缺失数值伪装成文本。"""

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def json_safe(value):
    """转换 NumPy、日期和多层表头；元组键使用 JSON 数组文本，冲突时报错。"""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (Path, pd.Period, pd.Timedelta)):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = (
                json.dumps(json_safe(list(key)), ensure_ascii=False, separators=(",", ":"))
                if isinstance(key, tuple)
                else str(key)
            )
            if name in result:
                raise ValueError(f"JSON 字段名转换后发生冲突: {name}")
            result[name] = json_safe(item)
        return result
    if isinstance(value, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        return [json_safe(item) for item in value]
    raise TypeError(f"不支持将 {type(value).__name__} 转换为 JSON 数据")
