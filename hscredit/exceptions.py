"""hscredit 统一异常体系.

提供项目级的异常类型，兼容原生异常语义，方便调用方统一捕获。
"""

from typing import Iterable, List


class HSCreditError(Exception):
    """hscredit 的基础异常类型。"""


class ValidationError(ValueError, HSCreditError):
    """参数或数据校验失败。"""


class InputValidationError(ValidationError):
    """输入数据校验失败。"""


class InputTypeError(TypeError, HSCreditError):
    """输入类型不符合要求。"""


class FeatureNotFoundError(KeyError, HSCreditError):
    """特征或字段不存在。"""


class StateError(RuntimeError, HSCreditError):
    """对象状态不符合预期。"""


class NotFittedError(ValueError, StateError):
    """对象尚未完成拟合。"""


class DependencyError(ImportError, HSCreditError):
    """缺少可选依赖。"""


class SerializationError(HSCreditError):
    """序列化或反序列化失败。"""


def raise_not_fitted(component_name: str, action: str = "请先调用fit方法") -> None:
    """抛出统一的未拟合异常。"""
    raise NotFittedError(f"{component_name}尚未拟合，{action}")


def raise_feature_not_found(feature_name: str, owner_name: str = "特征") -> None:
    """抛出统一的字段不存在异常。"""
    raise FeatureNotFoundError(f"{owner_name} '{feature_name}' 未找到")


def raise_missing_columns(columns: Iterable[str], data_name: str = "数据集") -> None:
    """抛出统一的缺失字段异常。"""
    missing: List[str] = list(columns)
    if missing:
        raise FeatureNotFoundError(f"{data_name}缺少以下字段: {missing}")


__all__ = [
    "HSCreditError",
    "ValidationError",
    "InputValidationError",
    "InputTypeError",
    "FeatureNotFoundError",
    "StateError",
    "NotFittedError",
    "DependencyError",
    "SerializationError",
    "raise_not_fitted",
    "raise_feature_not_found",
    "raise_missing_columns",
]