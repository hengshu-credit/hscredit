"""多框架超参数搜索空间到 Optuna 的内部适配器。

本模块只依赖 hscredit 已有的 NumPy、SciPy 与 Optuna。公开声明对象位于
``search_space.py``；这里负责格式识别、统一校验、采样、公开参数还原，以及
手工搜索点到 ``Study.enqueue_trial`` 所需参数的逆变换。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import norm

try:
    import optuna
except ImportError:  # pragma: no cover - ModelTuner 会给出面向用户的依赖提示
    optuna = None


_LATENT_PREFIX = "__hscredit__"
_NORMAL_TAIL = 4.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _is_integer_type(value: Any) -> bool:
    try:
        return value is int or (isinstance(value, type) and issubclass(value, np.integer))
    except TypeError:
        return False


def _is_float_type(value: Any) -> bool:
    try:
        return value is float or (isinstance(value, type) and issubclass(value, np.floating))
    except TypeError:
        return False


def _check_bounds(name: str, low: Any, high: Any) -> None:
    if not _is_number(low) or not _is_number(high):
        raise ValueError(f"参数 {name!r} 的上下界必须是数值")
    if not np.isfinite(float(low)) or not np.isfinite(float(high)):
        raise ValueError(f"参数 {name!r} 的上下界必须是有限数值")
    if low > high:
        raise ValueError(f"参数 {name!r} 的下界({low})不能大于上界({high})")


def _positive_step(name: str, value: Any, field: str) -> float:
    try:
        step = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参数 {name!r} 的 {field} 必须是大于 0 的有限数值") from exc
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"参数 {name!r} 的 {field} 必须是大于 0 的有限数值")
    return step


def _categorical_spec(name: str, choices: Sequence[Any], prior: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    choices = list(choices)
    if not choices:
        raise ValueError(f"参数 {name!r} 的 choices 不能为空列表")
    result: Dict[str, Any] = {"type": "categorical", "choices": choices}
    if prior is not None:
        weights = np.asarray(prior, dtype=float)
        if len(weights) != len(choices):
            raise ValueError(f"参数 {name!r} 的 prior 长度必须与 choices 一致")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"参数 {name!r} 的 prior 必须是总和大于 0 的非负有限数值")
        result["prior"] = (weights / weights.sum()).tolist()
    return result


def _normalize_dict(name: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    if "type" not in raw:
        raise ValueError(f"参数 {name!r} 的搜索空间字典缺少 'type' 键: {raw!r}")
    kind = str(raw["type"]).strip().lower()

    if kind == "categorical":
        return _categorical_spec(name, raw.get("choices", raw.get("options", [])), raw.get("prior"))
    if kind == "choice":
        return _categorical_spec(name, raw.get("choices", raw.get("options", [])))

    if kind in {"int", "float", "uniform", "loguniform", "quniform", "qloguniform", "randint"}:
        if "low" not in raw or "high" not in raw:
            raise ValueError(f"参数 {name!r} 的搜索空间必须提供 'low' 和 'high': {raw!r}")
        low, high = raw["low"], raw["high"]
        _check_bounds(name, low, high)

        if kind in {"int", "randint"}:
            if not isinstance(low, (int, np.integer)) or isinstance(low, (bool, np.bool_)):
                raise ValueError(f"参数 {name!r} 的 int 类型要求整数边界: {raw!r}")
            if not isinstance(high, (int, np.integer)) or isinstance(high, (bool, np.bool_)):
                raise ValueError(f"参数 {name!r} 的 int 类型要求整数边界: {raw!r}")
            result: Dict[str, Any] = {"type": "int", "low": int(low), "high": int(high)}
            if raw.get("log", False):
                if int(low) <= 0:
                    raise ValueError(f"参数 {name!r} 使用 log=True 时下界必须大于 0")
                if raw.get("step", 1) != 1:
                    raise ValueError(f"参数 {name!r} 使用 log=True 时 step 必须为 1")
                result["log"] = True
            elif raw.get("step") is not None:
                step = _positive_step(name, raw["step"], "step")
                if not step.is_integer():
                    raise ValueError(f"参数 {name!r} 的 int step 必须是整数")
                result["step"] = int(step)
            if raw.get("dtype") is not None:
                result["dtype"] = raw["dtype"]
            return result

        if kind == "loguniform":
            result = {
                "type": "float",
                "low": float(np.exp(float(low))),
                "high": float(np.exp(float(high))),
                "log": True,
            }
            if not np.isfinite(result["low"]) or not np.isfinite(result["high"]) or result["low"] <= 0:
                raise ValueError(f"参数 {name!r} 的 loguniform 对数边界转换后必须是正有限数值")
            return result

        if kind == "qloguniform":
            return {
                "type": "qloguniform",
                "low": float(low),
                "high": float(high),
                "q": _positive_step(name, raw.get("q"), "q"),
            }

        if kind == "quniform":
            return {
                "type": "quniform",
                "low": float(low),
                "high": float(high),
                "q": _positive_step(name, raw.get("q"), "q"),
            }

        result = {"type": "float", "low": float(low), "high": float(high)}
        if kind == "float" and raw.get("log", False):
            if float(low) <= 0:
                raise ValueError(f"参数 {name!r} 使用 log=True 时下界必须大于 0")
            if raw.get("step") is not None:
                raise ValueError(f"参数 {name!r} 不能同时设置 log=True 和 step")
            result["log"] = True
        elif kind == "float" and raw.get("step") is not None:
            result["step"] = _positive_step(name, raw["step"], "step")
        if raw.get("dtype") is not None:
            result["dtype"] = raw["dtype"]
        return result

    if kind in {"normal", "qnormal", "lognormal", "qlognormal"}:
        if "mu" not in raw or "sigma" not in raw:
            raise ValueError(f"参数 {name!r} 的 {kind} 类型必须提供 'mu' 和 'sigma'")
        mu, sigma = float(raw["mu"]), float(raw["sigma"])
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"参数 {name!r} 的 sigma 必须是大于 0 的有限数值")
        is_log = kind in {"lognormal", "qlognormal"} or bool(raw.get("log", False))
        result = {
            "type": "normal",
            "mu": mu,
            "sigma": sigma,
            "low": float(np.exp(mu - _NORMAL_TAIL * sigma)) if is_log else mu - _NORMAL_TAIL * sigma,
            "high": float(np.exp(mu + _NORMAL_TAIL * sigma)) if is_log else mu + _NORMAL_TAIL * sigma,
        }
        if is_log:
            result["log"] = True
        if kind in {"qnormal", "qlognormal"} or raw.get("q") is not None:
            result["q"] = _positive_step(name, raw.get("q"), "q")
        return result

    raise ValueError(f"参数 {name!r} 的搜索空间类型未知或不支持: {kind!r}")


def _normalize_tuple(name: str, raw: tuple) -> Dict[str, Any]:
    if not raw:
        raise ValueError(f"参数 {name!r} 的元组搜索空间不能为空")
    if len(raw) == 3 and (_is_integer_type(raw[2]) or _is_float_type(raw[2])):
        low, high, dtype = raw
        _check_bounds(name, low, high)
        if _is_integer_type(dtype):
            if not float(low).is_integer() or not float(high).is_integer():
                raise ValueError(f"参数 {name!r} 指定 int 类型时上下界必须是整数")
            return {"type": "int", "low": int(low), "high": int(high)}
        return {"type": "float", "low": float(low), "high": float(high)}

    if len(raw) == 3 and isinstance(raw[2], str) and raw[2].strip().lower() in {"uniform", "log-uniform"}:
        low, high, prior = raw
        _check_bounds(name, low, high)
        result = {"type": "float", "low": float(low), "high": float(high)}
        if prior.strip().lower() == "log-uniform":
            if float(low) <= 0:
                raise ValueError(f"参数 {name!r} 的 log-uniform 下界必须大于 0")
            result["log"] = True
        return result

    if len(raw) == 2 and all(_is_number(value) for value in raw):
        low, high = raw
        _check_bounds(name, low, high)
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in raw):
            return {"type": "int", "low": int(low), "high": int(high)}
        return {"type": "float", "low": float(low), "high": float(high)}

    return _categorical_spec(name, raw)


def _normalize_scipy(name: str, raw: Any) -> Dict[str, Any]:
    dist_name = getattr(getattr(raw, "dist", None), "name", None)
    args = getattr(raw, "args", ())
    kwds = getattr(raw, "kwds", {}) or {}
    if dist_name == "randint":
        low = int(args[0] if args else kwds.get("low"))
        high_arg = int(args[1] if len(args) > 1 else kwds.get("high"))
        return {"type": "int", "low": low, "high": high_arg - 1}
    if dist_name in {"loguniform", "reciprocal"}:
        low = float(args[0] if args else kwds.get("a"))
        high = float(args[1] if len(args) > 1 else kwds.get("b"))
        return _normalize_dict(name, {"type": "float", "low": low, "high": high, "log": True})
    if dist_name == "uniform":
        loc = float(args[0] if args else kwds.get("loc", 0.0))
        scale = float(args[1] if len(args) > 1 else kwds.get("scale", 1.0))
        return _normalize_dict(name, {"type": "float", "low": loc, "high": loc + scale})
    raise ValueError(f"参数 {name!r} 的 scipy 分布暂不支持: {dist_name!r}")


def _normalize_optuna(name: str, raw: Any) -> Dict[str, Any]:
    distributions = optuna.distributions
    if isinstance(raw, distributions.IntDistribution):
        result: Dict[str, Any] = {"type": "int", "low": int(raw.low), "high": int(raw.high)}
        if raw.log:
            result["log"] = True
        elif raw.step != 1:
            result["step"] = int(raw.step)
        return result
    if isinstance(raw, distributions.FloatDistribution):
        result = {"type": "float", "low": float(raw.low), "high": float(raw.high)}
        if raw.log:
            result["log"] = True
        elif raw.step is not None:
            result["step"] = float(raw.step)
        return result
    if isinstance(raw, distributions.CategoricalDistribution):
        return _categorical_spec(name, raw.choices)
    raise ValueError(f"参数 {name!r} 的 Optuna 分布类型不支持: {type(raw).__name__}")


def normalize_space_param(name: str, raw: Any) -> Dict[str, Any]:
    """将一个参数声明转换为统一内部规格。"""
    declared_name = getattr(raw, "name", None)
    if declared_name is not None and declared_name != name:
        raise ValueError(f"搜索空间字典参数名 {name!r} 与声明中的参数名 {declared_name!r} 不一致")
    if hasattr(raw, "to_spec") and callable(raw.to_spec):
        return _normalize_dict(name, raw.to_spec())
    if optuna is not None and isinstance(raw, optuna.distributions.BaseDistribution):
        return _normalize_optuna(name, raw)
    if hasattr(raw, "dist") and callable(getattr(raw, "rvs", None)):
        return _normalize_scipy(name, raw)
    if isinstance(raw, tuple):
        return _normalize_tuple(name, raw)
    if isinstance(raw, list):
        return _categorical_spec(name, raw)
    if isinstance(raw, Mapping):
        return _normalize_dict(name, raw)
    raise ValueError(f"参数 {name!r} 的搜索空间定义无法识别: {raw!r}。支持字典、元组、列表、" "hscredit 同名维度、SciPy 分布或 Optuna 分布对象")


def normalize_search_space(search_space: Optional[Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """把支持框架的搜索空间声明统一为内部规格字典。"""
    if search_space is None:
        return None
    if isinstance(search_space, (list, tuple)):
        result: Dict[str, Dict[str, Any]] = {}
        for dimension in search_space:
            name = getattr(dimension, "name", None)
            if not name:
                raise ValueError("skopt Dimension 列表中的每个维度都必须设置 name")
            if name in result:
                raise ValueError(f"skopt Dimension 列表包含重复参数名: {name!r}")
            result[name] = normalize_space_param(name, dimension)
        return result
    if not isinstance(search_space, Mapping):
        raise ValueError("search_space 必须是参数字典或带 name 的 skopt Dimension 列表，" f"当前类型: {type(search_space).__name__}")
    return {str(name): normalize_space_param(str(name), raw) for name, raw in search_space.items()}


class SearchSpaceAdapter:
    """统一搜索空间的 Optuna 采样和手工点转换器。"""

    def __init__(self, search_space: Optional[Any]) -> None:
        self.space = normalize_search_space(search_space)

    @property
    def names(self) -> List[str]:
        return list((self.space or {}).keys())

    @staticmethod
    def latent_name(name: str) -> str:
        return f"{_LATENT_PREFIX}{name}"

    def to_internal_name(self, name: str) -> str:
        """把公开参数名转换为 Optuna Study 中实际记录的参数名。"""
        if name not in (self.space or {}):
            raise ValueError(f"搜索空间中不存在参数: {name!r}")
        spec = (self.space or {})[name]
        if spec["type"] in {"quniform", "qloguniform", "normal"} or (
            spec["type"] == "categorical" and "prior" in spec
        ):
            return self.latent_name(name)
        return name

    def to_public_name(self, name: str) -> str:
        """把 Optuna 内部潜变量名还原为公开参数名。"""
        if name.startswith(_LATENT_PREFIX):
            public_name = name[len(_LATENT_PREFIX) :]
            if public_name in (self.space or {}):
                return public_name
        return name

    @staticmethod
    def _cast(value: Any, spec: Mapping[str, Any]) -> Any:
        dtype = spec.get("dtype")
        if dtype is None:
            return value
        try:
            return dtype(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"参数值 {value!r} 无法转换为 dtype={dtype!r}") from exc

    @staticmethod
    def _quantize(value: float, q: float) -> float:
        return float(np.round(float(value) / q) * q)

    def sample_one(self, trial: Any, name: str, spec: Optional[Mapping[str, Any]] = None) -> Any:
        spec = dict(spec or (self.space or {})[name])
        kind = spec["type"]
        if kind == "int":
            value = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=1 if spec.get("log") else spec.get("step", 1),
                log=bool(spec.get("log", False)),
            )
            return self._cast(value, spec)
        if kind == "float":
            value = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=None if spec.get("log") else spec.get("step"),
                log=bool(spec.get("log", False)),
            )
            return self._cast(value, spec)
        if kind == "categorical" and "prior" not in spec:
            return trial.suggest_categorical(name, spec["choices"])

        latent = self.latent_name(name)
        if kind == "categorical":
            u = trial.suggest_float(latent, 0.0, 1.0)
            cumulative = np.cumsum(spec["prior"])
            index = min(int(np.searchsorted(cumulative, u, side="right")), len(spec["choices"]) - 1)
            return spec["choices"][index]
        if kind == "quniform":
            raw = trial.suggest_float(latent, spec["low"], spec["high"])
            return self._quantize(raw, spec["q"])
        if kind == "qloguniform":
            raw = trial.suggest_float(latent, spec["low"], spec["high"])
            return self._quantize(float(np.exp(raw)), spec["q"])
        if kind == "normal":
            lower_u, upper_u = norm.cdf(-_NORMAL_TAIL), norm.cdf(_NORMAL_TAIL)
            u = trial.suggest_float(latent, float(lower_u), float(upper_u))
            raw = float(norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)) * spec["sigma"] + spec["mu"])
            value = float(np.exp(raw)) if spec.get("log") else raw
            if spec.get("q") is not None:
                value = self._quantize(value, spec["q"])
            return value
        raise ValueError(f"未知参数类型: {kind}")

    def sample(self, trial: Any, skip: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        skipped = set(skip or ())
        return {
            name: self.sample_one(trial, name, spec) for name, spec in (self.space or {}).items() if name not in skipped
        }

    def public_params(self, params_or_trial: Any) -> Dict[str, Any]:
        raw_params = params_or_trial.params if hasattr(params_or_trial, "params") else params_or_trial
        result: Dict[str, Any] = {}
        for name, spec in (self.space or {}).items():
            latent = self.latent_name(name)
            if name in raw_params:
                result[name] = self._cast(raw_params[name], spec)
            elif latent in raw_params:
                result[name] = self._materialize_latent(raw_params[latent], spec)
        return result

    def _materialize_latent(self, latent: Any, spec: Mapping[str, Any]) -> Any:
        kind = spec["type"]
        if kind == "categorical":
            cumulative = np.cumsum(spec["prior"])
            index = min(int(np.searchsorted(cumulative, float(latent), side="right")), len(spec["choices"]) - 1)
            return spec["choices"][index]
        if kind == "quniform":
            return self._quantize(float(latent), spec["q"])
        if kind == "qloguniform":
            return self._quantize(float(np.exp(float(latent))), spec["q"])
        if kind == "normal":
            raw = float(norm.ppf(np.clip(float(latent), 1e-12, 1 - 1e-12)) * spec["sigma"] + spec["mu"])
            value = float(np.exp(raw)) if spec.get("log") else raw
            return self._quantize(value, spec["q"]) if spec.get("q") is not None else value
        raise ValueError(f"参数类型 {kind!r} 不使用潜变量")

    @staticmethod
    def _contains_choice(choices: Sequence[Any], value: Any) -> bool:
        return any(type(value) is type(choice) and value == choice for choice in choices)

    def _validate_public_value(self, name: str, value: Any, spec: Mapping[str, Any]) -> None:
        kind = spec["type"]
        if kind == "categorical":
            if not self._contains_choice(spec["choices"], value):
                raise ValueError(f"参数 {name!r} 的值 {value!r} 不在 choices 中")
            return
        if kind == "int":
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"参数 {name!r} 的手工搜索点必须是整数")
            if value < spec["low"] or value > spec["high"]:
                raise ValueError(f"参数 {name!r} 的值 {value!r} 超出搜索区间")
            step = spec.get("step", 1)
            if not spec.get("log") and (int(value) - spec["low"]) % step != 0:
                raise ValueError(f"参数 {name!r} 的值 {value!r} 不符合 step={step}")
            return
        if not _is_number(value) or not np.isfinite(float(value)):
            raise ValueError(f"参数 {name!r} 的手工搜索点必须是有限数值")
        if kind == "float" and (float(value) < spec["low"] or float(value) > spec["high"]):
            raise ValueError(f"参数 {name!r} 的值 {value!r} 超出搜索区间")
        if kind == "float" and spec.get("step") is not None:
            steps = (float(value) - spec["low"]) / spec["step"]
            if not np.isclose(steps, round(steps), atol=1e-8):
                raise ValueError(f"参数 {name!r} 的值 {value!r} 不符合 step={spec['step']}")
        if kind == "normal" and not spec.get("q"):
            if float(value) < spec["low"] or float(value) > spec["high"]:
                raise ValueError(f"参数 {name!r} 的值 {value!r} 超出截断搜索区间")
        if spec.get("q") is not None and not np.isclose(
            float(value), self._quantize(float(value), spec["q"]), atol=1e-8
        ):
            raise ValueError(f"参数 {name!r} 的值 {value!r} 不符合 q={spec['q']} 的量化规则")

    def _inverse_transformed(self, value: Any, spec: Mapping[str, Any]) -> float:
        kind = spec["type"]
        if kind == "categorical":
            index = next(
                index for index, choice in enumerate(spec["choices"]) if type(value) is type(choice) and value == choice
            )
            left = float(sum(spec["prior"][:index]))
            right = left + float(spec["prior"][index])
            if left == right:
                raise ValueError(f"权重为 0 的类别 {value!r} 无法作为手工搜索点入队")
            return (left + right) / 2.0
        if kind in {"quniform", "qloguniform"}:
            q = spec["q"]
            low_raw, high_raw = spec["low"], spec["high"]
            lower_value = float(value) - q / 2.0
            upper_value = float(value) + q / 2.0
            if kind == "qloguniform":
                lower_value = max(lower_value, np.finfo(float).tiny)
                left, right = math.log(lower_value), math.log(upper_value)
            else:
                left, right = lower_value, upper_value
            left, right = max(left, low_raw), min(right, high_raw)
            if left > right:
                raise ValueError(f"手工搜索点 {value!r} 无法由当前量化搜索空间生成")
            return (left + right) / 2.0
        if kind == "normal":
            numeric = float(value)
            if spec.get("q") is not None:
                half = spec["q"] / 2.0
                lower = max(numeric - half, spec["low"])
                upper = min(numeric + half, spec["high"])
                if spec.get("log"):
                    lower = max(lower, np.finfo(float).tiny)
                if lower > upper:
                    raise ValueError(f"手工搜索点 {value!r} 无法由当前正态量化空间生成")
                numeric = (lower + upper) / 2.0
            if spec.get("log"):
                if numeric <= 0:
                    raise ValueError("对数正态分布的手工搜索点必须大于 0")
                numeric = math.log(numeric)
            return float(norm.cdf((numeric - spec["mu"]) / spec["sigma"]))
        raise ValueError(f"参数类型 {kind!r} 不使用潜变量")

    def to_internal_point(self, point: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(point, Mapping):
            raise ValueError(f"手工搜索点必须是字典，收到: {type(point).__name__}")
        unknown = set(point) - set(self.names)
        if unknown:
            raise ValueError(f"手工搜索点包含搜索空间中不存在的参数: {sorted(unknown)}")
        result: Dict[str, Any] = {}
        for name, value in point.items():
            spec = (self.space or {})[name]
            self._validate_public_value(name, value, spec)
            if spec["type"] in {"quniform", "qloguniform", "normal"} or (
                spec["type"] == "categorical" and "prior" in spec
            ):
                result[self.latent_name(name)] = self._inverse_transformed(value, spec)
            else:
                result[name] = value
        return result
