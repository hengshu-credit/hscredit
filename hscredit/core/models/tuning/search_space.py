"""超参数搜索空间统一入口 - 跨框架同名符号.

本模块提供与主流超参数搜索库**同名**的符号，使用户无需单独安装或导入
optuna / skopt / hyperopt，只需::

    from hscredit.core.models.tuning.search_space import *

即可用原框架的代码风格定义 ``search_space``，直接传入 hscredit 的
``tune`` / ``ModelTuner`` / ``AutoTuner``，内部统一归一化为 optuna 风格 DSL。

各框架风格对应关系：

* **optuna** —— ``suggest_int`` / ``suggest_float`` / ``suggest_categorical`` /
  ``suggest_uniform`` / ``suggest_discrete_uniform`` / ``suggest_loguniform``
* **optuna 分布对象** —— ``IntDistribution`` / ``FloatDistribution`` /
  ``CategoricalDistribution``
* **scikit-optimize (skopt)** —— ``Real`` / ``Integer`` / ``Categorical`` /
  ``Dimension``
* **hyperopt** —— ``uniform`` / ``loguniform`` / ``quniform`` / ``qloguniform`` /
  ``choice`` / ``randint`` / ``normal`` / ``qnormal`` / ``lognormal`` / ``qlognormal``
* **GridSearchCV / bayesian-optimization** —— 直接用列表（离散候选）或元组（上下界），
  无需额外符号

注：``normal`` / ``lognormal`` / ``qnormal`` / ``qlognormal`` 因 optuna 无原生
正态采样，通过在截断区间均匀采样后逆 CDF 变换实现（截断区间取 [mu-4σ, mu+4σ]）。

**参考样例**

>>> from hscredit.core.models.tuning.search_space import *
>>> from hscredit.core.models import XGBoost
>>>
>>> # optuna / skopt / hyperopt / GridSearch 风格混用，无需关心底层归一化
>>> search_space = {
...     'max_depth': suggest_int('max_depth', 2, 6),            # optuna 风格
...     'learning_rate': Real(1e-3, 0.1, prior='log-uniform'),  # skopt 风格
...     'min_child_weight': Integer(1, 10),                     # skopt 风格
...     'booster': suggest_categorical('booster', ['gbtree', 'dart']),  # optuna 风格
...     'subsample': uniform('subsample', 0.6, 1.0),            # hyperopt 风格
...     'colsample_bytree': [0.6, 0.8, 1.0],                    # GridSearch 风格
...     'n_estimators': (50, 200),                              # bayesian-optimization 风格
... }
>>> best = XGBoost(random_state=42).tune(X_train, y_train, search_space=search_space)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

__all__ = [
    # 基类（skopt 风格）
    "Dimension",
    # skopt 风格
    "Real",
    "Integer",
    "Categorical",
    # optuna 分布对象风格
    "IntDistribution",
    "FloatDistribution",
    "CategoricalDistribution",
    # optuna suggest_* 风格
    "suggest_int",
    "suggest_float",
    "suggest_categorical",
    "suggest_uniform",
    "suggest_discrete_uniform",
    "suggest_loguniform",
    # hyperopt hp.* 风格
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "choice",
    "randint",
    "normal",
    "qnormal",
    "lognormal",
    "qlognormal",
]


class Dimension:
    """搜索空间维度基类（仿 ``skopt.space.Dimension``）。

    所有维度对象携带 ``name`` 与 ``to_spec`` 方法：``to_spec`` 返回内部 DSL 字典，
    由 :func:`normalize_search_space` 统一消费。用户通常不直接实例化本类，
    而是通过 ``Real`` / ``Integer`` / ``Categorical`` 等子类构造。

    **参数**

    :param name: 维度名，仅用于记录，可选；搜索空间字典的键即最终参数名

    **属性**

    - ``name``: 维度名
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def to_spec(self) -> dict:
        """返回内部 DSL 字典，供 :func:`normalize_search_space` 消费."""
        raise NotImplementedError

    def __repr__(self) -> str:
        spec = self.to_spec()
        fields = ", ".join(f"{k}={v!r}" for k, v in spec.items() if k != "type")
        return f"{type(self).__name__}({fields})"


class IntDistribution(Dimension):
    """整数维度（同名于 ``optuna.distributions.IntDistribution``）。

    **参数**

    :param low: 下界（含）
    :param high: 上界（含）
    :param log: 是否在对数尺度上采样，默认 False；与 ``step`` 互斥
    :param step: 采样步长，默认 1；仅当 ``log=False`` 生效
    :param name: 维度名，仅用于记录，可选

    **属性**

    - ``low`` / ``high`` / ``step`` / ``log`` / ``name``
    """

    def __init__(
        self,
        low: int,
        high: int,
        log: bool = False,
        step: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.low = int(low)
        self.high = int(high)
        self.step = int(step)
        self.log = bool(log)

    def to_spec(self) -> dict:
        spec: dict = {"type": "int", "low": self.low, "high": self.high}
        if self.log:
            spec["log"] = True
        elif self.step != 1:
            spec["step"] = self.step
        return spec


class FloatDistribution(Dimension):
    """浮点维度（同名于 ``optuna.distributions.FloatDistribution``）。

    **参数**

    :param low: 下界（含）
    :param high: 上界（含）
    :param log: 是否在对数尺度上采样，默认 False；与 ``step`` 互斥
    :param step: 采样步长，默认 None（连续采样）；与 ``log`` 互斥
    :param name: 维度名，仅用于记录，可选

    **属性**

    - ``low`` / ``high`` / ``step`` / ``log`` / ``name``
    """

    def __init__(
        self,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
        name: Optional[str] = None,
        _distribution: Optional[str] = None,
        _q: Optional[float] = None,
    ) -> None:
        super().__init__(name)
        self.low = float(low)
        self.high = float(high)
        self.step = float(step) if step is not None else None
        self.log = bool(log)
        self._distribution = _distribution
        self._q = float(_q) if _q is not None else None

    def to_spec(self) -> dict:
        if self._distribution is not None:
            spec = {"type": self._distribution, "low": self.low, "high": self.high}
            if self._q is not None:
                spec["q"] = self._q
            return spec
        spec: dict = {"type": "float", "low": self.low, "high": self.high}
        if self.log:
            spec["log"] = True
        if self.step is not None:
            spec["step"] = self.step
        return spec


class CategoricalDistribution(Dimension):
    """类别维度（同名于 ``optuna.distributions.CategoricalDistribution``）。

    **参数**

    :param choices: 候选值列表
    :param name: 维度名，仅用于记录，可选

    **属性**

    - ``choices`` / ``name``
    """

    def __init__(
        self,
        choices: Sequence[Any],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.choices = list(choices)

    def to_spec(self) -> dict:
        return {"type": "categorical", "choices": list(self.choices)}


class NormalDistribution(Dimension):
    """正态/对数正态维度（仿 hyperopt ``hp.normal`` / ``hp.lognormal``）。

    optuna 无原生正态采样，搜索时在截断区间 [mu-4σ, mu+4σ] 均匀采样后经逆
    CDF 变换为目标分布（``log=True`` 时在对数空间变换后取指数）。

    **参数**

    :param mu: 均值
    :param sigma: 标准差，需 > 0
    :param q: 量化步长，默认 None（不量化）
    :param log: 是否为对数正态，默认 False
    :param name: 维度名，仅用于记录，可选

    **属性**

    - ``mu`` / ``sigma`` / ``q`` / ``log`` / ``name``
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        q: Optional[float] = None,
        log: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)

    def to_spec(self) -> dict:
        # 按状态返回具体类型字符串，便于 _space_param_from_dict 的 param_type 分支识别
        if self.log and self.q is not None:
            param_type = 'qlognormal'
        elif self.log:
            param_type = 'lognormal'
        elif self.q is not None:
            param_type = 'qnormal'
        else:
            param_type = 'normal'
        lo = self.mu - 4 * self.sigma
        hi = self.mu + 4 * self.sigma
        low = float(np.exp(lo)) if self.log else float(lo)
        high = float(np.exp(hi)) if self.log else float(hi)
        spec: dict = {
            "type": param_type,
            "mu": self.mu,
            "sigma": self.sigma,
            "low": low,
            "high": high,
        }
        if self.q is not None:
            spec["q"] = self.q
        if self.log:
            spec["log"] = True
        return spec


class Integer(IntDistribution):
    """整数维度（仿 ``skopt.space.Integer``）。

    等价于 :class:`IntDistribution` 的无 step/log 形式，保留 ``Integer`` 名以便
    与 skopt 代码风格一致。

    **参数**

    :param low: 下界（含）
    :param high: 上界（含）
    :param name: 维度名，仅用于记录，可选
    """

    def __init__(
        self,
        low: int,
        high: int,
        prior: str = "uniform",
        base: int = 10,
        transform: Optional[str] = None,
        name: Optional[str] = None,
        dtype: Any = np.int64,
    ) -> None:
        prior_norm = str(prior).strip().lower()
        if prior_norm not in {"uniform", "log-uniform"}:
            raise ValueError("Integer 的 prior 仅支持 'uniform' 或 'log-uniform'")
        if transform not in {None, "identity", "normalize"}:
            raise ValueError("Integer 的 transform 仅支持 None、'identity' 或 'normalize'")
        if base <= 0 or base == 1:
            raise ValueError("Integer 的 base 必须大于 0 且不能等于 1")
        super().__init__(low, high, step=1, log=prior_norm == "log-uniform", name=name)
        self.prior = prior
        self.base = base
        self.transform = transform
        self.dtype = dtype

    def to_spec(self) -> dict:
        spec = super().to_spec()
        if self.dtype is not np.int64:
            spec["dtype"] = self.dtype
        return spec


class Real(FloatDistribution):
    """实数维度（仿 ``skopt.space.Real``）。

    **参数**

    :param low: 下界（含）
    :param high: 上界（含）
    :param prior: 采样先验，``'uniform'``（默认）或 ``'log-uniform'``（对数均匀）
    :param name: 维度名，仅用于记录，可选

    **属性**

    - ``low`` / ``high`` / ``prior`` / ``name``
    """

    def __init__(
        self,
        low: float,
        high: float,
        prior: str = "uniform",
        base: int = 10,
        transform: Optional[str] = None,
        name: Optional[str] = None,
        dtype: Any = float,
    ) -> None:
        prior_norm = str(prior).strip().lower()
        if prior_norm not in {"uniform", "log-uniform"}:
            raise ValueError("Real 的 prior 仅支持 'uniform' 或 'log-uniform'")
        if transform not in {None, "identity", "normalize"}:
            raise ValueError("Real 的 transform 仅支持 None、'identity' 或 'normalize'")
        if base <= 0 or base == 1:
            raise ValueError("Real 的 base 必须大于 0 且不能等于 1")
        log = prior_norm == "log-uniform"
        super().__init__(low, high, step=None, log=log, name=name)
        self.prior = prior
        self.base = base
        self.transform = transform
        self.dtype = dtype

    def to_spec(self) -> dict:
        spec: dict = {"type": "float", "low": self.low, "high": self.high}
        if self.log:
            spec["log"] = True
        if self.dtype is not float:
            spec["dtype"] = self.dtype
        return spec


class Categorical(CategoricalDistribution):
    """类别维度（仿 ``skopt.space.Categorical``）。

    **参数**

    :param categories: 候选值列表
    :param name: 维度名，仅用于记录，可选
    """

    def __init__(
        self,
        categories: Sequence[Any],
        prior: Optional[List[float]] = None,
        transform: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if transform not in {None, "identity", "onehot", "string", "label"}:
            raise ValueError(
                "Categorical 的 transform 仅支持 None、'identity'、'onehot'、'string' 或 'label'"
            )
        super().__init__(categories, name=name)
        self.prior = list(prior) if prior is not None else None
        self.transform = transform

    def to_spec(self) -> dict:
        spec = super().to_spec()
        if self.prior is not None:
            spec["prior"] = list(self.prior)
        return spec


# ---------------------------------------------------------------------------
# optuna suggest_* 风格（同名于 optuna Trial 的 suggest_* 方法）
# 与 optuna 不同的是此处为模块级函数，返回维度对象，供 search_space 字典使用
# ---------------------------------------------------------------------------


def suggest_int(
    name: str,
    low: int,
    high: int,
    step: int = 1,
    log: bool = False,
) -> IntDistribution:
    """整数参数（同名于 ``optuna.Trial.suggest_int``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :param step: 采样步长，默认 1
    :param log: 是否对数尺度采样，默认 False
    :return: 整数维度对象
    """
    return IntDistribution(low, high, step=step, log=log, name=name)


def suggest_float(
    name: str,
    low: float,
    high: float,
    step: Optional[float] = None,
    log: bool = False,
) -> FloatDistribution:
    """浮点参数（同名于 ``optuna.Trial.suggest_float``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :param step: 采样步长，默认 None（连续）
    :param log: 是否对数尺度采样，默认 False
    :return: 浮点维度对象
    """
    return FloatDistribution(low, high, step=step, log=log, name=name)


def suggest_categorical(name: str, choices: Sequence[Any]) -> CategoricalDistribution:
    """类别参数（同名于 ``optuna.Trial.suggest_categorical``）。

    :param name: 参数名
    :param choices: 候选值列表
    :return: 类别维度对象
    """
    return CategoricalDistribution(choices, name=name)


def suggest_discrete_uniform(name: str, low: float, high: float, q: float) -> FloatDistribution:
    """等间距离散浮点参数（同名于 optuna 已弃用的 ``suggest_discrete_uniform``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :param q: 步长
    :return: 浮点维度对象（带 step）
    """
    return FloatDistribution(low, high, step=q, log=False, name=name)


def suggest_uniform(name: str, low: float, high: float) -> FloatDistribution:
    """均匀浮点参数（同名于 Optuna 已弃用的 ``suggest_uniform``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :return: 连续浮点维度对象
    """
    return FloatDistribution(low, high, log=False, step=None, name=name)


def suggest_loguniform(name: str, low: float, high: float) -> FloatDistribution:
    """对数均匀浮点参数（同名于 optuna 已弃用的 ``suggest_loguniform``）。

    :param name: 参数名
    :param low: 下界（含，需 > 0）
    :param high: 上界（含）
    :return: 浮点维度对象（log=True）
    """
    return FloatDistribution(low, high, step=None, log=True, name=name)


# ---------------------------------------------------------------------------
# hyperopt hp.* 风格（同名于 hyperopt.hp 的函数）
# 与 hyperopt 不同的是此处返回维度对象，供 search_space 字典使用
# ---------------------------------------------------------------------------


def uniform(name: str, low: float, high: float) -> FloatDistribution:
    """均匀分布浮点参数（同名于 ``hyperopt.hp.uniform``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :return: 浮点维度对象
    """
    return FloatDistribution(low, high, name=name, _distribution="uniform")


def loguniform(name: str, low: float, high: float) -> FloatDistribution:
    """对数均匀分布浮点参数（同名于 ``hyperopt.hp.loguniform``）。

    :param name: 参数名
    :param low: 对数空间下界（含）
    :param high: 对数空间上界（含）
    :return: 浮点维度对象（log=True）
    """
    return FloatDistribution(low, high, name=name, _distribution="loguniform")


def quniform(name: str, low: float, high: float, q: float) -> FloatDistribution:
    """量化均匀分布浮点参数（同名于 ``hyperopt.hp.quniform``）。

    :param name: 参数名
    :param low: 下界（含）
    :param high: 上界（含）
    :param q: 量化步长
    :return: 浮点维度对象（带 step）
    """
    return FloatDistribution(low, high, name=name, _distribution="quniform", _q=q)


def qloguniform(name: str, low: float, high: float, q: float) -> FloatDistribution:
    """量化对数均匀分布浮点参数（同名于 ``hyperopt.hp.qloguniform``）。

    :param name: 参数名
    :param low: 对数空间下界（含）
    :param high: 对数空间上界（含）
    :param q: 最终值空间的量化步长
    :return: 量化对数均匀维度对象
    """
    return FloatDistribution(low, high, name=name, _distribution="qloguniform", _q=q)


def choice(name: str, options: Sequence[Any]) -> CategoricalDistribution:
    """类别参数（同名于 ``hyperopt.hp.choice``）。

    :param name: 参数名
    :param options: 候选值列表
    :return: 类别维度对象
    """
    return CategoricalDistribution(options, name=name)


def randint(
    name: str,
    *args: int,
    low: Optional[int] = None,
    upper: Optional[int] = None,
) -> IntDistribution:
    """整数参数（同名于 ``hyperopt.hp.randint``）。

    hyperopt ``hp.randint(label, upper)`` 返回 [0, upper) 内整数；本实现支持
    可选 ``low`` 参数扩展为 [low, upper)。

    :param name: 参数名
    :param upper: 上界（不含）
    :param low: 下界（含），默认 0
    :return: 整数维度对象
    """
    if upper is not None:
        if args:
            raise TypeError("randint 同时使用位置参数和 upper 关键字时含义不明确")
        lower = 0 if low is None else int(low)
        upper_value = int(upper)
    elif len(args) == 1:
        upper_value = int(args[0])
        lower = 0 if low is None else int(low)
    elif len(args) == 2 and low is None:
        lower, upper_value = map(int, args)
    else:
        raise TypeError("randint 用法为 randint(name, upper) 或 randint(name, low, upper)")
    high = upper_value - 1
    if high < lower:
        raise ValueError(f"参数 {name!r} 的 randint 上界({upper_value})需大于下界({lower})")
    return IntDistribution(lower, high, step=1, log=False, name=name)


def normal(name: str, mu: float, sigma: float) -> NormalDistribution:
    """正态分布参数（同名于 ``hyperopt.hp.normal``）。

    :param name: 参数名
    :param mu: 均值
    :param sigma: 标准差（> 0）
    :return: 正态维度对象
    """
    return NormalDistribution(mu, sigma, q=None, log=False, name=name)


def qnormal(name: str, mu: float, sigma: float, q: float) -> NormalDistribution:
    """量化正态分布参数（同名于 ``hyperopt.hp.qnormal``）。

    :param name: 参数名
    :param mu: 均值
    :param sigma: 标准差（> 0）
    :param q: 量化步长
    :return: 正态维度对象（带 q）
    """
    return NormalDistribution(mu, sigma, q=q, log=False, name=name)


def lognormal(name: str, mu: float, sigma: float) -> NormalDistribution:
    """对数正态分布参数（同名于 ``hyperopt.hp.lognormal``）。

    :param name: 参数名
    :param mu: 对数空间均值
    :param sigma: 对数空间标准差（> 0）
    :return: 正态维度对象（log=True）
    """
    return NormalDistribution(mu, sigma, q=None, log=True, name=name)


def qlognormal(name: str, mu: float, sigma: float, q: float) -> NormalDistribution:
    """量化对数正态分布参数（同名于 ``hyperopt.hp.qlognormal``）。

    :param name: 参数名
    :param mu: 对数空间均值
    :param sigma: 对数空间标准差（> 0）
    :param q: 量化步长
    :return: 正态维度对象（log=True，带 q）
    """
    return NormalDistribution(mu, sigma, q=q, log=True, name=name)
