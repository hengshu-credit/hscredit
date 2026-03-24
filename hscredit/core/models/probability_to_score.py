"""概率转评分模块.

提供多种将模型预测概率转换为评分的方法，支持信用评分和欺诈评分两种场景。

**核心功能**

- 标准评分卡方法: 基于log-odds的标准信用评分转换
- 线性映射方法: 简单直接的线性转换
- 分位数映射: 基于数据分布的分位数映射
- 多种方向模式: 支持"递增"和"递减"两种评分方向
- 边界截断: 自动处理超出范围的评分

**评分转换原理**

**1. 标准评分卡方法**

基于log-odds的标准转换公式:
    Score = A - B × ln(odds)
    where: odds = P / (1 - P)

参数A和B通过以下业务参数确定:
- 基准好坏比(base_odds): 如1:50(坏:好)
- 基准分数(base_score): 对应基准好坏比的分数，如600分
- PDO(Point of Double Odds): odds翻倍时分数变化量，如20分

**2. 线性映射方法**

将概率线性映射到评分区间:
    - ascending(递增，欺诈分): 概率越高，分数越高
    - descending(递减，信用分): 概率越高，分数越低

**3. 分位数映射**

根据概率在数据中的分位数映射到评分。

**评分方向**

| 模式 | 说明 | 典型应用 |
|------|------|----------|
| descending | 概率越高分数越低 | 信用评分(如300-1000) |
| ascending | 概率越高分数越高 | 欺诈评分(如0-100) |
| auto | 自动判断 | 根据上下边界自动推断 |

**依赖**
- numpy
- pandas

**示例**

>>> from hscredit.core.models import XGBoostRiskModel
>>> from hscredit.core.models.probability_to_score import ScoreTransformer
>>>
>>> # 训练基础模型
>>> model = XGBoostRiskModel()
>>> model.fit(X_train, y_train)
>>>
>>> # 信用评分(概率越高分越低): 300-1000分
>>> transformer = ScoreTransformer(
...     method='standard',
...     lower=300,
...     upper=1000,
...     direction='descending',
...     base_odds=0.02,
...     base_score=600,
...     pdo=20,
...     precision=0
... )
>>> transformer.fit(model, X_train)
>>> credit_scores = transformer.predict_score(X_test)
>>>
>>> # 欺诈评分(概率越高分越高): 0-100分
>>> fraud_transformer = ScoreTransformer(
...     method='linear',
...     lower=0,
...     upper=100,
...     direction='ascending',
...     precision=0
... )
>>> fraud_transformer.fit(model, X_train)
>>> fraud_scores = fraud_transformer.predict_score(X_test)
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseScoreTransformer(BaseEstimator, ABC):
    """评分转换器基类.

    所有评分转换器的抽象基类，定义统一接口。

    **参数**

    :param lower: 评分下界，默认None(不限制)
    :param upper: 评分上界，默认None(不限制)
    :param direction: 评分方向，默认'auto'
        - 'descending': 概率越高分数越低(信用分)
        - 'ascending': 概率越高分数越高(欺诈分)
        - 'auto': 根据lower/upper自动判断
    :param precision: 评分精度(小数位数)，默认0(整数)
    :param clip: 是否对超出范围的评分进行截断，默认True
    """

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: Literal['descending', 'ascending', 'auto'] = 'auto',
        precision: int = 0,
        clip: bool = True
    ):
        self.lower = lower
        self.upper = upper
        self.direction = direction
        self.precision = precision
        self.clip = clip
        self._is_fitted = False

    def _determine_direction(self) -> str:
        """确定评分方向.

        :return: 'descending' 或 'ascending'
        """
        if self.direction != 'auto':
            return self.direction

        # 根据常见的分数范围自动判断
        if self.lower is not None and self.upper is not None:
            score_range = self.upper - self.lower
            # 信用评分通常范围较大(如300-1000)
            if score_range >= 500:
                return 'descending'  # 信用分：概率越低，分数越高
            # 欺诈评分通常范围较小(如0-100)
            if score_range <= 100:
                return 'ascending'   # 欺诈分：概率越高，分数越高

        # 默认: descending (信用分模式)
        return 'descending'

    def _clip_scores(self, scores: np.ndarray) -> np.ndarray:
        """截断超出范围的评分.

        :param scores: 原始评分
        :return: 截断后的评分
        """
        if not self.clip:
            return scores

        lower = self.lower if self.lower is not None else -np.inf
        upper = self.upper if self.upper is not None else np.inf
        return np.clip(scores, lower, upper)

    def _round_scores(self, scores: np.ndarray) -> np.ndarray:
        """四舍五入评分.

        :param scores: 原始评分
        :return: 四舍五入后的评分
        """
        return np.round(scores, self.precision)

    @abstractmethod
    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> 'BaseScoreTransformer':
        """拟合评分转换器.

        :param model: 已训练的分类模型，需有predict_proba方法
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :return: self
        """
        pass

    @abstractmethod
    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分.

        :param proba: 预测概率(正类概率)
        :return: 评分
        """
        pass

    def predict_score(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """预测评分.

        可通过传入X或proba之一来获取评分。

        :param X: 特征矩阵，用于预测概率
        :param proba: 直接传入预测概率
        :return: 评分数组

        **示例**

        >>> # 通过特征矩阵预测
        >>> scores = transformer.predict_score(X_test)

        >>> # 通过概率直接转换
        >>> proba = model.predict_proba(X_test)[:, 1]
        >>> scores = transformer.predict_score(proba=proba)
        """
        check_is_fitted(self)

        if proba is None:
            if X is None:
                raise ValueError("必须提供X或proba参数之一")
            if not hasattr(self, 'model_'):
                raise ValueError("未找到模型，请先调用fit()")
            proba = self.model_.predict_proba(X)[:, 1]

        proba = np.asarray(proba)

        # 确保概率在有效范围内
        proba = np.clip(proba, 1e-10, 1 - 1e-10)

        # 转换为评分
        scores = self.transform(proba)

        # 截断
        scores = self._clip_scores(scores)

        # 四舍五入
        scores = self._round_scores(scores)

        return scores

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率.

        注意: 这是一个近似转换，可能不完全准确。

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        raise NotImplementedError("此方法需要在子类中实现")

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备数据，支持两种传参风格.

        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名
        :return: (X, y) 处理后的数据
        """
        # scorecardpipeline风格：从X中提取target
        if y is None:
            if isinstance(X, pd.DataFrame) and target in X.columns:
                y = X[target].values
                X = X.drop(columns=[target])
            else:
                raise ValueError(f"y为None时，X必须是包含'{target}'列的DataFrame")
        else:
            if isinstance(y, pd.Series):
                y = y.values

        if isinstance(X, pd.DataFrame):
            X = X.values

        return X, y


class StandardScoreTransformer(BaseScoreTransformer):
    """标准评分卡转换器.

    基于log-odds的标准信用评分转换方法。
    公式: Score = A - B × ln(odds), 其中 odds = P / (1 - P)

    **参数**

    :param lower: 评分下界，默认None
    :param upper: 评分上界，默认None
    :param direction: 评分方向，默认'descending'(信用分模式)
    :param base_odds: 基准好坏比，默认0.05(5%坏样本率)
        - 表示在base_score对应的坏样本率
    :param base_score: 基准分数，默认600
        - 对应base_odds的分数
    :param pdo: Point of Double Odds，默认20
        - 当odds增加rate倍时，分数的变化量
    :param rate: 倍率，默认2
        - odds增加的倍数
    :param precision: 评分精度，默认0
    :param clip: 是否截断，默认True

    **示例**

    >>> transformer = StandardScoreTransformer(
    ...     lower=300,
    ...     upper=1000,
    ...     direction='descending',  # 信用分
    ...     base_odds=0.02,
    ...     base_score=600,
    ...     pdo=20,
    ...     rate=2
    ... )
    >>> transformer.fit(model, X_train)
    >>> scores = transformer.predict_score(X_test)
    """

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: Literal['descending', 'ascending', 'auto'] = 'descending',
        base_odds: float = 0.05,
        base_score: float = 600,
        pdo: float = 20,
        rate: float = 2,
        precision: int = 0,
        clip: bool = True
    ):
        super().__init__(lower, upper, direction, precision, clip)
        self.base_odds = base_odds
        self.base_score = base_score
        self.pdo = pdo
        self.rate = rate

    def _compute_parameters(self) -> Tuple[float, float]:
        """计算评分公式中的参数A和B.

        根据以下两个方程求解:
        1. base_score = A - B × ln(base_odds)
        2. base_score + pdo = A - B × ln(rate × base_odds)

        解得:
        B = pdo / ln(rate)
        A = base_score + B × ln(base_odds)

        :return: (A, B)
        """
        B = self.pdo / np.log(self.rate)
        A = self.base_score + B * np.log(self.base_odds)
        return A, B

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'StandardScoreTransformer':
        """拟合评分转换器.

        主要保存模型引用，参数在初始化时已确定。

        :param model: 已训练的分类模型
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名，默认'target'
        :return: self
        """
        # 准备数据(用于验证，实际参数已预设)
        X_prep, y_prep = self._prepare_data(X, y, target)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 计算参数
        self.A_, self.B_ = self._compute_parameters()

        # 确定方向
        self.direction_ = self._determine_direction()

        # 验证参数合理性
        self._validate_parameters()

        self._is_fitted = True

        return self

    def _validate_parameters(self):
        """验证参数合理性."""
        # 计算极端概率对应的分数
        test_probs = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
        scores = [self._transform_single(p) for p in test_probs]

        # 检查方向是否与预期一致
        if self.direction_ == 'descending':
            # 递减：低概率应该对应高分
            if scores[0] < scores[-1]:
                warnings.warn(
                    f"评分方向可能与预期不符。低概率对应{scores[0]:.1f}分，"
                    f"高概率对应{scores[-1]:.1f}分。对于信用分(descending)，"
                    f"低概率(好客户)应该高分。"
                )
        else:
            # 递增：高概率应该对应高分(欺诈分)
            if scores[0] > scores[-1]:
                warnings.warn(
                    f"评分方向可能与预期不符。低概率对应{scores[0]:.1f}分，"
                    f"高概率对应{scores[-1]:.1f}分。对于欺诈分(ascending)，"
                    f"高概率(坏客户)应该高分。"
                )

    def _transform_single(self, proba: float) -> float:
        """转换单个概率.

        :param proba: 单个概率值
        :return: 评分
        """
        odds = proba / (1 - proba)
        score = self.A_ - self.B_ * np.log(odds)
        return score

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分.

        :param proba: 预测概率
        :return: 评分数组
        """
        check_is_fitted(self)
        proba = np.asarray(proba)

        # 计算odds
        odds = proba / (1 - proba)

        # 标准评分卡公式
        scores = self.A_ - self.B_ * np.log(odds)

        # 如果方向是ascending(欺诈分)，反转分数
        if self.direction_ == 'ascending':
            lower = self.lower if self.lower is not None else scores.min()
            upper = self.upper if self.upper is not None else scores.max()
            scores = upper - (scores - lower)

        return scores

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率.

        公式推导:
        Score = A - B × ln(odds)
        => ln(odds) = (A - Score) / B
        => odds = exp((A - Score) / B)
        => P = odds / (1 + odds)

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        # 如果方向是ascending(欺诈分)，先反转分数
        if self.direction_ == 'ascending':
            lower = self.lower if self.lower is not None else scores.min()
            upper = self.upper if self.upper is not None else scores.max()
            scores = upper - (scores - lower)

        # 反向计算
        odds = np.exp((self.A_ - scores) / self.B_)
        proba = odds / (1 + odds)

        return proba


class LinearScoreTransformer(BaseScoreTransformer):
    """线性评分转换器.

    将概率线性映射到评分区间。
    简单直接，适用于快速实现。

    **参数**

    :param lower: 评分下界，默认0
    :param upper: 评分上界，默认100
    :param direction: 评分方向，默认'ascending'(欺诈分模式)
    :param precision: 评分精度，默认0
    :param clip: 是否截断，默认True

    **转换公式**

    - descending(信用分): Score = upper - (upper - lower) × P
    - ascending(欺诈分): Score = lower + (upper - lower) × P

    **示例**

    >>> # 欺诈分(0-100, 越大越危险)
    >>> transformer = LinearScoreTransformer(
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending'
    ... )
    >>> transformer.fit(model, X_train)
    >>> scores = transformer.predict_score(X_test)
    """

    def __init__(
        self,
        lower: Optional[float] = 0,
        upper: Optional[float] = 100,
        direction: Literal['descending', 'ascending', 'auto'] = 'ascending',
        precision: int = 0,
        clip: bool = True
    ):
        super().__init__(lower, upper, direction, precision, clip)

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'LinearScoreTransformer':
        """拟合评分转换器.

        :param model: 已训练的分类模型
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名，默认'target'
        :return: self
        """
        # 准备数据
        X_prep, y_prep = self._prepare_data(X, y, target)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 确定方向
        self.direction_ = self._determine_direction()

        self._is_fitted = True

        return self

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分.

        :param proba: 预测概率
        :return: 评分数组
        """
        check_is_fitted(self)
        proba = np.asarray(proba)

        lower = self.lower if self.lower is not None else 0
        upper = self.upper if self.upper is not None else 100

        if self.direction_ == 'descending':
            # 信用分: 低概率->高分
            scores = upper - (upper - lower) * proba
        else:
            # 欺诈分: 高概率->高分
            scores = lower + (upper - lower) * proba

        return scores

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率.

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        lower = self.lower if self.lower is not None else 0
        upper = self.upper if self.upper is not None else 100

        if self.direction_ == 'descending':
            # 反向: 高分->低概率
            proba = (upper - scores) / (upper - lower)
        else:
            # 反向: 高分->高概率
            proba = (scores - lower) / (upper - lower)

        return np.clip(proba, 0, 1)


class QuantileScoreTransformer(BaseScoreTransformer):
    """分位数评分转换器.

    根据概率在训练数据中的分位数映射到评分。
    适用于需要保持相对排名的场景。

    **参数**

    :param lower: 评分下界，默认0
    :param upper: 评分上界，默认100
    :param direction: 评分方向，默认'ascending'(欺诈分模式)
    :param n_quantiles: 分位数数量，默认100
    :param precision: 评分精度，默认0
    :param clip: 是否截断，默认True

    **转换原理**

    1. 在训练数据上计算概率分布
    2. 将概率映射到对应的分位数
    3. 将分位数线性映射到评分区间

    **示例**

    >>> transformer = QuantileScoreTransformer(
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending',
    ...     n_quantiles=100
    ... )
    >>> transformer.fit(model, X_train, y_train)
    >>> scores = transformer.predict_score(X_test)
    """

    def __init__(
        self,
        lower: Optional[float] = 0,
        upper: Optional[float] = 100,
        direction: Literal['descending', 'ascending', 'auto'] = 'ascending',
        n_quantiles: int = 100,
        precision: int = 0,
        clip: bool = True
    ):
        super().__init__(lower, upper, direction, precision, clip)
        self.n_quantiles = n_quantiles

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'QuantileScoreTransformer':
        """拟合评分转换器.

        学习训练数据的概率分布，用于后续分位数映射。

        :param model: 已训练的分类模型
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名，默认'target'
        :return: self
        """
        # 准备数据
        X_prep, y_prep = self._prepare_data(X, y, target)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 计算训练集概率
        self.train_proba_ = model.predict_proba(X_prep)[:, 1]

        # 保存分位数
        quantiles = np.linspace(0, 1, self.n_quantiles + 1)
        self.quantile_values_ = np.quantile(self.train_proba_, quantiles)

        # 确定方向
        self.direction_ = self._determine_direction()

        self._is_fitted = True

        return self

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分.

        :param proba: 预测概率
        :return: 评分数组
        """
        check_is_fitted(self)
        proba = np.asarray(proba)

        lower = self.lower if self.lower is not None else 0
        upper = self.upper if self.upper is not None else 100

        # 计算分位数排名
        # 使用搜索找到每个概率对应的分位数
        quantile_ranks = np.searchsorted(self.quantile_values_, proba, side='right') - 1
        quantile_ranks = np.clip(quantile_ranks, 0, self.n_quantiles - 1)

        # 将分位数映射到评分
        if self.direction_ == 'descending':
            # 信用分: 低分位数(低概率)->高分
            scores = upper - (upper - lower) * quantile_ranks / (self.n_quantiles - 1)
        else:
            # 欺诈分: 高分位数(高概率)->高分
            scores = lower + (upper - lower) * quantile_ranks / (self.n_quantiles - 1)

        return scores

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率(近似值).

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        lower = self.lower if self.lower is not None else 0
        upper = self.upper if self.upper is not None else 100

        if self.direction_ == 'descending':
            # 反向计算分位数排名
            quantile_ranks = (upper - scores) / (upper - lower) * (self.n_quantiles - 1)
        else:
            quantile_ranks = (scores - lower) / (upper - lower) * (self.n_quantiles - 1)

        quantile_ranks = np.clip(quantile_ranks, 0, self.n_quantiles - 1).astype(int)

        # 使用中位数作为估计概率
        proba = self.quantile_values_[quantile_ranks + 1]

        return proba


class ScoreTransformer(BaseScoreTransformer):
    """统一评分转换器接口.

    提供统一的接口，支持多种转换方法。

    **参数**

    :param method: 转换方法，默认'standard'
        - 'standard': 标准评分卡方法(StandardScoreTransformer)
        - 'linear': 线性映射(LinearScoreTransformer)
        - 'quantile': 分位数映射(QuantileScoreTransformer)
    :param lower: 评分下界，默认None
    :param upper: 评分上界，默认None
    :param direction: 评分方向，默认'auto'
    :param precision: 评分精度，默认0
    :param clip: 是否截断，默认True
    :param target: 目标列名，默认'target'
        - 用于从DataFrame中提取目标变量
    :param kwargs: 传递给具体转换器的参数
        - standard方法: base_odds, base_score, pdo
        - quantile方法: n_quantiles

    **常用配置**

    **信用评分(概率越高分越低)**
    >>> transformer = ScoreTransformer(
    ...     method='standard',
    ...     lower=300,
    ...     upper=1000,
    ...     direction='descending',
    ...     base_odds=0.02,
    ...     base_score=600,
    ...     pdo=20
    ... )

    **欺诈评分(概率越高分越高)**
    >>> transformer = ScoreTransformer(
    ...     method='linear',
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending'
    ... )

    **示例**

    **sklearn风格**
    >>> transformer = ScoreTransformer(method='standard')
    >>> transformer.fit(model, X_train, y_train)
    >>> scores = transformer.predict_score(X_test)

    **scorecardpipeline风格**
    >>> transformer = ScoreTransformer(method='standard')
    >>> transformer.fit(model, df_train)  # df_train包含target列
    >>> scores = transformer.predict_score(df_test)
    """

    def __init__(
        self,
        method: Literal['standard', 'linear', 'quantile'] = 'standard',
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: Literal['descending', 'ascending', 'auto'] = 'auto',
        precision: int = 0,
        clip: bool = True,
        target: str = 'target',
        **kwargs
    ):
        super().__init__(lower, upper, direction, precision, clip)
        self.method = method
        self.target = target
        self.transformer_params = kwargs

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: Optional[str] = None,
        **kwargs
    ) -> 'ScoreTransformer':
        """拟合评分转换器.

        支持两种传参风格:

        **sklearn风格**::
            transformer.fit(model, X_train, y_train)

        **scorecardpipeline风格**::
            transformer.fit(model, df_train)  # df_train包含target列

        :param model: 已训练的分类模型
        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名，默认使用初始化时设置的target
        :return: self
        """
        # 使用初始化时设置的target
        target = target or self.target

        # 创建具体的转换器
        if self.method == 'standard':
            self.transformer_ = StandardScoreTransformer(
                lower=self.lower,
                upper=self.upper,
                direction=self.direction,
                precision=self.precision,
                clip=self.clip,
                **self.transformer_params
            )
        elif self.method == 'linear':
            self.transformer_ = LinearScoreTransformer(
                lower=self.lower,
                upper=self.upper,
                direction=self.direction,
                precision=self.precision,
                clip=self.clip
            )
        elif self.method == 'quantile':
            self.transformer_ = QuantileScoreTransformer(
                lower=self.lower,
                upper=self.upper,
                direction=self.direction,
                precision=self.precision,
                clip=self.clip,
                **self.transformer_params
            )
        else:
            raise ValueError(f"不支持的转换方法: {self.method}")

        # 拟合具体转换器
        self.transformer_.fit(model, X, y, target=target, **kwargs)

        # 复制重要属性
        self.model_ = self.transformer_.model_
        self.direction_ = self.transformer_.direction_

        self._is_fitted = True

        return self

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分.

        :param proba: 预测概率
        :return: 评分数组
        """
        check_is_fitted(self)
        return self.transformer_.transform(proba)

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率.

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        return self.transformer_.inverse_transform(scores)


def transform_probability_to_score(
    proba: Union[np.ndarray, pd.Series],
    method: str = 'standard',
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    direction: str = 'descending',
    precision: int = 0,
    **kwargs
) -> np.ndarray:
    """便捷函数: 将概率转换为评分.

    无需创建转换器对象，直接进行转换。
    注意: 此函数不进行拟合，直接使用预设参数。

    :param proba: 预测概率
    :param method: 转换方法，默认'standard'
    :param lower: 评分下界，默认None
    :param upper: 评分上界，默认None
    :param direction: 评分方向，默认'descending'
    :param precision: 评分精度，默认0
    :param kwargs: 其他参数
    :return: 评分数组

    **示例**

    >>> scores = transform_probability_to_score(
    ...     proba,
    ...     method='linear',
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending'
    ... )
    """
    proba = np.asarray(proba)
    proba = np.clip(proba, 1e-10, 1 - 1e-10)

    if method == 'standard':
        # 标准评分卡方法
        base_odds = kwargs.get('base_odds', 0.05)
        base_score = kwargs.get('base_score', 600)
        pdo = kwargs.get('pdo', 20)

        B = pdo / np.log(2)
        A = base_score + B * np.log(base_odds)

        odds = proba / (1 - proba)
        scores = A - B * np.log(odds)

        if direction == 'ascending':
            lower_val = lower if lower is not None else scores.min()
            upper_val = upper if upper is not None else scores.max()
            scores = upper_val - (scores - lower_val)

    elif method == 'linear':
        # 线性映射
        lower_val = lower if lower is not None else 0
        upper_val = upper if upper is not None else 100

        if direction == 'descending':
            scores = upper_val - (upper_val - lower_val) * proba
        else:
            scores = lower_val + (upper_val - lower_val) * proba

    else:
        raise ValueError(f"不支持的转换方法: {method}")

    # 截断
    if lower is not None or upper is not None:
        lower_val = lower if lower is not None else -np.inf
        upper_val = upper if upper is not None else np.inf
        scores = np.clip(scores, lower_val, upper_val)

    # 四舍五入
    scores = np.round(scores, precision)

    return scores


# 尝试导入matplotlib进行绘图
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_score_transformation_curve(
    transformer: BaseScoreTransformer,
    proba_range: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    show: bool = True
) -> Any:
    """绘制概率-评分转换曲线.

    :param transformer: 已拟合的评分转换器
    :param proba_range: 概率范围，默认0-1的100个点
    :param figsize: 图表大小，默认(10, 6)
    :param title: 图表标题，可选
    :param show: 是否显示图表，默认True
    :return: matplotlib Figure对象

    **示例**

    >>> transformer = ScoreTransformer(method='standard')
    >>> transformer.fit(model, X_train)
    >>> fig = plot_score_transformation_curve(transformer)
    >>> fig.savefig('transformation_curve.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    check_is_fitted(transformer)

    if proba_range is None:
        proba_range = np.linspace(0.001, 0.999, 100)

    # 计算对应的评分
    scores = transformer.transform(proba_range)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(proba_range, scores, 'b-', linewidth=2, label='转换曲线')

    # 添加参考线
    if transformer.lower is not None:
        ax.axhline(y=transformer.lower, color='r', linestyle='--', alpha=0.5, label=f'下界={transformer.lower}')
    if transformer.upper is not None:
        ax.axhline(y=transformer.upper, color='r', linestyle='--', alpha=0.5, label=f'上界={transformer.upper}')

    # 设置标签
    direction_text = "递增(ascending)" if transformer.direction_ == 'ascending' else "递减(descending)"
    ax.set_xlabel('预测概率 (P)', fontsize=12)
    ax.set_ylabel('评分', fontsize=12)

    if title is None:
        title = f'概率-评分转换曲线 ({direction_text})'
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def compare_score_transformers(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    methods: List[str] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    direction: str = 'descending',
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True
) -> Any:
    """对比多种评分转换方法.

    :param model: 已训练的分类模型
    :param X: 特征矩阵
    :param methods: 要对比的方法列表，默认['standard', 'linear', 'quantile']
    :param lower: 评分下界，默认None
    :param upper: 评分上界，默认None
    :param direction: 评分方向，默认'descending'
    :param figsize: 图表大小，默认(12, 5)
    :param show: 是否显示图表，默认True
    :return: matplotlib Figure对象

    **示例**

    >>> fig = compare_score_transformers(model, X_test)
    >>> fig.savefig('transformer_comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    if methods is None:
        methods = ['standard', 'linear', 'quantile']

    # 获取概率
    proba = model.predict_proba(X)[:, 1]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图: 转换曲线对比
    ax1 = axes[0]
    proba_range = np.linspace(0.001, 0.999, 100)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        try:
            transformer = ScoreTransformer(
                method=method,
                lower=lower,
                upper=upper,
                direction=direction
            )
            transformer.fit(model, X)
            scores = transformer.transform(proba_range)
            ax1.plot(proba_range, scores, label=method, color=color, linewidth=2)
        except Exception as e:
            warnings.warn(f"方法 {method} 失败: {e}")

    ax1.set_xlabel('预测概率 (P)', fontsize=11)
    ax1.set_ylabel('评分', fontsize=11)
    ax1.set_title('转换曲线对比', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 右图: 评分分布对比
    ax2 = axes[1]

    for method, color in zip(methods, colors):
        try:
            transformer = ScoreTransformer(
                method=method,
                lower=lower,
                upper=upper,
                direction=direction
            )
            transformer.fit(model, X)
            scores = transformer.predict_score(proba=proba)
            ax2.hist(scores, bins=30, alpha=0.5, label=method, color=color)
        except Exception as e:
            warnings.warn(f"方法 {method} 失败: {e}")

    ax2.set_xlabel('评分', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.set_title('评分分布对比', fontsize=12)
    ax2.legend(loc='best')

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# 将绘图方法添加到BaseScoreTransformer类
BaseScoreTransformer.plot_transformation_curve = plot_score_transformation_curve
