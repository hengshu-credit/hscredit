"""概率转评分模块.

提供多种将概率值转换为评分的方法，支持信用评分和欺诈评分两种场景。

**核心特点**

- **纯概率输入**: fit和predict方法只接收概率值，不依赖外部模型
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

**使用示例**

>>> from hscredit.core.models.probability_to_score import ScoreTransformer
>>>
>>> # 假设已有概率值(从任何模型获取)
>>> proba = model.predict_proba(X)[:, 1]  # 或其他方式获取概率
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
>>> transformer.fit(proba)  # 只传入概率值
>>> credit_scores = transformer.predict(proba)  # 输出评分
>>>
>>> # 欺诈评分(概率越高分越高): 0-100分
>>> fraud_transformer = ScoreTransformer(
...     method='linear',
...     lower=0,
...     upper=100,
...     direction='ascending',
...     precision=0
... )
>>> fraud_transformer.fit(proba)  # 只传入概率值
>>> fraud_scores = fraud_transformer.predict(proba)  # 输出评分
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
        proba: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'BaseScoreTransformer':
        """拟合评分转换器.

        :param proba: 训练数据的预测概率(正类概率)
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

    def predict(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """预测评分.

        :param proba: 预测概率(正类概率)
        :return: 评分数组

        **示例**

        >>> proba = model.predict_proba(X_test)[:, 1]  # 从模型获取概率
        >>> scores = transformer.predict(proba)  # 转换为评分
        """
        check_is_fitted(self)

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

    # 向后兼容的方法别名
    predict_score = predict

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率.

        注意: 这是一个近似转换，可能不完全准确。

        :param scores: 评分
        :return: 概率
        """
        check_is_fitted(self)
        raise NotImplementedError("此方法需要在子类中实现")


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
    :param step: score_odds_reference的步长，默认None(自动计算为pdo/10)
    :param precision: 评分精度，默认0
    :param clip: 是否截断，默认True

    **示例**

    >>> # 从模型获取概率值
    >>> proba = model.predict_proba(X_train)[:, 1]
    >>>
    >>> transformer = StandardScoreTransformer(
    ...     lower=300,
    ...     upper=1000,
    ...     direction='descending',  # 信用分
    ...     base_odds=0.02,
    ...     base_score=600,
    ...     pdo=20,
    ...     rate=2,
    ...     step=5  # 自定义步长
    ... )
    >>> transformer.fit(proba)  # 只传入概率值
    >>> scores = transformer.predict(proba_test)  # 输出评分
    >>> # 查看评分与odds对应关系
    >>> ref = transformer.score_odds_reference
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
        step: Optional[int] = None,
        precision: int = 0,
        clip: bool = True
    ):
        super().__init__(lower, upper, direction, precision, clip)
        self.base_odds = base_odds
        self.base_score = base_score
        self.pdo = pdo
        self.rate = rate
        self.step = step

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
        proba: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'StandardScoreTransformer':
        """拟合评分转换器.

        标准评分卡方法的参数在初始化时已确定，fit主要用于验证参数合理性。

        :param proba: 训练数据的预测概率(用于验证参数合理性)
        :return: self
        """
        # 保存训练概率用于参考
        self.train_proba_ = np.asarray(proba)

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

    @property
    def score_odds_reference(self) -> pd.DataFrame:
        """评分与逾期率理论对应参照表.

        基于当前评分参数生成评分与odds、逾期率的对应关系表。
        可通过 step 参数控制步长，或在初始化时设置 self.step。

        :return: DataFrame包含评分、odds、好坏客户比例、逾期率等列
        """
        # 计算评分范围（默认以base_score为中心，±5个pdo）
        step = self.step if self.step is not None else max(1, int(self.pdo / 10))
        min_score = max(0, int(self.base_score - 5 * self.pdo))
        max_score = int(self.base_score + 5 * self.pdo)
        scores = np.arange(min_score, max_score + 1, step)

        results = []
        for score in scores:
            odds = np.exp((self.A_ - score) / self.B_)
            prob = odds / (1 + odds)
            prob = np.clip(prob, 0, 1)

            # 计算好客户:坏客户比例
            # odds = 好客户数/坏客户数，即 好:坏 = odds:1
            if odds >= 1:
                good_to_bad_ratio = f"{odds:.1f}:1"
            else:
                good_to_bad_ratio = f"1:{1/odds:.1f}"

            results.append({
                '评分': score,
                '理论Odds': round(odds, 4),
                '好客户:坏客户': good_to_bad_ratio,
                '理论逾期率': round(prob, 6),
                '理论逾期率(%)': f"{prob*100:.4f}%",
                '对数Odds': round(np.log(odds), 4) if odds > 0 else -np.inf,
            })

        return pd.DataFrame(results)

    def get_score_reference_by_prob(self, prob_range: tuple = (0.001, 0.5),
                                     n_points: int = 50) -> pd.DataFrame:
        """根据逾期率范围获取对应的评分参照表.

        :param prob_range: 概率范围，默认(0.001, 0.5)
        :param n_points: 采样点数，默认50
        :return: DataFrame包含概率、odds、评分等列
        """
        min_prob, max_prob = prob_range
        min_prob = max(0.0001, min(min_prob, 0.9999))
        max_prob = max(0.0001, min(max_prob, 0.9999))
        probs = np.linspace(min_prob, max_prob, n_points)

        results = []
        for prob in probs:
            odds = prob / (1 - prob)
            score = self.A_ - self.B_ * np.log(odds)

            results.append({
                '理论逾期率': round(prob, 6),
                '理论逾期率(%)': f"{prob*100:.4f}%",
                '理论Odds': round(odds, 4),
                '评分': round(score, 2),
            })

        return pd.DataFrame(results)


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

    >>> # 从模型获取概率值
    >>> proba = model.predict_proba(X_train)[:, 1]
    >>>
    >>> # 欺诈分(0-100, 越大越危险)
    >>> transformer = LinearScoreTransformer(
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending'
    ... )
    >>> transformer.fit(proba)  # 只传入概率值
    >>> scores = transformer.predict(proba_test)  # 输出评分
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
        proba: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'LinearScoreTransformer':
        """拟合评分转换器.

        线性方法的参数在初始化时已确定，fit主要用于保存训练概率分布。

        :param proba: 训练数据的预测概率
        :return: self
        """
        # 保存训练概率用于参考
        self.train_proba_ = np.asarray(proba)

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

    >>> # 从模型获取概率值
    >>> proba = model.predict_proba(X_train)[:, 1]
    >>>
    >>> transformer = QuantileScoreTransformer(
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending',
    ...     n_quantiles=100
    ... )
    >>> transformer.fit(proba)  # 只传入概率值，学习概率分布
    >>> scores = transformer.predict(proba_test)  # 输出评分
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
        proba: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'QuantileScoreTransformer':
        """拟合评分转换器.

        学习训练数据的概率分布，用于后续分位数映射。

        :param proba: 训练数据的预测概率
        :return: self
        """
        # 保存训练概率
        self.train_proba_ = np.asarray(proba)

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
    >>> transformer.fit(proba_train)  # 只传入概率值
    >>> credit_scores = transformer.predict(proba_test)

    **欺诈评分(概率越高分越高)**
    >>> transformer = ScoreTransformer(
    ...     method='linear',
    ...     lower=0,
    ...     upper=100,
    ...     direction='ascending'
    ... )
    >>> transformer.fit(proba_train)  # 只传入概率值
    >>> fraud_scores = transformer.predict(proba_test)

    **完整示例**
    >>> from hscredit.core.models.probability_to_score import ScoreTransformer
    >>>
    >>> # 从模型获取概率值(可从任何模型获取)
    >>> proba_train = model.predict_proba(X_train)[:, 1]
    >>> proba_test = model.predict_proba(X_test)[:, 1]
    >>>
    >>> # 创建并拟合转换器
    >>> transformer = ScoreTransformer(method='standard')
    >>> transformer.fit(proba_train)
    >>>
    >>> # 转换概率为评分
    >>> scores = transformer.predict(proba_test)
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
        proba: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'ScoreTransformer':
        """拟合评分转换器.

        :param proba: 训练数据的预测概率(正类概率)
        :return: self
        """
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
        self.transformer_.fit(proba, **kwargs)

        # 复制重要属性
        self.direction_ = self.transformer_.direction_
        if hasattr(self.transformer_, 'train_proba_'):
            self.train_proba_ = self.transformer_.train_proba_

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
    show: bool = True,
    colors: Optional[List[str]] = None
) -> Any:
    """绘制概率-评分转换曲线.

    :param transformer: 已拟合的评分转换器
    :param proba_range: 概率范围，默认0-1的100个点
    :param figsize: 图表大小，默认(10, 6)
    :param title: 图表标题，可选
    :param show: 是否显示图表，默认True
    :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
    :return: matplotlib Figure对象

    **示例**

    >>> proba = model.predict_proba(X_train)[:, 1]  # 从模型获取概率
    >>> transformer = ScoreTransformer(method='standard')
    >>> transformer.fit(proba)  # 只传入概率值
    >>> fig = plot_score_transformation_curve(transformer)
    >>> fig.savefig('transformation_curve.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    # hscredit默认配色
    if colors is None:
        colors = ["#2639E9", "#F76E6C", "#FE7715"]

    check_is_fitted(transformer)

    # 辅助函数：设置坐标轴样式
    def _setup_axis_style(ax, color="#2639E9"):
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if proba_range is None:
        proba_range = np.linspace(0.001, 0.999, 100)

    # 计算对应的评分
    scores = transformer.transform(proba_range)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(proba_range, scores, color=colors[0], linewidth=2, label='转换曲线')

    # 添加参考线
    if transformer.lower is not None:
        ax.axhline(y=transformer.lower, color=colors[1], linestyle='--', alpha=0.5, label=f'下界={transformer.lower}')
    if transformer.upper is not None:
        ax.axhline(y=transformer.upper, color=colors[1], linestyle='--', alpha=0.5, label=f'上界={transformer.upper}')

    # 设置标签
    direction_text = "递增(ascending)" if transformer.direction_ == 'ascending' else "递减(descending)"
    ax.set_xlabel('预测概率 (P)', fontsize=12, fontweight='bold')
    ax.set_ylabel('评分', fontsize=12, fontweight='bold')

    if title is None:
        title = f'概率-评分转换曲线 ({direction_text})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=False)
    _setup_axis_style(ax)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def compare_score_transformers(
    proba: Union[np.ndarray, pd.Series],
    methods: List[str] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    direction: str = 'descending',
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True,
    colors: Optional[List[str]] = None
) -> Any:
    """对比多种评分转换方法.

    :param proba: 预测概率值(用于fit和对比)
    :param methods: 要对比的方法列表，默认['standard', 'linear', 'quantile']
    :param lower: 评分下界，默认None
    :param upper: 评分上界，默认None
    :param direction: 评分方向，默认'descending'
    :param figsize: 图表大小，默认(12, 5)
    :param show: 是否显示图表，默认True
    :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
    :return: matplotlib Figure对象

    **示例**

    >>> proba = model.predict_proba(X_test)[:, 1]  # 从模型获取概率
    >>> fig = compare_score_transformers(proba)
    >>> fig.savefig('transformer_comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    # hscredit默认配色
    if colors is None:
        colors = ["#2639E9", "#F76E6C", "#FE7715", "#2E8B57", "#9370DB"]

    if methods is None:
        methods = ['standard', 'linear', 'quantile']

    # 辅助函数：设置坐标轴样式
    def _setup_axis_style(ax, color="#2639E9"):
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    proba = np.asarray(proba)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图: 转换曲线对比
    ax1 = axes[0]
    proba_range = np.linspace(0.001, 0.999, 100)

    for i, method in enumerate(methods):
        try:
            transformer = ScoreTransformer(
                method=method,
                lower=lower,
                upper=upper,
                direction=direction
            )
            transformer.fit(proba)
            scores = transformer.transform(proba_range)
            ax1.plot(proba_range, scores, label=method, color=colors[i % len(colors)], linewidth=2)
        except Exception as e:
            warnings.warn(f"方法 {method} 失败: {e}")

    ax1.set_xlabel('预测概率 (P)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('评分', fontsize=11, fontweight='bold')
    ax1.set_title('转换曲线对比', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', frameon=False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax1)

    # 右图: 评分分布对比
    ax2 = axes[1]

    for i, method in enumerate(methods):
        try:
            transformer = ScoreTransformer(
                method=method,
                lower=lower,
                upper=upper,
                direction=direction
            )
            transformer.fit(proba)
            scores = transformer.predict(proba)
            ax2.hist(scores, bins=30, alpha=0.6, label=method, color=colors[i % len(colors)], edgecolor='white')
        except Exception as e:
            warnings.warn(f"方法 {method} 失败: {e}")

    ax2.set_xlabel('评分', fontsize=11, fontweight='bold')
    ax2.set_ylabel('频数', fontsize=11, fontweight='bold')
    ax2.set_title('评分分布对比', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', frameon=False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax2)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# 将绘图方法添加到BaseScoreTransformer类
BaseScoreTransformer.plot_transformation_curve = plot_score_transformation_curve
