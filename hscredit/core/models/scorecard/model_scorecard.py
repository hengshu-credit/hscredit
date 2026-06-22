# -*- coding: utf-8 -*-
"""通用模型评分卡模块.

将**任意输出概率的模型**（逻辑回归、sklearn 分类器、LightGBM、XGBoost、
CatBoost、NGBoost、TabNet/Net 等）转换为评分卡式的信用/欺诈评分。

**与 ScoreCard 的区别**

| 维度 | ScoreCard | ProbabilityScoreCard |
|------|-----------|----------------------|
| 输入模型 | 仅逻辑回归（WOE 线性） | 任意含 ``predict_proba`` 的模型 |
| 评分方式 | 分箱 + WOE 线性加权（可输出分箱分数表） | 概率 → 标准评分卡/线性/分位数/BoxCox 映射 |
| 适用 | 监管报送、可解释评分卡 | 树模型/神经网络等黑盒模型评分落地 |

**核心能力**

- 任意概率转评分（复用 :class:`ScoreTransformer`，支持 standard/linear/quantile/boxcox）
- 支持 LR / sklearn / LightGBM / XGBoost / CatBoost / NGBoost / Net 等模型
- 模型报告输出（评分-坏率对照表；底层为 BaseRiskModel 时复用其 report）
- 评分转换、基础参数信息留存、输出转换公式
- 持久化与离线模型加载（复用 ``hscredit.utils.io``）

**使用示例**

    >>> from hscredit.core.models import LightGBMRiskModel
    >>> from hscredit.core.models.scorecard import ProbabilityScoreCard
    >>>
    >>> model = LightGBMRiskModel()
    >>> card = ProbabilityScoreCard(
    ...     model=model, method='standard',
    ...     base_odds=0.05, base_score=600, pdo=20,
    ... )
    >>> card.fit(X_train, y_train)        # 自动训练底层模型并拟合评分映射
    >>> scores = card.predict_score(X_test)
    >>> card.score_formula()              # 输出转换公式
    >>> card.report(X_test, y_test)       # 评分-坏率对照表
    >>> card.save('model_scorecard.pkl')  # 持久化
    >>> card2 = ProbabilityScoreCard.load('model_scorecard.pkl')
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ....exceptions import NotFittedError, ValidationError
from .score_transformer import ScoreTransformer

logger = logging.getLogger(__name__)


class ProbabilityScoreCard(BaseEstimator):
    """通用模型评分卡（概率 → 评分）.

    将任意含 ``predict_proba`` 的模型输出的概率，通过 :class:`ScoreTransformer`
    转换为评分。兼容 sklearn 与 scorecardpipeline 双 API 风格。

    **参数**

    :param model: 含 ``predict_proba`` 的模型实例，可选。
        - 若为已训练模型，``fit`` 时不会重复训练（除非 ``prefit=False`` 且检测到未训练）
        - 若为未训练模型，``fit`` 时会先用 ``(X, y)`` 训练底层模型
        - 若为 None，``fit`` 时必须通过 ``proba`` 直接传入概率（纯概率转评分）
    :param method: 概率转评分方法，默认 'standard'，可选 'standard'/'linear'/'quantile'/'boxcox'
    :param lower: 评分下界，默认 None
    :param upper: 评分上界，默认 None
    :param direction: 评分方向，默认 'descending'（信用分：概率越高分越低）
    :param base_odds: 基准好坏比/坏样本率（standard 方法），默认 0.05
    :param base_score: 基准分数（standard 方法），默认 600
    :param pdo: Point of Double Odds（standard 方法），默认 20
    :param rate: 倍率（standard 方法），默认 2
    :param decimal: 评分精度（小数位数），默认 0
    :param clip: 是否截断到 [lower, upper]，默认 True
    :param prefit: 是否将 model 视为已训练（跳过训练），默认 None（自动检测）
    :param target: 目标列名，默认 'target'（用于从 DataFrame 提取 y）
    :param verbose: 是否输出详细信息，默认 False
    :param kwargs: 透传给 ScoreTransformer 的其他参数（如 n_quantiles、lmbda）

    **属性**

    :ivar model_: 实际使用的（已训练）模型
    :ivar transformer_: 已拟合的 ScoreTransformer
    :ivar A_/B_: 标准评分卡刻度参数（method='standard' 时可用）
    :ivar feature_names_in_: 训练特征名（DataFrame 输入时）
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        method: str = 'standard',
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: str = 'descending',
        base_odds: float = 0.05,
        base_score: float = 600,
        pdo: float = 20,
        rate: float = 2,
        decimal: int = 0,
        clip: bool = True,
        prefit: Optional[bool] = None,
        target: str = 'target',
        verbose: bool = False,
        **kwargs
    ):
        self.model = model
        self.method = method
        self.lower = lower
        self.upper = upper
        self.direction = direction
        self.base_odds = base_odds
        self.base_score = base_score
        self.pdo = pdo
        self.rate = rate
        self.decimal = decimal
        self.clip = clip
        self.prefit = prefit
        self.target = target
        self.verbose = verbose
        self.kwargs = kwargs

        self._is_fitted = False

    # ==================== 内部工具 ====================

    @staticmethod
    def _is_model_fitted(model: Any) -> bool:
        """判断模型是否已训练."""
        if model is None:
            return False
        # hscredit BaseRiskModel / 评分卡 风格
        if getattr(model, '_is_fitted', False):
            return True
        # sklearn 风格
        try:
            check_is_fitted(model)
            return True
        except Exception:
            pass
        # 兜底：常见的已训练标志属性
        return any(
            hasattr(model, attr)
            for attr in ('coef_', 'classes_', 'feature_importances_', 'booster_')
        )

    @staticmethod
    def _positive_proba(proba: np.ndarray) -> np.ndarray:
        """从 predict_proba 输出中提取正类（坏样本）概率."""
        proba = np.asarray(proba)
        if proba.ndim == 2:
            # 二分类取第 2 列；多分类取最后一列
            return proba[:, -1] if proba.shape[1] >= 2 else proba[:, 0]
        return proba

    def _model_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """通过底层模型计算正类概率."""
        model = self.model_ if getattr(self, 'model_', None) is not None else self.model
        if model is None:
            raise NotFittedError("未配置 model，无法从 X 计算概率，请改用 proba 参数")
        if not hasattr(model, 'predict_proba'):
            raise ValidationError(
                f"模型 {type(model).__name__} 不含 predict_proba 方法，无法转换为评分"
            )
        return self._positive_proba(model.predict_proba(X))

    def _build_transformer(self) -> ScoreTransformer:
        """根据当前参数构建 ScoreTransformer."""
        params: Dict[str, Any] = {}
        # standard 方法透传刻度参数
        if self.method == 'standard':
            params.update(
                base_odds=self.base_odds,
                base_score=self.base_score,
                pdo=self.pdo,
                rate=self.rate,
            )
        params.update(self.kwargs)
        return ScoreTransformer(
            method=self.method,
            lower=self.lower,
            upper=self.upper,
            direction=self.direction,
            decimal=self.decimal,
            clip=self.clip,
            target=self.target,
            **params,
        )

    # ==================== 训练 ====================

    def fit(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> 'ProbabilityScoreCard':
        """拟合通用模型评分卡.

        三种使用方式:

        1. 训练底层模型并拟合评分映射::

            card = ProbabilityScoreCard(model=LightGBMRiskModel())
            card.fit(X_train, y_train)

        2. 复用已训练模型，仅拟合评分映射::

            card = ProbabilityScoreCard(model=trained_model, prefit=True)
            card.fit(X_train, y_train)

        3. 纯概率转评分（无模型）::

            card = ProbabilityScoreCard(model=None)
            card.fit(proba=train_proba)

        :param X: 特征矩阵；若 y 为 None 且 init 指定了 target，则从 X 提取 y
        :param y: 目标变量，可选
        :param sample_weight: 样本权重，可选（训练底层模型时使用）
        :param proba: 直接传入训练集正类概率（无模型时使用），可选
        :return: self
        """
        # scorecardpipeline 风格：从 X 中提取 target
        if (
            proba is None
            and y is None
            and isinstance(X, pd.DataFrame)
            and self.target is not None
            and self.target in X.columns
        ):
            y = X[self.target]
            X = X.drop(columns=[self.target])

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()

        # 1. 确定训练概率
        if proba is not None:
            train_proba = self._positive_proba(np.asarray(proba))
            self.model_ = self.model  # 可能为 None
        else:
            if self.model is None:
                raise ValidationError("model 为 None 时必须通过 proba 参数传入训练概率")
            if X is None:
                raise ValidationError("必须提供 X（用于模型预测概率）或 proba")

            prefit = self.prefit
            if prefit is None:
                prefit = self._is_model_fitted(self.model)

            self.model_ = self.model
            if not prefit:
                if y is None:
                    raise ValidationError("底层模型未训练，fit 需要提供 y 进行训练")
                if self.verbose:
                    logger.info("训练底层模型: %s", type(self.model_).__name__)
                try:
                    self.model_.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    # 部分模型 fit 不支持 sample_weight
                    self.model_.fit(X, y)
            elif self.verbose:
                logger.info("复用已训练模型: %s", type(self.model_).__name__)

            train_proba = self._model_proba(X)

        # 2. 拟合评分转换器
        self.transformer_ = self._build_transformer()
        self.transformer_.fit(train_proba)

        # 3. 留存刻度参数
        self.A_ = getattr(self.transformer_.transformer_, 'A_', None)
        self.B_ = getattr(self.transformer_.transformer_, 'B_', None)
        self.direction_ = self.transformer_.direction_

        self._is_fitted = True
        return self

    # ==================== 预测 ====================

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率（透传底层模型的 predict_proba，返回二维概率矩阵）."""
        check_is_fitted(self, '_is_fitted')
        model = self.model_ if getattr(self, 'model_', None) is not None else self.model
        if model is None or not hasattr(model, 'predict_proba'):
            raise NotFittedError("当前评分卡无底层模型，无法 predict_proba，请使用 predict_score(proba=...)")
        return model.predict_proba(X)

    def predict_score(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """预测评分.

        :param X: 特征矩阵（通过底层模型计算概率）
        :param proba: 直接传入正类概率
        :return: 评分数组（已截断、四舍五入）
        """
        check_is_fitted(self, '_is_fitted')

        if proba is None:
            if X is None:
                raise ValidationError("必须提供 X 或 proba 参数之一")
            proba = self._model_proba(X)
        else:
            proba = self._positive_proba(np.asarray(proba))

        return self.transformer_.predict(proba)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测评分（与 predict_score 一致，保持评分卡模块语义）."""
        return self.predict_score(X)

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分（原始值，不截断/四舍五入）."""
        check_is_fitted(self, '_is_fitted')
        return self.transformer_.transform(self._positive_proba(np.asarray(proba)))

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率（近似）."""
        check_is_fitted(self, '_is_fitted')
        return self.transformer_.inverse_transform(scores)

    # ==================== 公式 / 参数 / 报告 ====================

    def score_formula(self, decimal: int = 4) -> Dict[str, Any]:
        """输出概率→评分的转换公式."""
        check_is_fitted(self, '_is_fitted')
        formula = self.transformer_.score_formula(decimal=decimal)
        formula['模型'] = type(self.model_).__name__ if getattr(self, 'model_', None) is not None else None
        return formula

    def get_params_info(self) -> Dict[str, Any]:
        """获取评分卡基础参数信息（用于留存/复核）."""
        check_is_fitted(self, '_is_fitted')
        return {
            '模型类型': type(self.model_).__name__ if getattr(self, 'model_', None) is not None else None,
            '转换方法': self.method,
            '评分方向': getattr(self, 'direction_', self.direction),
            'lower': self.lower,
            'upper': self.upper,
            'base_odds': self.base_odds,
            'base_score': self.base_score,
            'pdo': self.pdo,
            'rate': self.rate,
            'A': None if self.A_ is None else round(float(self.A_), 4),
            'B': None if self.B_ is None else round(float(self.B_), 4),
            '特征数': len(getattr(self, 'feature_names_in_', []) or []) or None,
        }

    def score_to_bad_rate_table(
        self,
        scores: Optional[np.ndarray] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        n_bins: int = 10,
        method: str = 'quantile',
    ) -> pd.DataFrame:
        """生成评分区间与坏样本率/odds/KS 的对照表.

        :param scores: 评分数组；为 None 时通过 X 计算
        :param y: 真实标签（坏=1）
        :param X: 特征矩阵（scores 为 None 时用于计算评分）
        :param n_bins: 分箱数，默认 10
        :param method: 分箱方式，'quantile'（等频）或 'uniform'（等宽）
        :return: 中文列名的评分-坏率对照表
        """
        check_is_fitted(self, '_is_fitted')

        if scores is None:
            if X is None:
                raise ValidationError("必须提供 scores 或 X 参数之一")
            scores = self.predict_score(X)
        scores = np.asarray(scores)

        df = pd.DataFrame({'score': scores})
        if y is not None:
            df['y'] = np.asarray(y)

        if method == 'quantile':
            df['score_bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop', precision=2)
        else:
            df['score_bin'] = pd.cut(df['score'], bins=n_bins, precision=2)

        grouped = df.groupby('score_bin', observed=True)
        rows = []
        cum_bad = cum_good = 0
        total_bad = float(df['y'].sum()) if 'y' in df else 0.0
        total_good = float((1 - df['y']).sum()) if 'y' in df else 0.0

        for interval, g in grouped:
            n = len(g)
            row = {'评分区间': str(interval), '样本数': n}
            if 'y' in df:
                n_bad = int(g['y'].sum())
                n_good = n - n_bad
                cum_bad += n_bad
                cum_good += n_good
                row['坏样本数'] = n_bad
                row['好样本数'] = n_good
                row['坏样本率'] = f"{(n_bad / n if n else 0):.2%}"
                row['Odds(好:坏)'] = f"{(n_good / n_bad):.2f}" if n_bad > 0 else 'inf'
                ks_val = abs(
                    (cum_bad / total_bad if total_bad else 0)
                    - (cum_good / total_good if total_good else 0)
                )
                row['KS'] = round(ks_val, 4)
            rows.append(row)

        return pd.DataFrame(rows)

    def report(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        scores: Optional[np.ndarray] = None,
        n_bins: int = 10,
        method: str = 'quantile',
    ) -> pd.DataFrame:
        """生成评分维度报告（评分-坏率对照表）.

        统一返回评分区间与坏样本率/odds/KS 的对照表。如需底层模型的多 Sheet
        风控报告，请使用 :meth:`model_report`。

        :param X: 特征矩阵（scores 为 None 时用于计算评分）
        :param y: 真实标签（坏=1）
        :param scores: 评分数组，可选
        :param n_bins: 分箱数，默认 10
        :param method: 分箱方式，'quantile' 或 'uniform'
        :return: 评分-坏率对照表 DataFrame
        """
        return self.score_to_bad_rate_table(
            scores=scores, y=y, X=X, n_bins=n_bins, method=method
        )

    def model_report(self, *args, **kwargs):
        """复用底层模型的报告能力.

        当底层模型为 ``BaseRiskModel`` 子类时，转调其 ``report``（生成多 Sheet
        风控建模报告），参数与 ``BaseRiskModel.report`` 一致。

        :return: 底层模型的报告对象
        :raises ValidationError: 底层模型不支持 report
        """
        check_is_fitted(self, '_is_fitted')
        model = getattr(self, 'model_', None)
        try:
            from ..base import BaseRiskModel
        except Exception:
            BaseRiskModel = ()
        if model is not None and BaseRiskModel and isinstance(model, BaseRiskModel):
            return model.report(*args, **kwargs)
        raise ValidationError(
            "底层模型不是 BaseRiskModel 子类，无法生成模型报告；请使用 report() 获取评分-坏率对照表"
        )

    # ==================== 持久化（复用 utils.io） ====================

    def save(self, file: str, engine: str = 'joblib', **kwargs) -> str:
        """保存评分卡到文件（复用 ``hscredit.utils.io.save_pickle``）.

        :param file: 文件路径
        :param engine: 序列化引擎，默认 'joblib'
        :param kwargs: 透传给 save_pickle 的参数（如 compression）
        :return: 保存的文件路径
        """
        import os
        from ....utils.io import save_pickle as _save_pickle

        check_is_fitted(self, '_is_fitted')
        file_dir = os.path.dirname(file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        _save_pickle(self, file, engine=engine, **kwargs)
        logger.info("通用模型评分卡已保存至: %s", file)
        return file

    @classmethod
    def load(cls, file: str, engine: str = 'auto', **kwargs) -> 'ProbabilityScoreCard':
        """从文件加载评分卡（离线模型加载，复用 ``hscredit.utils.io.load_pickle``）.

        :param file: 文件路径
        :param engine: 序列化引擎，默认 'auto'
        :param kwargs: 透传给 load_pickle 的参数
        :return: 加载的 ProbabilityScoreCard 实例
        """
        from ....utils.io import load_pickle as _load_pickle

        obj = _load_pickle(file, engine=engine, **kwargs)
        if not isinstance(obj, ProbabilityScoreCard):
            raise TypeError(
                f"加载的对象类型为 {type(obj).__name__}，不是 ProbabilityScoreCard"
            )
        return obj
