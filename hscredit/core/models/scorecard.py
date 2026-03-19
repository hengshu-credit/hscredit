# -*- coding: utf-8 -*-
"""
评分卡模型.

将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .logistic_regression import LogisticRegression


class ScoreCard(BaseEstimator, TransformerMixin):
    """评分卡模型.

    将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。

    **参数**

    :param pdo: odds 每增加 rate 倍时减少的分值，默认 60
    :param rate: 倍率，默认 2
    :param base_odds: 基础 odds，默认 35
        - 通常根据业务经验设置的基础比率（违约概率/正常概率）
        - 估算方法：（1-样本坏客户占比）/坏客户占比
        - 例如：35:1 => 0.972 => 坏样本率 2.8%
    :param base_score: 基础 odds 对应的分数，默认 750
    :param lr_model: 预训练的逻辑回归模型，可选
    :param lr_kwargs: 未传入 lr_model 时，通过 kwargs 传入 LR 参数进行训练，可选
    :param combiner: 特征分箱器，可选
    :param transfer: WOE 转换器，可选
    :param pipeline: 已训练的 pipeline，支持以下类型：
        - 末端为 LR：从 pipeline 中提取 LR 模型
        - 末端为分箱器：同时提取 combiner 和 transfer
        - 末端为 WOE 转换器：提取 transfer
    :param calculate_stats: 是否计算统计信息，默认 True

    **属性**

    :ivar factor: 补偿值 B，计算方式：pdo / ln(rate)
    :ivar offset: 刻度 A，计算方式：base_score - B * ln(base_odds)
    :ivar rules_: 评分卡规则字典，包含每个特征的分箱和对应分数
    :ivar base_effect_: 每个特征的基础效应分数

    **使用方式**

    方式1：从零开始训练（传入 WOE 数据）::

        >>> from hscredit.core.models import ScoreCard
        >>> scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scorecard.fit(X_woe, y)

    方式2：使用预训练的 LR 模型::

        >>> from hscredit.core.models import LogisticRegression
        >>> lr = LogisticRegression(calculate_stats=True)
        >>> lr.fit(X_woe, y)
        >>> scorecard = ScoreCard(lr_model=lr)

    方式3：未传入 lr_model，通过 kwargs 传入 LR 参数::

        >>> scorecard = ScoreCard(lr_kwargs={'C': 0.1, 'max_iter': 500})
        >>> scorecard.fit(X_woe, y)

    方式4：传入已训练的 pipeline（末端为 LR）::

        >>> from sklearn.pipeline import Pipeline
        >>> pipeline = Pipeline([
        ...     ('woe', WOEEncoder()),
        ...     ('lr', LogisticRegression())
        ... ])
        >>> pipeline.fit(X, y)
        >>> scorecard = ScoreCard(pipeline=pipeline)

    方式5：传入已训练的 pipeline（末端为分箱器）::

        >>> pipeline = Pipeline([
        ...     ('combiner', Combiner()),
        ...     ('woe', WOEEncoder())
        ... ])
        >>> pipeline.fit(X, y)
        >>> scorecard = ScoreCard(pipeline=pipeline)

    参考:
        - toad.ScoreCard
        - scorecardpipeline.ScoreCard
        - optbinning.Scorecard
    """

    def __init__(
        self,
        pdo: float = 60,
        rate: float = 2,
        base_odds: float = 35,
        base_score: float = 750,
        lr_model: Optional[Any] = None,
        lr_kwargs: Optional[Dict[str, Any]] = None,
        combiner: Optional[Any] = None,
        transfer: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        calculate_stats: bool = True,
        **kwargs
    ):
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score
        self.lr_model = lr_model
        self.lr_kwargs = lr_kwargs
        self.combiner = combiner
        self.transfer = transfer
        self.pipeline = pipeline
        self.calculate_stats = calculate_stats

        # 计算评分转换参数
        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)

        # 初始化属性
        self.rules_ = {}
        self.base_effect_ = None
        self._feature_names = None
        self.lr_model_ = None
        self._pipeline_components = {}

    @property
    def coef_(self) -> np.ndarray:
        """获取逻辑回归系数."""
        check_is_fitted(self)
        return self.lr_model_.coef_[0]

    @property
    def intercept_(self) -> float:
        """获取逻辑回归截距."""
        check_is_fitted(self)
        return self.lr_model_.intercept_[0]

    @property
    def n_features_(self) -> int:
        """获取非零系数特征数量."""
        return (self.coef_ != 0).sum()

    @property
    def feature_names_(self) -> list:
        """获取特征名列表."""
        if self._feature_names is not None:
            return self._feature_names
        if hasattr(self, 'rules_') and self.rules_:
            return list(self.rules_.keys())
        # 如果 lr_model_ 已设置，从模型获取特征数量
        if hasattr(self, 'lr_model_') and self.lr_model_ is not None:
            if hasattr(self.lr_model_, 'coef_'):
                n_features = len(self.lr_model_.coef_[0])
                return [f'feature_{i}' for i in range(n_features)]
        return []

    @property
    def score_odds_reference(self) -> pd.DataFrame:
        """评分与逾期率理论对应参照表.

        根据评分卡参数 (pdo, rate, base_odds, base_score) 计算评分与理论逾期率的对应关系。

        **评分公式:**
            Score = offset - factor * ln(Odds)
            其中: offset = base_score - factor * ln(base_odds)

        **反推公式:**
            Odds = exp((offset - Score) / factor)
            逾期率 = Odds / (1 + Odds)

        **重要说明:**
            - 评分与逾期率呈反向关系：评分越高，逾期率越低
            - 当 Score = base_score 时，对应的 Odds = 1/base_odds
            - 例如：base_score=750, base_odds=35 时，750分对应的逾期率约为 2.78%

        :return: DataFrame，包含以下列：
            - 评分: 评分值
            - 理论Odds: 对应的优势比
            - 理论逾期率: 对应的坏样本率（范围 [0, 1]）
            - 理论逾期率(%): 百分比格式的逾期率
            - 对数Odds: ln(Odds)

        **使用示例**

        ::

            >>> scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
            >>> scorecard.fit(X_woe, y)
            >>> ref_table = scorecard.score_odds_reference
            >>> print(ref_table.head(10))
        """
        check_is_fitted(self)

        # 生成评分范围：以 base_score 为中心，向两边扩展
        # 步长根据 pdo 确定，通常每个分箱区间对应一定的评分变化
        step = max(1, int(self.pdo / 10))  # 步长约为 pdo/10
        min_score = max(0, int(self.base_score - 5 * self.pdo))
        max_score = int(self.base_score + 5 * self.pdo)

        scores = np.arange(min_score, max_score + 1, step)

        # 计算每个评分对应的 Odds 和逾期率
        results = []
        for score in scores:
            # 计算 Odds: Odds = exp((offset - Score) / factor)
            odds = np.exp((self.offset - score) / self.factor)

            # 计算逾期率: prob = Odds / (1 + Odds)
            prob = odds / (1 + odds)

            # 如果逾期率超出 [0, 1] 范围，进行截断
            if prob < 0:
                prob = 0.0
            elif prob > 1:
                prob = 1.0

            results.append({
                '评分': score,
                '理论Odds': round(odds, 4),
                '理论逾期率': round(prob, 6),
                '理论逾期率(%)': f"{prob*100:.4f}%",
                '对数Odds': round(np.log(odds), 4) if odds > 0 else -np.inf,
            })

        return pd.DataFrame(results)

    def get_score_reference_by_prob(self, prob_range: tuple = (0.001, 0.5),
                                     n_points: int = 50) -> pd.DataFrame:
        """根据逾期率范围获取对应的评分参照表.

        :param prob_range: 逾期率范围 (min_prob, max_prob)，默认 (0.001, 0.5)
        :param n_points: 采样点数，默认 50
        :return: DataFrame，包含逾期率与评分的对应关系

        **使用示例**

        ::

            >>> # 获取逾期率 1%-30% 对应的评分范围
            >>> ref = scorecard.get_score_reference_by_prob((0.01, 0.30))
            >>> print(ref)
        """
        check_is_fitted(self)

        min_prob, max_prob = prob_range
        # 确保范围在 [0.0001, 0.9999] 内，避免无穷大
        min_prob = max(0.0001, min(min_prob, 0.9999))
        max_prob = max(0.0001, min(max_prob, 0.9999))

        # 在逾期率范围内均匀采样
        probs = np.linspace(min_prob, max_prob, n_points)

        results = []
        for prob in probs:
            # 逾期率转 Odds: Odds = prob / (1 - prob)
            odds = prob / (1 - prob)

            # Odds 转评分: Score = offset - factor * ln(Odds)
            score = self.offset - self.factor * np.log(odds)

            results.append({
                '理论逾期率': round(prob, 6),
                '理论逾期率(%)': f"{prob*100:.4f}%",
                '理论Odds': round(odds, 4),
                '评分': round(score, 2),
            })

        return pd.DataFrame(results)

    def _validate_pipeline_components(self):
        """验证 pipeline 组件并提取必要的模型组件.

        根据 pipeline 末端类型实现差异化的参数校验逻辑：
        - 末端为 LR：提取 LR 模型
        - 末端为分箱器 (combiner)：同时提取 combiner 和 transfer
        - 末端为 WOE 转换器：提取 transfer

        :raises ValueError: 参数组合无效时抛出
        """
        if self.pipeline is None:
            return

        # 获取 pipeline 的所有步骤
        pipeline_steps = self.pipeline.steps if hasattr(self.pipeline, 'steps') else []
        if not pipeline_steps:
            raise ValueError("pipeline 不能为空")

        # 遍历所有步骤，识别并提取组件
        for name, obj in pipeline_steps:
            # 跳过已识别的组件
            if obj in (self.lr_model_, self.combiner, self.transfer):
                continue

            # 检查是否为 LR 模型
            if self._is_lr_model(obj):
                self.lr_model_ = obj
                continue

            # 检查是否为 combiner（分箱器）
            if self._is_combiner(obj):
                self.combiner = obj
                continue

            # 检查是否为 WOE 转换器
            if self._is_woe_transformer(obj):
                self.transfer = obj
                continue

        # toad/scp 实现的 combiner 需要配套 transfer
        if self.combiner is not None and self.transfer is None:
            combiner_class = self.combiner.__class__.__name__.lower()
            # toad/scp 的 combiner 类名特征
            if 'combiner' in combiner_class and 'woe' not in combiner_class:
                raise ValueError(
                    "检测到 toad 或 scp 实现的 combiner，但未找到配套的 WOE 转换器。"
                    "请确保 pipeline 中包含 WOE 转换步骤，或显式传入 transfer 参数。"
                )

        # 存储 pipeline 组件信息
        self._pipeline_components = {
            'lr_model': self.lr_model_,
            'combiner': self.combiner,
            'transfer': self.transfer
        }

    def _is_lr_model(self, obj) -> bool:
        """判断对象是否为 LR 模型.

        :param obj: 待检测对象
        :return: 是否为 LR 模型
        """
        # 方法1: 检查是否有 coef_ 属性（已训练的线性模型）
        if hasattr(obj, 'coef_') and hasattr(obj, 'intercept_'):
            return True

        # 方法2: 检查类名
        class_name = obj.__class__.__name__.lower()
        lr_keywords = ('logistic', 'logit', 'linear', 'sgd', 'passiveaggressive')
        if any(kw in class_name for kw in lr_keywords):
            # 进一步检查是否有 predict/predict_proba 方法
            if hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                return True

        # 方法3: 检查是否有决策函数
        if hasattr(obj, 'decision_function') and hasattr(obj, 'classes_'):
            return True

        return False

    def _is_combiner(self, obj) -> bool:
        """判断对象是否为分箱器 (combiner).

        :param obj: 待检测对象
        :return: 是否为分箱器
        """
        # 方法1: 检查类名
        class_name = obj.__class__.__name__.lower()
        combiner_keywords = ('combiner', 'binner', 'binning', 'bins', 'chimerge', 'dtreebinner')
        if any(kw in class_name for kw in combiner_keywords):
            return True

        # 方法2: 检查是否有 bins 相关属性
        if hasattr(obj, 'bins') or hasattr(obj, 'bin_edges') or hasattr(obj, 'binning_table'):
            return True

        # 方法3: 检查是否有 transform 方法，且方法接受 WOE 参数（hscredit 风格）
        if hasattr(obj, 'transform'):
            import inspect
            try:
                sig = inspect.signature(obj.transform)
                params = list(sig.parameters.keys())
                if 'WOE' in params or 'woe' in params:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def _is_woe_transformer(self, obj) -> bool:
        """判断对象是否为 WOE 转换器.

        :param obj: 待检测对象
        :return: 是否为 WOE 转换器
        """
        # 方法1: 检查类名
        class_name = obj.__class__.__name__.lower()
        woe_keywords = ('woe', 'woetransformer', 'woeencoder', 'transfer', 'woeencoder')
        if any(kw in class_name for kw in woe_keywords):
            return True

        # 方法2: 检查是否有 woe 相关属性
        if hasattr(obj, 'woe_map') or hasattr(obj, '_woe_map') or hasattr(obj, 'woe_dict'):
            return True

        # 方法3: 检查 transform 方法返回值的特征
        # 对于 WOE 转换器，通常会有特定的属性标记
        if hasattr(obj, 'feature_names_in_') and hasattr(obj, 'transform'):
            # 可能是 sklearn 风格的转换器
            pass

        return False

    def _build_lr_model(self) -> LogisticRegression:
        """构建 LR 模型.

        根据传入参数优先级构建 LR 模型：
        1. 显式传入的 lr_model
        2. 从 pipeline 提取的 lr_model
        3. 通过 lr_kwargs 传入参数训练
        4. 使用默认参数训练

        :return: LR 模型实例
        """
        # 1. 如果已有 lr_model_（从 pipeline 提取），直接返回
        if self.lr_model_ is not None:
            return self.lr_model_

        # 2. 显式传入 lr_model
        if self.lr_model is not None:
            return self.lr_model

        # 3. 通过 lr_kwargs 传入参数
        if self.lr_kwargs is not None:
            lr_params = dict(self.lr_kwargs)
            lr_params.setdefault('calculate_stats', self.calculate_stats)
            return LogisticRegression(**lr_params)

        # 4. 使用默认参数
        return LogisticRegression(
            calculate_stats=self.calculate_stats,
            max_iter=1000
        )

    def _prepare_woe_data(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """准备 WOE 数据.

        根据 combiner/transfer 的配置进行数据转换：
        - 若有 combiner 且有 transfer：使用 transfer 转换
        - 若 combiner 自带 transform 方法（hscredit）：直接使用
        - 若无 combiner：假设输入已是 WOE 数据

        :param X: 原始数据
        :return: WOE 转换后的数据
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 情况1：既有 combiner 又有 transfer（toad/scp 风格）
        if self.combiner is not None and self.transfer is not None:
            # 使用 combiner 分箱，然后 transfer 转换
            X_binned = self.combiner.transform(X)
            X_woe = self.transfer.transform(X_binned)
            return X_woe

        # 情况2：combiner 自带 transform 方法（hscredit 风格）
        if self.combiner is not None and hasattr(self.combiner, 'transform'):
            # 检查是否为 hscredit 的 combiner（自带 WOE 转换）
            combiner_class = self.combiner.__class__.__name__
            # 尝试使用 metric='woe' 参数（hscredit 分箱器）
            try:
                X_woe = self.combiner.transform(X, metric='woe')
                return X_woe
            except:
                # 如果失败，尝试直接转换
                try:
                    X_woe = self.combiner.transform(X)
                    return X_woe
                except:
                    # 如果失败，尝试带 WOE 参数（其他库风格）
                    try:
                        X_woe = self.combiner.transform(X, WOE=True)
                        return X_woe
                    except:
                        pass

        # 情况3：仅有 transfer
        if self.transfer is not None:
            return self.transfer.transform(X)

        # 情况4：无转换器，假设输入已是 WOE 数据
        return X

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        X_woe: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> 'ScoreCard':
        """训练评分卡模型.

        :param X: 原始数据（可选，用于 pipeline 转换）
        :param y: 目标变量
        :param sample_weight: 样本权重
        :param X_woe: WOE 转换后的数据（可选，若提供则优先使用）
        :return: self
        """
        # 转换为 DataFrame/Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # 1. 验证并提取 pipeline 组件
        self._validate_pipeline_components()

        # 2. 获取 WOE 数据
        if X_woe is not None:
            # 优先使用传入的 WOE 数据
            if not isinstance(X_woe, pd.DataFrame):
                X_woe = pd.DataFrame(X_woe)
            X = X_woe
        elif X is not None:
            # 使用 pipeline 转换数据
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = self._prepare_woe_data(X)

        self._feature_names = X.columns.tolist()

        # 3. 构建并训练 LR 模型
        self.lr_model_: LogisticRegression = self._build_lr_model()

        # 如果 LR 模型未训练，则训练
        if not hasattr(self.lr_model_, 'coef_'):
            self.lr_model_.fit(X, y, sample_weight=sample_weight)

        # 4. 生成评分卡规则
        self._generate_rules(X)

        # 5. 计算基础效应（用于解释评分）
        sub_scores = self._woe_to_score(X)
        self.base_effect_ = pd.Series(
            np.median(sub_scores, axis=0),
            index=self.feature_names_
        )
        
        # 标记为已拟合
        self._is_fitted = True

        return self

    def _generate_rules(self, X: pd.DataFrame):
        """生成评分卡规则.

        :param X: 训练数据 (WOE 转换后的数据)
        """
        self.rules_ = {}

        for i, col in enumerate(self.feature_names_):
            coef = self.coef_[i]

            # 获取该特征的 WOE 值和分箱信息
            woe_values = None
            bins = None
            values = None  # 原始值到 WOE 的映射

            # 情况1: 从 hscredit 的 combiner 获取
            if self.combiner is not None and hasattr(self.combiner, 'bin_tables_'):
                if col in self.combiner.bin_tables_:
                    bin_table = self.combiner.bin_tables_[col]
                    if '分档WOE值' in bin_table.columns:
                        woe_values = bin_table['分档WOE值'].values
                        # 从分箱标签重建切分点
                        if '分箱标签' in bin_table.columns:
                            bins = self._parse_bin_labels(bin_table['分箱标签'].values)
                        else:
                            bins = None

            # 情况2: 从 toad 的 WOETransformer 获取
            elif self.transfer is not None and col in self.transfer:
                transfer_rule = self.transfer[col]
                if 'woe' in transfer_rule:
                    woe_values = transfer_rule['woe']
                if 'value' in transfer_rule:
                    values = transfer_rule['value']  # 原始值
                # toad 的 WOETransformer 没有 bins，需要尝试从 combiner 获取
                if self.combiner is not None and col in self.combiner:
                    combiner_rule = self.combiner[col]
                    if isinstance(combiner_rule, dict) and 'bins' in combiner_rule:
                        bins = combiner_rule['bins']

            # 情况3: 从训练数据推断（WOE 值本身就是唯一标识）
            if woe_values is None:
                unique_woe = X[col].dropna().unique()
                woe_values = sorted(unique_woe)
                bins = None

            # 确保 woe_values 是数组
            woe_values = np.asarray(woe_values)

            # 计算每个 WOE 对应的分数
            scores = []
            for woe in woe_values:
                score = self._woe_to_point(woe, coef)
                scores.append(score)

            self.rules_[col] = {
                'bins': bins if bins is not None else woe_values,
                'woe': woe_values,
                'scores': np.array(scores),
                'coef': coef,
                'values': values  # 保存原始值映射（如果有）
            }

    def _generate_rules_from_lr_model(self):
        """从预训练的 LR 模型生成评分卡规则.

        当传入预训练的 lr_model 但尚未 fit 时调用此方法生成基本规则.
        """
        self.rules_ = {}

        for i, col in enumerate(self.feature_names_):
            coef = self.lr_model_.coef_[0][i]

            # 对于预训练模型，我们不知道具体的 WOE 映射
            # 创建一个简单的规则，假设 WOE 值直接对应分数
            self.rules_[col] = {
                'bins': None,
                'woe': None,
                'scores': None,  # 将在 _woe_to_score 中动态计算
                'coef': coef,
                'values': None
            }

    def _parse_bin_labels(self, bin_labels: np.ndarray) -> list:
        """解析分箱标签为切分点或类别组.

        :param bin_labels: 分箱标签数组
        :return: 切分点列表（数值型）或类别组列表（类别型）
        """
        import re

        numeric_splits = []
        categorical_splits = []

        for label in bin_labels:
            label_str = str(label)
            # 尝试匹配数值区间，如 "(-inf, 25.0]" 或 "[25.0, 35.0)"
            match = re.match(r'\((-inf|[\d.-]+),\s*([\d.]+)\]|\[([\d.]+),\s*(inf|[\d.]+)\)', label_str)
            if match:
                if match.group(1) is not None:  # (-inf, x] 格式
                    numeric_splits.append(float(match.group(2)))
                elif match.group(3) is not None:  # [x, y) 格式
                    upper = match.group(4)
                    if upper != 'inf':
                        numeric_splits.append(float(upper))
            else:
                # 类别型或其他格式
                categorical_splits.append(label)

        # 如果有数值型切分点，返回排序后的唯一值
        if numeric_splits:
            return sorted(list(set(numeric_splits)))

        # 否则返回类别型列表
        return categorical_splits if categorical_splits else []

    def _woe_to_point(self, woe: float, coef: float) -> float:
        """将 WOE 值转换为分数.

        :param woe: WOE 值
        :param coef: 逻辑回归系数
        :return: 分数
        """
        # Score = A - B * (intercept + coef * woe)
        # 但这里只计算特征贡献部分：-B * coef * woe
        return -self.factor * coef * woe

    def _woe_to_score(self, X: pd.DataFrame) -> np.ndarray:
        """将 WOE 数据转换为分数矩阵.

        :param X: WOE 数据
        :return: 分数矩阵
        """
        scores = np.zeros((X.shape[0], len(self.feature_names_)))
        
        for i, col in enumerate(self.feature_names_):
            if col in X.columns:
                coef = self.coef_[i]
                scores[:, i] = -self.factor * coef * X[col].values
        
        return scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测评分.

        :param X: 原始数据或 WOE 转换后的数据
        :return: 评分数组
        """
        # 确保 X 是 DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 如果传入了预训练的 lr_model 但尚未 fit，则自动初始化
        if self.lr_model is not None and self.lr_model_ is None:
            # 优先从输入数据获取特征名
            self._feature_names = list(X.columns)
            
            # 设置 lr_model_
            self.lr_model_ = self.lr_model
            
            # 生成评分规则
            self._generate_rules_from_lr_model()
            
            # 标记为已拟合
            self._is_fitted = True
        
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 使用 pipeline 进行数据转换
        if self.pipeline is not None and (self.combiner is not None or self.transfer is not None):
            X = self._prepare_woe_data(X)

        # 如果有 combiner 和 transfer，则先转换（兼容旧接口）
        elif self.combiner is not None or self.transfer is not None:
            if self.combiner is not None and self.transfer is not None:
                # toad/scp 风格：先分箱再转 WOE
                X_binned = self.combiner.transform(X)
                X = self.transfer.transform(X_binned)
            elif self.combiner is not None and hasattr(self.combiner, 'transform'):
                # hscredit 风格：combiner 自带 WOE 转换，使用 metric='woe'
                try:
                    X = self.combiner.transform(X, metric='woe')
                except:
                    # 如果失败，尝试不带参数
                    X = self.combiner.transform(X)
            elif self.transfer is not None:
                # 仅使用 transfer
                X = self.transfer.transform(X)
            else:
                raise ValueError(
                    "请提供 WOE 转换后的数据，或配置 combiner 和 transfer 参数。"
                )

        # 确保列顺序一致
        X = X[self.feature_names_]

        # 计算每个特征的分数
        sub_scores = self._woe_to_score(X)

        # 总分 = 截距分数 + 各特征分数之和
        intercept_score = self.offset - self.factor * self.intercept_
        total_score = intercept_score + sub_scores.sum(axis=1)

        return total_score

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率（使用底层 LR 模型）.

        :param X: 数据
        :return: 概率数组，shape (n_samples, 2)
        """
        check_is_fitted(self)
        return self.lr_model_.predict_proba(X)

    def scorecard_scale(self) -> pd.DataFrame:
        """输出评分卡基础配置.

        :return: 包含基础配置的 DataFrame
        """
        check_is_fitted(self)

        scale_df = pd.DataFrame([
            {
                "刻度项": "base_odds",
                "刻度值": self.base_odds,
                "备注": "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比"
            },
            {
                "刻度项": "base_score",
                "刻度值": self.base_score,
                "备注": "基础ODDS对应的分数"
            },
            {
                "刻度项": "rate",
                "刻度值": self.rate,
                "备注": "设置分数的倍率"
            },
            {
                "刻度项": "pdo",
                "刻度值": self.pdo,
                "备注": "表示分数增长PDO时，ODDS值增长到RATE倍"
            },
            {
                "刻度项": "B",
                "刻度值": self.factor,
                "备注": "补偿值，计算方式：pdo / ln(rate)"
            },
            {
                "刻度项": "A",
                "刻度值": self.offset,
                "备注": "刻度，计算方式：base_score - B * ln(base_odds)"
            },
        ])

        return scale_df

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """输出评分卡分箱信息及其对应的分数.

        :param feature_map: 特征描述字典，格式为 {特征名: 描述}
        :return: 评分卡 DataFrame
        """
        check_is_fitted(self)

        if feature_map is None:
            feature_map = {}

        rows = []
        for col in self.feature_names_:
            rule = self.rules_[col]
            
            # 格式化分箱标签
            bins = rule['bins']
            scores = rule['scores']
            
            if isinstance(bins[0], (list, np.ndarray)):
                # 类别特征
                for bin_vals, score in zip(bins, scores):
                    bin_label = ', '.join([str(v) for v in bin_vals])
                    rows.append({
                        '变量名称': col,
                        '变量含义': feature_map.get(col, ''),
                        '变量分箱': bin_label,
                        '对应分数': round(score, 2)
                    })
            else:
                # 数值特征
                # 检查 bins 是否已经是格式化的标签字符串
                has_string_bins = (len(bins) > 0 and isinstance(bins[0], str) and
                                 ('[' in str(bins[0]) or '(' in str(bins[0])))
                
                if has_string_bins:
                    # bins 已经是格式化的分箱标签，直接使用
                    for i, (bin_label, score) in enumerate(zip(bins, scores)):
                        rows.append({
                            '变量名称': col,
                            '变量含义': feature_map.get(col, ''),
                            '变量分箱': bin_label,
                            '对应分数': round(score, 2)
                        })
                else:
                    # bins 是数值切分点，需要格式化
                    for i, score in enumerate(scores):
                        if i == 0:
                            # 第一个区间: [-inf, first_bin)
                            if len(bins) > 0:
                                bin_label = f'[-inf, {bins[0]})'
                            else:
                                bin_label = '[-inf, +inf)'
                        elif i == len(scores) - 1:
                            # 最后一个区间
                            if len(bins) == 0:
                                bin_label = '[-inf, +inf)'
                            elif pd.isna(bins[-1]):
                                bin_label = '缺失值'
                            elif len(bins) >= 1:
                                # 最后一个区间从最后一个切分点开始
                                bin_label = f'[{bins[-1]}, +inf)'
                            else:
                                bin_label = '[-inf, +inf)'
                        else:
                            # 中间区间
                            if i - 1 < len(bins) and i < len(bins):
                                bin_label = f'[{bins[i-1]}, {bins[i]})'
                            else:
                                bin_label = f'bin_{i}'
                        
                        rows.append({
                            '变量名称': col,
                            '变量含义': feature_map.get(col, ''),
                            '变量分箱': bin_label,
                            '对应分数': round(score, 2)
                        })

        return pd.DataFrame(rows)

    def score_to_bad_rate_table(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        method: str = 'quantile'
    ) -> pd.DataFrame:
        """输出评分区间对应坏率和odds的对照表.

        :param scores: 评分数组
        :param y: 真实标签
        :param n_bins: 分箱数量，默认 10
        :param method: 分箱方法，'quantile' 或 'uniform'
        :return: 对照表 DataFrame
        """
        # 创建 DataFrame
        df = pd.DataFrame({'score': scores, 'y': y})
        
        # 分箱
        if method == 'quantile':
            df['score_bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop')
        else:
            df['score_bin'] = pd.cut(df['score'], bins=n_bins)
        
        # 统计
        stats = df.groupby('score_bin').agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        
        stats.columns = ['评分区间', '样本数', '坏样本数', '坏样本率']
        
        # 计算 odds
        stats['好样本数'] = stats['样本数'] - stats['坏样本数']
        stats['Odds'] = stats['好样本数'] / stats['坏样本数'].replace(0, np.nan)
        stats['Odds'] = stats['Odds'].replace(np.nan, np.inf)
        
        # 计算 KS
        stats['累计好样本占比'] = stats['好样本数'].cumsum() / stats['好样本数'].sum()
        stats['累计坏样本占比'] = stats['坏样本数'].cumsum() / stats['坏样本数'].sum()
        stats['KS'] = abs(stats['累计坏样本占比'] - stats['累计好样本占比'])
        
        # 格式化
        stats['坏样本率'] = stats['坏样本率'].apply(lambda x: f'{x:.2%}')
        stats['Odds'] = stats['Odds'].apply(lambda x: f'{x:.2f}' if x != np.inf else 'inf')
        stats['KS'] = stats['KS'].apply(lambda x: f'{x:.4f}')
        
        return stats

    def save_pickle(
        self,
        file: str,
        engine: str = 'joblib'
    ) -> None:
        """保存模型为 pickle/joblib/dill 文件.

        :param file: 文件路径
        :param engine: 保存引擎，'pickle'/'joblib'/'dill'，默认 'joblib'
        """
        # 创建目录
        file_dir = os.path.dirname(file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        if engine == 'pickle':
            import pickle
            with open(file, 'wb') as f:
                pickle.dump(self, f)
        elif engine == 'joblib':
            import joblib
            joblib.dump(self, file)
        elif engine == 'dill':
            try:
                import dill
                with open(file, 'wb') as f:
                    dill.dump(self, f)
            except ImportError:
                raise ImportError("使用 dill 需要安装: pip install dill")
        else:
            raise ValueError(f"engine 参数必须是 'pickle'/'joblib'/'dill' 之一，当前为: {engine}")

        print(f"模型已保存至: {file}")

    @classmethod
    def load_pickle(cls, file: str, engine: str = 'joblib') -> 'ScoreCard':
        """从文件加载模型.

        :param file: 文件路径
        :param engine: 加载引擎，'pickle'/'joblib'/'dill'
        :return: ScoreCard 实例
        """
        if engine == 'pickle':
            import pickle
            with open(file, 'rb') as f:
                return pickle.load(f)
        elif engine == 'joblib':
            import joblib
            return joblib.load(file)
        elif engine == 'dill':
            try:
                import dill
                with open(file, 'rb') as f:
                    return dill.load(f)
            except ImportError:
                raise ImportError("使用 dill 需要安装: pip install dill")
        else:
            raise ValueError(f"engine 参数必须是 'pickle'/'joblib'/'dill' 之一，当前为: {engine}")

    def export_pmml(
        self,
        pmml_file: str = 'scorecard.pmml',
        debug: bool = False
    ) -> Optional[Any]:
        """导出评分卡模型为 PMML 文件.

        需要 JDK 1.8+ 和 sklearn2pmml 库。

        **重要说明**：
        hscredit 的 ScoreCard 计算方式为：
        - 总分 = 截距分数 + 各特征分数之和
        - 截距分数 = offset - factor * intercept_
        - 特征分数 = -factor * coef * WOE

        PMML 导出时，我们将截距分数作为 LinearRegression 的 intercept_，
        各特征的转换器直接输出该特征的分数，LR 的 coef_ 设为 1。

        :param pmml_file: PMML 文件路径
        :param debug: 是否开启调试模式
        :return: 如果 debug=True，返回 pipeline 对象
        """
        try:
            from sklearn_pandas import DataFrameMapper
            from sklearn.linear_model import LinearRegression
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer
        except ImportError as e:
            raise ImportError(
                "导出 PMML 需要安装以下依赖：\n"
                "  pip install sklearn-pandas sklearn2pmml\n"
                "同时需要 JDK 1.8+ 环境"
            ) from e

        check_is_fitted(self)

        # 计算截距分数
        intercept_score = self.offset - self.factor * self.intercept_
        if debug:
            print(f"截距分数 (intercept_score): {intercept_score:.4f}")
            print(f"  offset: {self.offset:.4f}")
            print(f"  factor: {self.factor:.4f}")
            print(f"  intercept_: {self.intercept_:.4f}")

        # 构建 DataFrameMapper
        mapper = []
        samples = {}

        for var, rule in self.rules_.items():
            bins = rule['bins']
            scores = rule['scores']

            if isinstance(bins[0], (np.ndarray, list)):
                # 类别特征：使用 LookupTransformer
                mapping = {}
                default_value = 0.0
                
                for bin_vals, score in zip(bins, scores):
                    for bin_val in bin_vals:
                        if pd.isna(bin_val) or bin_val == 'nan':
                            default_value = float(score)
                        else:
                            mapping[str(bin_val)] = float(score)

                mapper.append((
                    [var],
                    LookupTransformer(mapping=mapping, default_value=default_value)
                ))
                samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), 20)]
            else:
                # 数值特征：检查 bins 是否已经是字符串标签
                has_string_bins = (len(bins) > 0 and isinstance(bins[0], str) and
                                 ('[' in str(bins[0]) or '(' in str(bins[0])))
                
                if has_string_bins:
                    # bins 是格式化的字符串标签，尝试提取数值边界
                    numeric_bins = self._extract_numeric_bins(list(bins))
                    
                    if numeric_bins:
                        # 成功提取数值边界，使用 ExpressionTransformer
                        expression_string = self._build_expression(numeric_bins, scores)
                        mapper.append(([var], ExpressionTransformer(expression_string)))
                        samples[var] = np.random.random(20) * 100
                    else:
                        # 无法提取，使用 LookupTransformer（但会丢失范围语义）
                        # 这是一个妥协方案：由于 PMML 的 LookupTransformer 无法处理范围判断
                        # 我们给出警告，并使用简化映射
                        print(f"警告：特征 '{var}' 的分箱标签无法解析为数值边界，PMML 导出可能不正确。")
                        
                        mapping = {}
                        default_value = 0.0
                        
                        for bin_label, score in zip(bins, scores):
                            if bin_label == '缺失值' or pd.isna(bin_label):
                                default_value = float(score)
                            else:
                                mapping[bin_label] = float(score)
                        
                        mapper.append((
                            [var],
                            LookupTransformer(mapping=mapping, default_value=default_value)
                        ))
                        if mapping:
                            samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), 20)]
                        else:
                            samples[var] = np.random.random(20) * 100
                else:
                    # bins 是数值切分点，使用 ExpressionTransformer
                    expression_string = self._build_expression(bins, scores)
                    mapper.append(([var], ExpressionTransformer(expression_string)))
                    samples[var] = np.random.random(20) * 100

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        # 构建 PMML Pipeline
        # 关键：启用 fit_intercept=True，这样 LR 会学习截距项
        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=True)),
        ])

        # 拟合虚拟数据
        sample_df = pd.DataFrame(samples)
        sample_y = pd.Series(np.random.randint(0, 2, 20), name='score')
        pipeline.fit(sample_df, sample_y)
        
        # 设置正确的系数和截距
        # coef_ 设为 1（因为每个特征的转换器已经输出了正确的分数）
        pipeline.named_steps['scorecard'].coef_ = np.ones(len(mapper))
        # intercept_ 设为截距分数
        pipeline.named_steps['scorecard'].intercept_ = intercept_score
        
        if debug:
            print(f"\nPMML Pipeline 配置:")
            print(f"  特征数: {len(mapper)}")
            print(f"  coef_: {pipeline.named_steps['scorecard'].coef_}")
            print(f"  intercept_: {pipeline.named_steps['scorecard'].intercept_:.4f}")

        # 导出 PMML
        try:
            sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=debug)
            print(f"PMML 文件已导出至: {pmml_file}")
            print(f"  截距分数: {intercept_score:.4f}")
            print(f"  特征数量: {len(mapper)}")
        except Exception as e:
            import traceback
            print(f"导出 PMML 失败: {e}")
            traceback.print_exc()
            if debug:
                return pipeline
            raise

        if debug:
            return pipeline

    def _build_expression(self, bins: Union[np.ndarray, list], scores: np.ndarray) -> str:
        """构建数值特征的表达式字符串.

        :param bins: 分箱边界（数值数组）或格式化的分箱标签数组
        :param scores: 对应分数
        :return: 表达式字符串
        """
        # 如果 bins 是字符串标签，尝试提取数值边界
        if len(bins) > 0 and isinstance(bins[0], str):
            numeric_bins = self._extract_numeric_bins(bins)
            if numeric_bins:
                bins = numeric_bins
            else:
                # 无法提取，使用 LookupTransformer 方式
                raise ValueError("无法从分箱标签中提取数值边界，请使用 LookupTransformer 方式")
        
        # 确保 bins 是数组
        bins = np.asarray(bins)
        
        expression = ""
        has_empty = len(bins) > 0 and pd.isna(bins[-1])

        if has_empty:
            score_empty = scores[-1]
            bin_scores = scores[:-1]
            bin_vars = bins[:-1]
            expression = f"{score_empty} if pandas.isnull(X[0]) "
        else:
            bin_scores = scores
            bin_vars = bins

        total_bins = len(bin_scores)
        end_string = ""

        for i in range(total_bins):
            if i == 0:
                _expression = f"{bin_scores[i]}"
            elif i == total_bins - 1:
                _expression += f" if X[0] < {bin_vars[i-1]} else {bin_scores[i]}"
            else:
                _expression += f" if X[0] < {bin_vars[i-1]} else ({bin_scores[i]} "
                end_string += ")"

        _expression += end_string

        if has_empty:
            expression += f"else ({_expression})" if _expression.count('else') > 0 else _expression
        else:
            expression = _expression

        return expression

    def _extract_numeric_bins(self, bin_labels: list) -> Optional[list]:
        """从格式化的分箱标签中提取数值边界.

        :param bin_labels: 分箱标签列表，如 ['[-inf, 10.5)', '[10.5, 20.5)', '[20.5, +inf)']
        :return: 数值边界列表 [10.5, 20.5]，如果无法提取则返回 None
        """
        import re
        
        numeric_bins = []
        
        for label in bin_labels:
            if not isinstance(label, str):
                continue
            
            # 匹配区间格式：[x, y) 或 (-inf, x] 等
            # 提取数值边界
            # 查找所有数字（包括小数和负数）
            matches = re.findall(r'[-+]?\d*\.?\d+', label)
            
            # 对于区间标签，我们提取右边界（左闭右开区间）
            # 例如：[-inf, 10.5) -> 10.5
            # 例如：[10.5, 20.5) -> 10.5, 20.5
            if matches:
                # 跳过包含 'inf' 的数字（无穷大）
                for match in matches:
                    try:
                        num = float(match)
                        if not np.isinf(num):
                            numeric_bins.append(num)
                    except (ValueError, TypeError):
                        pass
        
        # 去重并排序
        if numeric_bins:
            unique_bins = sorted(list(set(numeric_bins)))
            return unique_bins
        
        return None

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性（基于系数绝对值）.

        :return: 特征重要性 DataFrame
        """
        check_is_fitted(self)

        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'coef': self.coef_,
            'importance': np.abs(self.coef_)
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_reason(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        keep: int = 3
    ) -> pd.DataFrame:
        """获取评分的主要原因（Top K 影响特征）.

        :param X: 数据
        :param keep: 保留前 K 个原因
        :return: 原因 DataFrame
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 计算各特征分数
        sub_scores = self._woe_to_score(X[self.feature_names_])
        
        # 计算与基础效应的差异
        effect_diff = sub_scores - self.base_effect_.values
        
        # 找出 Top K 原因
        reasons_list = []
        for i in range(len(X)):
            row_diff = effect_diff[i]
            # 按绝对值排序
            top_indices = np.argsort(np.abs(row_diff))[::-1][:keep]
            
            reasons = []
            for idx in top_indices:
                feature = self.feature_names_[idx]
                diff = row_diff[idx]
                direction = "降低" if diff < 0 else "提升"
                reasons.append(f"{feature}({direction}{abs(diff):.1f}分)")
            
            reasons_list.append('; '.join(reasons))

        return pd.DataFrame({'reason': reasons_list})

    def score_to_probability_table(
        self,
        scores: Optional[np.ndarray] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        n_bins: int = 10,
        method: str = 'quantile',
        score_bins: Optional[list] = None
    ) -> pd.DataFrame:
        """生成评分与逾期率对应表.

        基于评分卡的概率与好坏比转换逻辑，反推每个分数段对应的逾期率（概率）。
        该表可用于策略人员预估风险并监控评分漂移情况。

        **计算公式**

        - Score = A - B * ln(Odds)
        - Odds = exp((A - Score) / B)
        - Probability = Odds / (1 + Odds)

        :param scores: 评分数组，如果为 None 则通过 X 计算
        :param X: 特征数据，用于计算评分（当 scores 为 None 时使用）
        :param y: 真实标签，用于计算实际逾期率，可选
        :param n_bins: 分箱数量，默认 10
        :param method: 分箱方法，'quantile'（等频）或 'uniform'（等宽）或 'custom'（自定义）
        :param score_bins: 自定义分箱边界，当 method='custom' 时使用
        :return: 评分与逾期率对应表 DataFrame，包含以下字段：
            - 评分区间: 分数段范围
            - 评分中位数: 区间中位分数
            - 理论逾期率: 基于评分计算的 theoretical bad rate
            - 理论Odds: 基于评分计算的 theoretical odds
            - 样本数: 该区间样本数量（需提供 y）
            - 坏样本数: 该区间坏样本数量（需提供 y）
            - 实际逾期率: 该区间实际坏样本率（需提供 y）
            - 实际Odds: 该区间实际 odds（需提供 y）
            - 累计样本占比: 累计样本占比（需提供 y）
            - 累计坏样本占比: 累计坏样本占比（需提供 y）
            - 累计KS: 累计 KS 值（需提供 y）

        **使用示例**

        ::

            # 方式1：已有评分和标签
            table = scorecard.score_to_probability_table(scores=scores, y=y)

            # 方式2：从原始数据计算评分
            table = scorecard.score_to_probability_table(X=X_test, y=y_test)

            # 方式3：自定义分箱
            score_bins = [300, 400, 500, 600, 700, 800, 900, 1000]
            table = scorecard.score_to_probability_table(scores=scores, y=y, 
                                                         method='custom', score_bins=score_bins)
        """
        check_is_fitted(self)

        # 计算评分
        if scores is None:
            if X is None:
                raise ValueError("必须提供 scores 或 X 参数之一")
            scores = self.predict(X)

        scores = np.asarray(scores)

        # 创建基础表
        if method == 'custom' and score_bins is not None:
            # 自定义分箱
            bins = pd.IntervalIndex.from_breaks(score_bins)
            score_series = pd.Series(scores)
            score_bin = pd.cut(score_series, bins=bins, include_lowest=True)
        elif method == 'uniform':
            # 等宽分箱
            score_bin = pd.cut(scores, bins=n_bins, include_lowest=True)
        else:
            # 等频分箱（默认）
            score_bin = pd.qcut(scores, q=n_bins, duplicates='drop')

        # 计算每个分箱的统计信息
        result = []
        # 获取分箱的类别（兼容不同 pandas 版本和类型）
        if hasattr(score_bin, 'cat'):
            categories = score_bin.cat.categories
        elif hasattr(score_bin, 'categories'):
            categories = score_bin.categories
        else:
            categories = pd.Series(score_bin).unique()

        for interval in categories:
            mask = score_bin == interval
            bin_scores = scores[mask]

            if len(bin_scores) == 0:
                continue

            # 计算评分中位数
            score_median = np.median(bin_scores)

            # 计算理论逾期率和 Odds
            # Score = offset - factor * ln(Odds) => Odds = exp((offset - Score) / factor)
            odds_theoretical = np.exp((self.offset - score_median) / self.factor)
            prob_theoretical = odds_theoretical / (1 + odds_theoretical)

            row = {
                '评分区间': f"[{interval.left:.0f}, {interval.right:.0f})",
                '评分中位数': round(score_median, 2),
                '理论逾期率': f"{prob_theoretical:.2%}",
                '理论Odds': f"{odds_theoretical:.2f}",
                '样本数': len(bin_scores),
            }

            # 如果提供了真实标签，计算实际统计
            if y is not None:
                y_arr = np.asarray(y)
                bin_y = y_arr[mask]

                n_samples = len(bin_y)
                n_bad = int(bin_y.sum())
                n_good = n_samples - n_bad

                prob_actual = n_bad / n_samples if n_samples > 0 else 0
                odds_actual = n_good / n_bad if n_bad > 0 else np.inf

                row.update({
                    '坏样本数': n_bad,
                    '好样本数': n_good,
                    '实际逾期率': f"{prob_actual:.2%}",
                    '实际Odds': f"{odds_actual:.2f}" if odds_actual != np.inf else "inf",
                })

            result.append(row)

        result_df = pd.DataFrame(result)

        # 计算累计统计（如果提供了真实标签）
        if y is not None:
            result_df['累计样本数'] = result_df['样本数'].cumsum()
            result_df['累计坏样本数'] = result_df['坏样本数'].cumsum()
            result_df['累计好样本数'] = result_df['好样本数'].cumsum()
            result_df['累计样本占比'] = (result_df['累计样本数'] / result_df['样本数'].sum()).apply(lambda x: f"{x:.2%}")
            result_df['累计坏样本占比'] = (result_df['累计坏样本数'] / result_df['坏样本数'].sum()).apply(lambda x: f"{x:.2%}")
            result_df['累计好样本占比'] = (result_df['累计好样本数'] / result_df['好样本数'].sum()).apply(lambda x: f"{x:.2%}")
            result_df['累计KS'] = (result_df['累计坏样本占比'].str.rstrip('%').astype(float) / 100 -
                                    result_df['累计好样本占比'].str.rstrip('%').astype(float)).abs().apply(lambda x: f"{x:.4f}")

        return result_df

    def get_detailed_score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: Optional[Union[int, list]] = None,
        include_reason: bool = True
    ) -> pd.DataFrame:
        """获取每个样本的详细评分信息.

        针对每个样本生成评分卡每一项对应的分箱、WOE值、以及分数，
        并给出评分最终给出的原因。

        :param X: 特征数据（原始数据或 WOE 转换后的数据）
        :param sample_idx: 样本索引，如果为 None 则返回所有样本
        :param include_reason: 是否包含评分原因分析，默认 True
        :return: 详细评分信息 DataFrame，包含以下字段：
            - 样本索引
            - 总分
            - 各特征的：原始值、分箱、WOE值、分数
            - 评分原因（如果 include_reason=True）

        **使用示例**

        ::

            # 获取所有样本的详细评分
            detail = scorecard.get_detailed_score(X_test)

            # 获取指定样本的详细评分
            detail = scorecard.get_detailed_score(X_test, sample_idx=[0, 1, 2])

            # 不包含评分原因
            detail = scorecard.get_detailed_score(X_test, include_reason=False)
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 数据转换（如果需要）
        X_woe = self._prepare_woe_data(X) if (self.combiner is not None or self.transfer is not None) else X.copy()

        # 确保列顺序一致
        X_woe = X_woe[self.feature_names_]

        # 选择样本
        if sample_idx is not None:
            if isinstance(sample_idx, int):
                sample_idx = [sample_idx]
            X = X.iloc[sample_idx]
            X_woe = X_woe.iloc[sample_idx]

        # 计算每个特征的分数
        sub_scores = self._woe_to_score(X_woe)

        # 计算总分
        intercept_score = self.offset - self.factor * self.intercept_
        total_scores = intercept_score + sub_scores.sum(axis=1)

        # 构建详细结果 - 使用多级列名 (MultiIndex)
        # 第一级：特征名，第二级：子项（原始值、分箱、WOE、分数）
        data_dict = {
            ('样本信息', '样本索引'): [],
            ('样本信息', '总分'): [],
            ('样本信息', '截距分数'): [],
        }

        # 为每个特征添加多级列
        for col in self.feature_names_:
            data_dict[(col, '原始值')] = []
            data_dict[(col, '分箱')] = []
            data_dict[(col, 'WOE')] = []
            data_dict[(col, '分数')] = []

        for i, (idx, row) in enumerate(X.iterrows()):
            data_dict[('样本信息', '样本索引')].append(idx)
            data_dict[('样本信息', '总分')].append(round(total_scores[i], 2))
            data_dict[('样本信息', '截距分数')].append(round(intercept_score, 2))

            # 添加每个特征的详细信息
            for j, col in enumerate(self.feature_names_):
                rule = self.rules_[col]
                woe_value = X_woe.iloc[i, j]
                score = sub_scores[i, j]

                # 查找对应的分箱
                bin_label = self._find_bin_label(col, row[col], rule)

                data_dict[(col, '原始值')].append(row[col])
                data_dict[(col, '分箱')].append(bin_label)
                data_dict[(col, 'WOE')].append(round(woe_value, 4))
                data_dict[(col, '分数')].append(round(score, 2))

        # 创建多级列名的 DataFrame
        result_df = pd.DataFrame(data_dict)
        result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)

        # 添加评分原因
        if include_reason:
            reasons = self._generate_reasons(X_woe, sub_scores, n_reasons=3)
            result_df[('评分分析', '评分原因')] = reasons

        return result_df

    def _find_bin_label(self, feature: str, value, rule: dict) -> str:
        """根据特征值查找对应的分箱标签.

        :param feature: 特征名
        :param value: 特征值
        :param rule: 评分卡规则
        :return: 分箱标签
        """
        bins = rule['bins']

        # 处理空 bins 的情况
        if bins is None or len(bins) == 0:
            return '未知'

        # 检查是否为类别特征（bins 是列表的列表）
        if isinstance(bins[0], (list, np.ndarray)):
            # 类别特征
            for idx, bin_vals in enumerate(bins):
                if value in bin_vals or str(value) in [str(v) for v in bin_vals]:
                    return ', '.join([str(v) for v in bin_vals])
            return '其他'

        # 数值特征：检查 bins 是否都是数值
        try:
            # 尝试将 bins 转换为数值数组
            numeric_bins = [b for b in bins if not pd.isna(b)]
            if len(numeric_bins) == 0:
                return '未知'

            # 检查是否为数值型
            if not isinstance(numeric_bins[0], (int, float, np.number)):
                # 可能是字符串标签，直接返回
                return str(bins[0]) if len(bins) > 0 else '未知'

            # 数值特征处理
            if pd.isna(value):
                return '缺失值'

            value = float(value)

            for i, bin_edge in enumerate(bins):
                if pd.isna(bin_edge):
                    continue
                bin_edge = float(bin_edge)
                if i == 0 and value < bin_edge:
                    return f'[-inf, {bin_edge})'
                elif i > 0:
                    prev_edge = bins[i-1]
                    if not pd.isna(prev_edge):
                        prev_edge = float(prev_edge)
                        if prev_edge <= value < bin_edge:
                            return f'[{prev_edge}, {bin_edge})'

            # 最后一个区间
            last_edge = bins[-1]
            if not pd.isna(last_edge):
                return f'[{float(last_edge)}, +inf)'

            return '未知'
        except (TypeError, ValueError):
            # 无法比较，可能是类别型数据
            return str(value)

    def _generate_reasons(
        self,
        X_woe: pd.DataFrame,
        sub_scores: np.ndarray,
        n_reasons: int = 3
    ) -> list:
        """生成评分原因.

        :param X_woe: WOE 数据
        :param sub_scores: 各特征分数矩阵
        :param n_reasons: 原因数量
        :return: 原因列表
        """
        # 计算与基础效应的差异
        effect_diff = sub_scores - self.base_effect_.values

        reasons_list = []
        for i in range(len(X_woe)):
            row_diff = effect_diff[i]
            # 按绝对值排序，取前 N 个
            top_indices = np.argsort(np.abs(row_diff))[::-1][:n_reasons]

            reasons = []
            for idx in top_indices:
                feature = self.feature_names_[idx]
                diff = row_diff[idx]
                score = sub_scores[i, idx]

                if diff < 0:
                    direction = "拉低"
                    reasons.append(f"{feature}拉低{abs(diff):.1f}分(当前{score:.1f}分)")
                else:
                    direction = "提升"
                    reasons.append(f"{feature}提升{abs(diff):.1f}分(当前{score:.1f}分)")

            reasons_list.append('; '.join(reasons))

        return reasons_list
