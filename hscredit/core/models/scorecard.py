# -*- coding: utf-8 -*-
"""
评分卡模型.

将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。

**核心设计原则:**

1. **fit 阶段**: 输入 WOE 转换后的数据（参考 toad/scorecardpipeline 风格）
2. **predict 阶段**: 输入原始数据，自动进行 WOE 转换
3. **灵活配置**: 支持多种方式传入分箱器、WOE转换器和LR模型
4. **pipeline 支持**: 自动识别和提取 pipeline 中的组件
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import inspect

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
    :param combiner: 特征分箱器，可选。支持以下类型：
        - hscredit 分箱器：支持 transform(X, metric='woe')
        - toad/scorecardpipeline 分箱器：输出分箱索引
    :param transfer: WOE 转换器，可选。支持以下类型：
        - hscredit WOEEncoder：支持 transform(X)
        - toad WOETransformer
    :param pipeline: 已训练的 pipeline，支持以下类型：
        - 末端为 LR：从 pipeline 中提取 LR 模型
        - 包含分箱器+WOE转换器+LR：提取所有组件
    :param calculate_stats: 是否计算统计信息，默认 True
    :param verbose: 是否输出详细信息，默认 False

    **属性**

    :ivar factor: 补偿值 B，计算方式：pdo / ln(rate)
    :ivar offset: 刻度 A，计算方式：base_score - B * ln(base_odds)
    :ivar rules_: 评分卡规则字典，包含每个特征的分箱和对应分数
    :ivar base_effect_: 每个特征的基础效应分数

    **使用方式**

    **方式1：从零开始训练（推荐，传入 WOE 数据）**

        >>> from hscredit.core.models import ScoreCard
        >>> from hscredit.core.binning import OptimalBinning
        >>> from hscredit.core.encoders import WOEEncoder
        >>> 
        >>> # 步骤1：分箱和 WOE 转换
        >>> binner = OptimalBinning(method='optimal_iv', max_n_bins=5)
        >>> binner.fit(X_train, y_train)
        >>> X_train_woe = binner.transform(X_train, metric='woe')
        >>> 
        >>> # 步骤2：训练评分卡（传入 WOE 数据）
        >>> scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scorecard.fit(X_train_woe, y_train)
        >>> 
        >>> # 步骤3：预测（传入原始数据，自动转换）
        >>> scores = scorecard.predict(X_test)  # 自动进行 WOE 转换

    **方式2：配置 combiner（分箱器自动作为 WOE 转换器）**

        >>> # hscredit 分箱器支持 transform(X, metric='woe')
        >>> binner = OptimalBinning(method='optimal_iv', max_n_bins=5)
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 配置 combiner，predict 时会自动使用 combiner.transform(X, metric='woe')
        >>> scorecard = ScoreCard(combiner=binner)
        >>> scorecard.fit(X_train_woe, y_train)  # fit 仍需传入 WOE 数据
        >>> scores = scorecard.predict(X_test)   # predict 传入原始数据

    **方式3：配置 combiner + transfer（toad/scorecardpipeline 风格）**

        >>> # toad 风格：分箱器和 WOE 转换器分开
        >>> from toad import Combiner, WOETransformer
        >>> combiner = Combiner()
        >>> combiner.fit(X_train, y_train)
        >>> transfer = WOETransformer()
        >>> transfer.fit(combiner.transform(X_train), y_train)
        >>> 
        >>> scorecard = ScoreCard(combiner=combiner, transfer=transfer)
        >>> X_train_woe = transfer.transform(combiner.transform(X_train))
        >>> scorecard.fit(X_train_woe, y_train)
        >>> scores = scorecard.predict(X_test)

    **方式4：使用预训练的 LR 模型**

        >>> from hscredit.core.models import LogisticRegression
        >>> lr = LogisticRegression(calculate_stats=True)
        >>> lr.fit(X_woe, y)
        >>> scorecard = ScoreCard(lr_model=lr)
        >>> scorecard.fit(X_woe, y)  # 主要用于生成评分规则

    **方式5：传入已训练的 pipeline**

        >>> from sklearn.pipeline import Pipeline
        >>> pipeline = Pipeline([
        ...     ('binner', OptimalBinning(method='optimal_iv')),
        ...     ('woe', WOEEncoder()),
        ...     ('lr', LogisticRegression())
        ... ])
        >>> pipeline.fit(X_train, y_train)
        >>> scorecard = ScoreCard(pipeline=pipeline)
        >>> scorecard.fit(X_train_woe, y_train)
        >>> scores = scorecard.predict(X_test)

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
        verbose: bool = False,
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
        self.verbose = verbose

        # 计算评分转换参数
        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)

        # 初始化属性
        self.rules_ = {}
        self.base_effect_ = None
        self._feature_names = None
        self.lr_model_ = None
        self._pipeline_components = {}
        
        # 内部标志：combiner 是否可以直接作为 WOE 转换器（hscredit 风格）
        self._combiner_is_woe_transformer = False
        
        # 检查外部传入的 combiner 是否支持直接 WOE 转换
        if self.combiner is not None:
            self._check_combiner_woe_capability()
        
        if verbose:
            print(f"ScoreCard 初始化: pdo={pdo}, rate={rate}, base_odds={base_odds}, base_score={base_score}")
            if self.combiner is not None:
                print(f"  - combiner: {self.combiner.__class__.__name__}, 支持WOE转换: {self._combiner_is_woe_transformer}")

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
        """评分与逾期率理论对应参照表."""
        check_is_fitted(self)

        step = max(1, int(self.pdo / 10))
        min_score = max(0, int(self.base_score - 5 * self.pdo))
        max_score = int(self.base_score + 5 * self.pdo)
        scores = np.arange(min_score, max_score + 1, step)

        results = []
        for score in scores:
            odds = np.exp((self.offset - score) / self.factor)
            prob = odds / (1 + odds)
            prob = np.clip(prob, 0, 1)

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
        """根据逾期率范围获取对应的评分参照表."""
        check_is_fitted(self)

        min_prob, max_prob = prob_range
        min_prob = max(0.0001, min(min_prob, 0.9999))
        max_prob = max(0.0001, min(max_prob, 0.9999))
        probs = np.linspace(min_prob, max_prob, n_points)

        results = []
        for prob in probs:
            odds = prob / (1 - prob)
            score = self.offset - self.factor * np.log(odds)

            results.append({
                '理论逾期率': round(prob, 6),
                '理论逾期率(%)': f"{prob*100:.4f}%",
                '理论Odds': round(odds, 4),
                '评分': round(score, 2),
            })

        return pd.DataFrame(results)

    def _validate_pipeline_components(self):
        """验证并提取 pipeline 组件.
        
        识别 pipeline 中的 LR 模型、分箱器和 WOE 转换器。
        """
        # 首先检查外部传入的 combiner
        self._check_combiner_woe_capability()
        
        if self.pipeline is None:
            return

        steps = getattr(self.pipeline, 'steps', [])
        if not steps:
            raise ValueError("pipeline 不能为空或需要 .steps 属性")

        if self.verbose:
            print(f"从 pipeline 提取组件，共 {len(steps)} 个步骤:")

        for name, obj in steps:
            # 识别 LR 模型
            if self._is_lr_model(obj) and self.lr_model_ is None:
                self.lr_model_ = obj
                if self.verbose:
                    print(f"  - 识别到 LR 模型: {name} ({obj.__class__.__name__})")
                continue

            # 识别 combiner（如果尚未传入）
            if self.combiner is None and self._is_combiner(obj):
                self.combiner = obj
                if self.verbose:
                    print(f"  - 识别到分箱器: {name} ({obj.__class__.__name__})")
                continue

            # 识别 transfer
            if self.transfer is None and self._is_woe_transformer(obj):
                self.transfer = obj
                if self.verbose:
                    print(f"  - 识别到 WOE 转换器: {name} ({obj.__class__.__name__})")
                continue

        # 再次检查 combiner 是否可以直接作为 WOE 转换器
        self._check_combiner_woe_capability()

    def _check_combiner_woe_capability(self):
        """检查 combiner 是否可以直接输出 WOE（hscredit 风格）."""
        if self.combiner is None:
            return

        # 方法1：检查是否有 bin_tables_ 属性（hscredit 分箱器特征）
        if hasattr(self.combiner, 'bin_tables_'):
            self._combiner_is_woe_transformer = True
            if self.verbose:
                print("  - 分箱器支持直接 WOE 转换（hscredit 风格）")
            return

        # 方法2：检查 transform 是否支持 metric='woe' 参数
        if hasattr(self.combiner, 'transform'):
            try:
                sig = inspect.signature(self.combiner.transform)
                params = list(sig.parameters.keys())
                if 'metric' in params:
                    self._combiner_is_woe_transformer = True
                    if self.verbose:
                        print("  - 分箱器支持 metric='woe' 参数")
                    return
            except (ValueError, TypeError):
                pass

        # 方法3：检查是否有专门的方法用于 WOE 转换
        if hasattr(self.combiner, 'transform_woe') or hasattr(self.combiner, 'woe_transform'):
            self._combiner_is_woe_transformer = True
            if self.verbose:
                print("  - 分箱器有专门的 WOE 转换方法")
            return

        self._combiner_is_woe_transformer = False

    def _is_lr_model(self, obj) -> bool:
        """判断对象是否为 LR 模型."""
        # 检查是否有 coef_ 和 intercept_ 属性
        if hasattr(obj, 'coef_') and hasattr(obj, 'intercept_'):
            return True

        # 检查类名
        class_name = obj.__class__.__name__.lower()
        lr_keywords = ('logistic', 'logit', 'linear', 'sgd', 'passiveaggressive')
        if any(kw in class_name for kw in lr_keywords):
            if hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                return True

        # 检查是否有决策函数
        if hasattr(obj, 'decision_function') and hasattr(obj, 'classes_'):
            return True

        return False

    def _is_combiner(self, obj) -> bool:
        """判断对象是否为分箱器."""
        # 检查类名
        class_name = obj.__class__.__name__.lower()
        combiner_keywords = ('combiner', 'binner', 'binning', 'bins', 'chimerge', 
                            'dtreebinner', 'optimalbinning', 'uniformbinning')
        if any(kw in class_name for kw in combiner_keywords):
            return True

        # 检查是否有分箱相关属性
        if any(hasattr(obj, attr) for attr in ['bins', 'bin_edges', 'binning_table', 
                                                  'splits_', 'bin_tables_']):
            return True

        return False

    def _is_woe_transformer(self, obj) -> bool:
        """判断对象是否为 WOE 转换器."""
        # 检查类名
        class_name = obj.__class__.__name__.lower()
        woe_keywords = ('woe', 'woetransformer', 'woeencoder', 'transfer')
        if any(kw in class_name for kw in woe_keywords):
            return True

        # 检查是否有 woe 相关属性
        if any(hasattr(obj, attr) for attr in ['woe_map', '_woe_map', 'woe_dict']):
            return True

        return False

    def _build_lr_model(self) -> LogisticRegression:
        """构建 LR 模型."""
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

    def _transform_to_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """将原始数据转换为 WOE 数据.

        转换优先级：
        1. 如果 combiner 支持直接 WOE 转换（hscredit 风格），使用 combiner.transform(X, metric='woe')
        2. 如果配置了 combiner + transfer（toad 风格），先分箱再转 WOE
        3. 如果只有 transfer，直接使用 transfer
        4. 如果没有转换器，假设输入已是 WOE 数据

        :param X: 原始数据
        :return: WOE 数据
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 情况1：combiner 支持直接 WOE 转换（hscredit 风格）
        if self._combiner_is_woe_transformer and self.combiner is not None:
            try:
                # 尝试使用 metric='woe' 参数
                X_woe = self.combiner.transform(X, metric='woe')
                if self.verbose:
                    print(f"使用 combiner.transform(X, metric='woe') 进行 WOE 转换")
                return X_woe
            except Exception as e:
                if self.verbose:
                    print(f"combiner.transform(X, metric='woe') 失败: {e}")
                # 尝试其他方法
                try:
                    X_woe = self.combiner.transform_woe(X)
                    if self.verbose:
                        print(f"使用 combiner.transform_woe(X) 进行 WOE 转换")
                    return X_woe
                except:
                    pass

        # 情况2：既有 combiner 又有 transfer（toad/scp 风格）
        if self.combiner is not None and self.transfer is not None:
            X_binned = self.combiner.transform(X)
            X_woe = self.transfer.transform(X_binned)
            if self.verbose:
                print(f"使用 combiner + transfer 进行 WOE 转换")
            return X_woe

        # 情况3：仅有 transfer
        if self.transfer is not None:
            X_woe = self.transfer.transform(X)
            if self.verbose:
                print(f"使用 transfer 进行 WOE 转换")
            return X_woe

        # 情况4：无转换器，假设输入已是 WOE 数据
        if self.verbose:
            print(f"无转换器配置，假设输入已是 WOE 数据")
        return X

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'ScoreCard':
        """训练评分卡模型.

        **重要说明**: fit 方法期望输入的是 **WOE 转换后的数据**。
        这是参考 toad 和 scorecardpipeline 的主流设计。

        如果您有原始数据，请先使用分箱器进行 WOE 转换：
            >>> binner = OptimalBinning()
            >>> binner.fit(X_train, y_train)
            >>> X_train_woe = binner.transform(X_train, metric='woe')
            >>> scorecard.fit(X_train_woe, y_train)

        :param X: WOE 转换后的训练数据（特征矩阵）
        :param y: 目标变量
        :param sample_weight: 样本权重，可选
        :return: self
        """
        if self.verbose:
            print("=" * 60)
            print("ScoreCard.fit() 开始训练")
            print(f"输入数据类型: {type(X).__name__}")

        # 转换为 DataFrame/Series
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index if len(y) == len(X) else None)

        # 1. 验证并提取 pipeline 组件
        self._validate_pipeline_components()

        # 2. 记录特征名
        self._feature_names = X.columns.tolist()
        if self.verbose:
            print(f"特征数量: {len(self._feature_names)}")
            print(f"特征列表: {self._feature_names}")

        # 3. 构建并训练/获取 LR 模型
        self.lr_model_ = self._build_lr_model()

        # 如果 LR 模型未训练，则训练
        if not hasattr(self.lr_model_, 'coef_'):
            if self.verbose:
                print("训练 LR 模型...")
            self.lr_model_.fit(X, y, sample_weight=sample_weight)
        else:
            if self.verbose:
                print("使用预训练的 LR 模型")

        # 4. 生成评分卡规则
        self._generate_rules(X)

        # 5. 计算基础效应
        sub_scores = self._woe_to_score(X)
        self.base_effect_ = pd.Series(
            np.median(sub_scores, axis=0),
            index=self.feature_names_
        )
        
        self._is_fitted = True

        if self.verbose:
            print(f"评分卡训练完成，总分 = 截距分数 + 各特征分数之和")
            print(f"截距分数: {self.offset - self.factor * self.intercept_:.2f}")
            print("=" * 60)

        return self

    def _generate_rules(self, X: pd.DataFrame):
        """生成评分卡规则."""
        self.rules_ = {}

        for i, col in enumerate(self.feature_names_):
            coef = self.coef_[i]

            # 获取该特征的 WOE 值
            woe_values = None
            bins = None
            values = None

            # 从 hscredit 的 combiner 获取分箱信息
            if self.combiner is not None and hasattr(self.combiner, 'bin_tables_'):
                if col in self.combiner.bin_tables_:
                    bin_table = self.combiner.bin_tables_[col]
                    if '分档WOE值' in bin_table.columns:
                        woe_values = bin_table['分档WOE值'].values
                        if '分箱标签' in bin_table.columns:
                            bins = self._parse_bin_labels(bin_table['分箱标签'].values)

            # 从 toad 的 transfer 获取
            elif self.transfer is not None and hasattr(self.transfer, 'get'):
                if col in self.transfer:
                    transfer_rule = self.transfer[col]
                    if isinstance(transfer_rule, dict):
                        woe_values = transfer_rule.get('woe')
                        values = transfer_rule.get('value')
                        # 尝试从 combiner 获取 bins
                        if self.combiner is not None and hasattr(self.combiner, 'get'):
                            if col in self.combiner:
                                combiner_rule = self.combiner[col]
                                if isinstance(combiner_rule, dict):
                                    bins = combiner_rule.get('bins')

            # 从训练数据推断
            if woe_values is None:
                unique_woe = X[col].dropna().unique()
                woe_values = sorted(unique_woe)
                bins = None

            woe_values = np.asarray(woe_values)

            # 计算每个 WOE 对应的分数
            scores = [self._woe_to_point(woe, coef) for woe in woe_values]

            self.rules_[col] = {
                'bins': bins if bins is not None else woe_values,
                'woe': woe_values,
                'scores': np.array(scores),
                'coef': coef,
                'values': values
            }

    def _parse_bin_labels(self, bin_labels: np.ndarray) -> list:
        """解析分箱标签为切分点或类别组."""
        numeric_splits = []
        categorical_splits = []

        for label in bin_labels:
            label_str = str(label)
            # 匹配数值区间
            match = re.match(r'\((-inf|[\d.-]+),\s*([\d.]+)\]|\[([\d.]+),\s*(inf|[\d.]+)\)', label_str)
            if match:
                if match.group(1) is not None:
                    numeric_splits.append(float(match.group(2)))
                elif match.group(3) is not None:
                    upper = match.group(4)
                    if upper != 'inf':
                        numeric_splits.append(float(upper))
            else:
                categorical_splits.append(label)

        if numeric_splits:
            return sorted(list(set(numeric_splits)))
        return categorical_splits if categorical_splits else []

    def _woe_to_point(self, woe: float, coef: float) -> float:
        """将 WOE 值转换为分数."""
        return -self.factor * coef * woe

    def _woe_to_score(self, X: pd.DataFrame) -> np.ndarray:
        """将 WOE 数据转换为分数矩阵."""
        scores = np.zeros((X.shape[0], len(self.feature_names_)))
        
        for i, col in enumerate(self.feature_names_):
            if col in X.columns:
                coef = self.coef_[i]
                scores[:, i] = -self.factor * coef * X[col].values
        
        return scores

    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'auto'
    ) -> np.ndarray:
        """预测评分.

        :param X: 输入数据
        :param input_type: 输入数据类型，可选：
            - 'auto': 自动检测（默认）
            - 'raw': 原始数据，会进行 WOE 转换
            - 'woe': WOE 数据，直接使用
        :return: 评分数组
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 检测输入数据类型
        is_woe_data = self._detect_input_type(X)
        
        if input_type == 'auto':
            # 自动检测
            if is_woe_data:
                if self.verbose:
                    print("检测到输入为 WOE 数据，直接使用")
                X_woe = X
            else:
                if self.verbose:
                    print("检测到输入为原始数据，进行 WOE 转换")
                X_woe = self._transform_to_woe(X)
        elif input_type == 'raw':
            # 强制作为原始数据处理
            X_woe = self._transform_to_woe(X)
        elif input_type == 'woe':
            # 强制作为 WOE 数据
            X_woe = X
        else:
            raise ValueError(f"input_type 必须是 'auto'/'raw'/'woe' 之一，当前为: {input_type}")

        # 确保列顺序一致
        X_woe = X_woe[self.feature_names_]

        # 计算每个特征的分数
        sub_scores = self._woe_to_score(X_woe)

        # 总分 = 截距分数 + 各特征分数之和
        intercept_score = self.offset - self.factor * self.intercept_
        total_score = intercept_score + sub_scores.sum(axis=1)

        return total_score

    def _detect_input_type(self, X: pd.DataFrame) -> bool:
        """检测输入数据是否为 WOE 数据.
        
        :param X: 输入数据
        :return: True 如果是 WOE 数据，False 如果是原始数据
        """
        # 方法1：检查数值范围（WOE 通常在 [-5, 5] 范围内）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].dropna()
            if len(X_numeric) > 0:
                min_val = X_numeric.min().min()
                max_val = X_numeric.max().max()
                # WOE 数据通常在这个范围内
                if -10 < min_val and max_val < 10:
                    # 进一步检查：如果所有值都是小数且范围较小，可能是 WOE
                    if abs(min_val) < 5 and abs(max_val) < 5:
                        return True

        # 方法2：检查是否有整数列（原始数据常见）
        for col in X.columns:
            if X[col].dtype in ['int64', 'int32']:
                # 如果是整数，可能是原始数据
                if X[col].nunique() > 10:  # 唯一值较多，可能是原始数值
                    return False

        # 默认假设为原始数据（更安全）
        return False

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率（使用底层 LR 模型）."""
        check_is_fitted(self)
        
        # 需要将原始数据转为 WOE 数据后再预测概率
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        is_woe = self._detect_input_type(X)
        if not is_woe:
            X = self._transform_to_woe(X)
        
        return self.lr_model_.predict_proba(X)

    def scorecard_scale(self) -> pd.DataFrame:
        """输出评分卡基础配置."""
        check_is_fitted(self)

        return pd.DataFrame([
            {"刻度项": "base_odds", "刻度值": self.base_odds,
             "备注": "基础比率 = (1-坏样本率)/坏样本率"},
            {"刻度项": "base_score", "刻度值": self.base_score,
             "备注": "基础 odds 对应的分数"},
            {"刻度项": "rate", "刻度值": self.rate,
             "备注": "odds 倍数"},
            {"刻度项": "pdo", "刻度值": self.pdo,
             "备注": "odds 翻倍时分数减少量"},
            {"刻度项": "B (factor)", "刻度值": round(self.factor, 4),
             "备注": "pdo / ln(rate)"},
            {"刻度项": "A (offset)", "刻度值": round(self.offset, 4),
             "备注": "base_score - B * ln(base_odds)"},
        ])

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """输出评分卡分箱信息及其对应的分数."""
        check_is_fitted(self)

        if feature_map is None:
            feature_map = {}

        rows = []
        for col in self.feature_names_:
            rule = self.rules_[col]
            bins = rule['bins']
            scores = rule['scores']
            
            if bins is None or len(bins) == 0:
                continue

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
                has_string_bins = (len(bins) > 0 and isinstance(bins[0], str) and
                                 ('[' in str(bins[0]) or '(' in str(bins[0])))
                
                if has_string_bins:
                    for bin_label, score in zip(bins, scores):
                        rows.append({
                            '变量名称': col,
                            '变量含义': feature_map.get(col, ''),
                            '变量分箱': bin_label,
                            '对应分数': round(score, 2)
                        })
                else:
                    # 数值切分点，格式化
                    for i, score in enumerate(scores):
                        if i == 0:
                            bin_label = f'[-inf, {bins[0]})' if len(bins) > 0 else '[-inf, +inf)'
                        elif i == len(scores) - 1:
                            bin_label = f'[{bins[-1]}, +inf)' if len(bins) > 0 else '[-inf, +inf)'
                        else:
                            bin_label = f'[{bins[i-1]}, {bins[i]})'
                        
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
        """输出评分区间对应坏率和 odds 的对照表."""
        df = pd.DataFrame({'score': scores, 'y': y})
        
        if method == 'quantile':
            df['score_bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop')
        else:
            df['score_bin'] = pd.cut(df['score'], bins=n_bins)
        
        stats = df.groupby('score_bin').agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        
        stats.columns = ['评分区间', '样本数', '坏样本数', '坏样本率']
        stats['好样本数'] = stats['样本数'] - stats['坏样本数']
        stats['Odds'] = stats['好样本数'] / stats['坏样本数'].replace(0, np.nan)
        stats['累计好样本占比'] = stats['好样本数'].cumsum() / stats['好样本数'].sum()
        stats['累计坏样本占比'] = stats['坏样本数'].cumsum() / stats['坏样本数'].sum()
        stats['KS'] = abs(stats['累计坏样本占比'] - stats['累计好样本占比'])
        
        stats['坏样本率'] = stats['坏样本率'].apply(lambda x: f'{x:.2%}')
        stats['Odds'] = stats['Odds'].apply(lambda x: f'{x:.2f}' if x != np.inf else 'inf')
        stats['KS'] = stats['KS'].apply(lambda x: f'{x:.4f}')
        
        return stats

    def save_pickle(self, file: str, engine: str = 'joblib') -> None:
        """保存模型."""
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
            import dill
            with open(file, 'wb') as f:
                dill.dump(self, f)
        else:
            raise ValueError(f"engine 必须是 'pickle'/'joblib'/'dill' 之一")

        print(f"模型已保存至: {file}")

    @classmethod
    def load_pickle(cls, file: str, engine: str = 'joblib') -> 'ScoreCard':
        """加载模型."""
        if engine == 'pickle':
            import pickle
            with open(file, 'rb') as f:
                return pickle.load(f)
        elif engine == 'joblib':
            import joblib
            return joblib.load(file)
        elif engine == 'dill':
            import dill
            with open(file, 'rb') as f:
                return dill.load(f)
        else:
            raise ValueError(f"engine 必须是 'pickle'/'joblib'/'dill' 之一")

    def export_pmml(self, pmml_file: str = 'scorecard.pmml', debug: bool = False):
        """导出 PMML 文件."""
        try:
            from sklearn_pandas import DataFrameMapper
            from sklearn.linear_model import LinearRegression
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer
        except ImportError as e:
            raise ImportError("导出 PMML 需要: pip install sklearn-pandas sklearn2pmml") from e

        check_is_fitted(self)

        intercept_score = self.offset - self.factor * self.intercept_

        mapper = []
        samples = {}

        for var, rule in self.rules_.items():
            bins = rule['bins']
            scores = rule['scores']

            if bins is None or len(bins) == 0:
                continue

            if isinstance(bins[0], (np.ndarray, list)):
                # 类别特征
                mapping = {}
                default_value = 0.0
                
                for bin_vals, score in zip(bins, scores):
                    for bin_val in bin_vals:
                        if pd.isna(bin_val) or bin_val == 'nan':
                            default_value = float(score)
                        else:
                            mapping[str(bin_val)] = float(score)

                mapper.append(([var], LookupTransformer(mapping=mapping, default_value=default_value)))
                samples[var] = [list(mapping.keys())[0]] * 20 if mapping else ['A'] * 20
            else:
                # 数值特征
                has_string_bins = (len(bins) > 0 and isinstance(bins[0], str) and
                                 ('[' in str(bins[0]) or '(' in str(bins[0])))
                
                if has_string_bins:
                    numeric_bins = self._extract_numeric_bins(list(bins))
                    if numeric_bins:
                        expression_string = self._build_pmml_expression(numeric_bins, scores)
                    else:
                        # 简化处理
                        expression_string = f"{scores[0]}"
                else:
                    expression_string = self._build_pmml_expression(bins, scores)
                
                mapper.append(([var], ExpressionTransformer(expression_string)))
                samples[var] = np.random.random(20) * 100

        if not mapper:
            raise ValueError("没有有效的评分规则可以导出")

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=True)),
        ])

        sample_df = pd.DataFrame(samples)
        sample_y = pd.Series(np.random.randint(0, 2, 20), name='score')
        pipeline.fit(sample_df, sample_y)
        
        pipeline.named_steps['scorecard'].coef_ = np.ones(len(mapper))
        pipeline.named_steps['scorecard'].intercept_ = intercept_score

        sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=debug)
        print(f"PMML 文件已导出至: {pmml_file}")

        if debug:
            return pipeline

    def _build_pmml_expression(self, bins: Union[np.ndarray, list], scores: np.ndarray) -> str:
        """构建 PMML 表达式字符串."""
        bins = np.asarray(bins)
        
        expression = ""
        has_empty = len(bins) > 0 and pd.isna(bins[-1])

        if has_empty:
            score_empty = scores[-1]
            bin_scores = scores[:-1]
            expression = f"{score_empty} if pandas.isnull(X[0]) "
        else:
            bin_scores = scores

        total_bins = len(bin_scores)
        end_string = ""

        for i in range(total_bins):
            if i == 0:
                _expression = f"{bin_scores[i]}"
            elif i == total_bins - 1:
                _expression += f" if X[0] < {bins[i-1]} else {bin_scores[i]}"
            else:
                _expression += f" if X[0] < {bins[i-1]} else ({bin_scores[i]} "
                end_string += ")"

        _expression += end_string

        if has_empty:
            expression += f"else ({_expression})"
        else:
            expression = _expression

        return expression

    def _extract_numeric_bins(self, bin_labels: list) -> Optional[list]:
        """从格式化的分箱标签中提取数值边界."""
        numeric_bins = []
        
        for label in bin_labels:
            if not isinstance(label, str):
                continue
            matches = re.findall(r'[-+]?\d*\.?\d+', label)
            for match in matches:
                try:
                    num = float(match)
                    if not np.isinf(num):
                        numeric_bins.append(num)
                except (ValueError, TypeError):
                    pass
        
        if numeric_bins:
            return sorted(list(set(numeric_bins)))
        return None

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性."""
        check_is_fitted(self)

        return pd.DataFrame({
            'feature': self.feature_names_,
            'coef': self.coef_,
            'importance': np.abs(self.coef_)
        }).sort_values('importance', ascending=False)

    def get_reason(self, X: Union[pd.DataFrame, np.ndarray], keep: int = 3) -> pd.DataFrame:
        """获取评分的主要原因."""
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 转换数据
        is_woe = self._detect_input_type(X)
        if not is_woe:
            X = self._transform_to_woe(X)

        sub_scores = self._woe_to_score(X[self.feature_names_])
        effect_diff = sub_scores - self.base_effect_.values
        
        reasons_list = []
        for i in range(len(X)):
            row_diff = effect_diff[i]
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
        """生成评分与逾期率对应表."""
        check_is_fitted(self)

        if scores is None:
            if X is None:
                raise ValueError("必须提供 scores 或 X 参数之一")
            scores = self.predict(X)

        scores = np.asarray(scores)

        if method == 'custom' and score_bins is not None:
            bins = pd.IntervalIndex.from_breaks(score_bins)
            score_series = pd.Series(scores)
            score_bin = pd.cut(score_series, bins=bins, include_lowest=True)
        elif method == 'uniform':
            score_bin = pd.cut(scores, bins=n_bins, include_lowest=True)
        else:
            score_bin = pd.qcut(scores, q=n_bins, duplicates='drop')

        result = []
        categories = score_bin.cat.categories if hasattr(score_bin, 'cat') else pd.Series(score_bin).unique()

        for interval in categories:
            mask = score_bin == interval
            bin_scores = scores[mask]

            if len(bin_scores) == 0:
                continue

            score_median = np.median(bin_scores)
            odds_theoretical = np.exp((self.offset - score_median) / self.factor)
            prob_theoretical = odds_theoretical / (1 + odds_theoretical)

            row = {
                '评分区间': f"[{interval.left:.0f}, {interval.right:.0f})",
                '评分中位数': round(score_median, 2),
                '理论逾期率': f"{prob_theoretical:.2%}",
                '理论Odds': f"{odds_theoretical:.2f}",
                '样本数': len(bin_scores),
            }

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

        return pd.DataFrame(result)

    def get_detailed_score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: Optional[Union[int, list]] = None,
        include_reason: bool = True
    ) -> pd.DataFrame:
        """获取每个样本的详细评分信息."""
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 转换数据
        is_woe = self._detect_input_type(X)
        X_woe = X if is_woe else self._transform_to_woe(X)

        if sample_idx is not None:
            if isinstance(sample_idx, int):
                sample_idx = [sample_idx]
            X = X.iloc[sample_idx]
            X_woe = X_woe.iloc[sample_idx]

        sub_scores = self._woe_to_score(X_woe[self.feature_names_])
        intercept_score = self.offset - self.factor * self.intercept_
        total_scores = intercept_score + sub_scores.sum(axis=1)

        data_dict = {
            ('样本信息', '样本索引'): [],
            ('样本信息', '总分'): [],
            ('样本信息', '截距分数'): [],
        }

        for col in self.feature_names_:
            data_dict[(col, '原始值')] = []
            data_dict[(col, '分箱')] = []
            data_dict[(col, 'WOE')] = []
            data_dict[(col, '分数')] = []

        for i, (idx, row) in enumerate(X.iterrows()):
            data_dict[('样本信息', '样本索引')].append(idx)
            data_dict[('样本信息', '总分')].append(round(total_scores[i], 2))
            data_dict[('样本信息', '截距分数')].append(round(intercept_score, 2))

            for j, col in enumerate(self.feature_names_):
                rule = self.rules_[col]
                woe_value = X_woe.iloc[i, j]
                score = sub_scores[i, j]
                bin_label = self._find_bin_label(col, row[col], rule)

                data_dict[(col, '原始值')].append(row[col])
                data_dict[(col, '分箱')].append(bin_label)
                data_dict[(col, 'WOE')].append(round(woe_value, 4))
                data_dict[(col, '分数')].append(round(score, 2))

        result_df = pd.DataFrame(data_dict)
        result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)

        if include_reason:
            reasons = self._generate_reasons(X_woe, sub_scores, n_reasons=3)
            result_df[('评分分析', '评分原因')] = reasons

        return result_df

    def _find_bin_label(self, feature: str, value, rule: dict) -> str:
        """根据特征值查找对应的分箱标签."""
        bins = rule['bins']

        if bins is None or len(bins) == 0:
            return '未知'

        if isinstance(bins[0], (list, np.ndarray)):
            for bin_vals in bins:
                if value in bin_vals or str(value) in [str(v) for v in bin_vals]:
                    return ', '.join([str(v) for v in bin_vals])
            return '其他'

        try:
            numeric_bins = [b for b in bins if not pd.isna(b)]
            if len(numeric_bins) == 0:
                return '未知'

            if not isinstance(numeric_bins[0], (int, float, np.number)):
                return str(bins[0]) if len(bins) > 0 else '未知'

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

            last_edge = bins[-1]
            if not pd.isna(last_edge):
                return f'[{float(last_edge)}, +inf)'

            return '未知'
        except (TypeError, ValueError):
            return str(value)

    def _generate_reasons(self, X_woe: pd.DataFrame, sub_scores: np.ndarray, n_reasons: int = 3) -> list:
        """生成评分原因."""
        effect_diff = sub_scores - self.base_effect_.values

        reasons_list = []
        for i in range(len(X_woe)):
            row_diff = effect_diff[i]
            top_indices = np.argsort(np.abs(row_diff))[::-1][:n_reasons]

            reasons = []
            for idx in top_indices:
                feature = self.feature_names_[idx]
                diff = row_diff[idx]
                score = sub_scores[i, idx]

                if diff < 0:
                    reasons.append(f"{feature}拉低{abs(diff):.1f}分(当前{score:.1f}分)")
                else:
                    reasons.append(f"{feature}提升{abs(diff):.1f}分(当前{score:.1f}分)")

            reasons_list.append('; '.join(reasons))

        return reasons_list
