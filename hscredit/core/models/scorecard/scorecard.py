# -*- coding: utf-8 -*-
"""
评分卡模型.

将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。
继承 StandardScoreTransformer 实现评分计算。

**核心设计原则:**

1. **fit 阶段**: 输入 WOE 转换后的数据（参考 toad/scorecardpipeline 风格）
2. **predict 阶段**: 输入原始数据，自动进行 WOE 转换
3. **灵活配置**: 支持多种方式传入分箱器、WOE转换器和LR模型
4. **pipeline 支持**: 自动识别和提取 pipeline 中的组件
5. **评分计算**: 继承 StandardScoreTransformer，统一参数命名

**继承关系:**
    ScoreCard -> StandardScoreTransformer -> BaseScoreTransformer

**评分公式:**
    Score = A - B × ln(odds)
    其中: A = base_score + B × ln(base_odds)
          B = pdo / ln(rate)
"""

import logging
import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import inspect

from ....exceptions import DependencyError, NotFittedError, ValidationError
from ..classical.logistic_regression import LogisticRegression
from .score_transformer import StandardScoreTransformer

logger = logging.getLogger(__name__)


class _ScoreCardLoadDispatcher:
    """按访问方式分派 ScoreCard 的模型加载与规则加载接口."""

    def __get__(self, instance: Optional['ScoreCard'], owner):
        if instance is not None:
            return instance.load_rules

        def load_model(file: str, engine: str = 'auto', **kwargs):
            return owner.load_pickle(file, engine=engine, **kwargs)

        load_model.__name__ = 'load'
        load_model.__qualname__ = f'{owner.__name__}.load'
        load_model.__doc__ = (
            "从持久化文件加载评分卡模型；在实例上调用 load(...) 时兼容加载评分规则。"
        )
        return load_model


class ScoreCard(StandardScoreTransformer):
    """评分卡模型.

    将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。
    继承 StandardScoreTransformer 实现评分计算，统一参数命名。

    **参数**

    :param pdo: Point of Double Odds，odds增加rate倍时分数变化量，默认 60
    :param rate: 倍率，默认 2
        - odds增加的倍数
    :param base_odds: 好坏比（好客户:坏客户），默认 35
        - 当 base_odds >= 1 时，解释为好坏比。例如 35 表示 35:1，坏样本率 ≈ 2.8%
        - 当 base_odds < 1 时，解释为坏样本率或坏好比(P(bad)/P(good))
    :param base_score: 基础 odds 对应的分数，默认 750
    :param step: score_odds_reference的步长，默认None(自动计算为pdo/10)
    :param direction: 评分方向，默认 'descending'（信用分模式）
        - 'descending': 概率越高分越低（信用分，分越高越好）
        - 'ascending': 概率越高分越高（欺诈分，分越高越差）
    :param lr_model: 预训练的逻辑回归模型，可选
        - 如果传入，predict前不需要调用fit
        - 如果未传入，predict前必须先调用fit训练
    :param lr_kwargs: 未传入 lr_model 时，通过 kwargs 传入 LR 参数进行训练，可选
    :param binner: 特征分箱器，可选。支持以下类型：
        - hscredit 分箱器：支持 transform(X, metric='woe')
        - toad/scorecardpipeline 分箱器：输出分箱索引
    :param encoder: WOE 转换器，可选。支持以下类型：
        - hscredit WOEEncoder：支持 transform(X)
        - toad WOETransformer
    :param pipeline: 已训练的 pipeline，支持以下类型：
        - 末端为 LR：从 pipeline 中提取 LR 模型
        - 包含分箱器+WOE转换器+LR：提取所有组件
    :param calculate_stats: 是否计算统计信息，默认 True
    :param verbose: 是否输出详细信息，默认 False
    :param target: 目标列名，默认'target'

    **属性**

    :ivar A_: 刻度参数 A = base_score + B × ln(actual_odds)，其中 actual_odds = 1/base_odds (当 base_odds >= 1)
    :ivar B_: 补偿参数 B = pdo / ln(rate)
    :ivar rules_: 评分卡规则字典，包含每个特征的分箱和对应分数
    :ivar base_effect_: 每个特征的基础效应分数

    **继承方法**

    从 StandardScoreTransformer 继承的方法:
        - transform(proba): 将概率转换为评分
        - inverse_transform(scores): 将评分反向转换为概率
        - predict_score(X, proba): 通过概率预测评分
        - score_odds_reference: 评分与odds对应关系表
        - get_score_reference_by_prob(): 根据概率获取评分参考

    **评分公式**

    继承自 StandardScoreTransformer:
        Score = A - B × ln(odds)
        其中: odds = P(bad) / P(good)
              B = pdo / ln(rate)
              A = base_score + B × ln(actual_odds)
              actual_odds = 1/base_odds (当 base_odds >= 1，好坏比)
              actual_odds = base_odds   (当 base_odds < 1，坏样本率)

    **使用方式**

    **方式1：从零开始训练（fit传入 WOE 数据，predict传入原始数据）**

        >>> from hscredit.core.models import ScoreCard
        >>> from hscredit.core.binning import OptimalBinning
        >>> 
        >>> # 步骤1：分箱和 WOE 转换
        >>> binner = OptimalBinning(method='best_iv', max_n_bins=5)
        >>> binner.fit(X_train, y_train)
        >>> X_train_woe = binner.transform(X_train, metric='woe')
        >>> 
        >>> # 步骤2：训练评分卡（传入 WOE 数据）
        >>> scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scorecard.fit(X_train_woe, y_train)  # 默认 input_type='woe'
        >>> 
        >>> # 步骤3：预测（传入原始数据，自动转换）
        >>> scores = scorecard.predict(X_test)  # 默认 input_type='raw'

    **方式2：fit传入原始数据（需要配置binner进行WOE转换）**

        >>> scorecard = ScoreCard(binner=binner)  # 配置binner用于WOE转换
        >>> scorecard.fit(X_train, y_train, input_type='raw')  # 传入原始数据
        >>> scores = scorecard.predict(X_test)  # predict默认传入原始数据

    **方式3：使用预训练LR模型（无需fit，直接predict）**

        >>> lr = LogisticRegression()
        >>> lr.fit(X_train_woe, y_train)
        >>> scorecard = ScoreCard(lr_model=lr)  # 传入预训练模型
        >>> # 不需要调用fit，直接predict
        >>> scores = scorecard.predict(X_test, input_type='woe')  # 传入WOE数据

    **引用**

    标准评分卡刻度公式 ``Score = A - B·ln(odds)``（A=offset、B=factor=pdo/ln(rate)）出自
    Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent
    Credit Scoring.* Wiley。API 设计对标 toad.ScoreCard、scorecardpipeline.ScoreCard 与
    optbinning.Scorecard（https://gnpalencia.org/optbinning/scorecard.html）。
    """

    # 标记是否需要fit（根据是否传入lr_model决定）
    _skip_fit_check = False

    def __init__(
        self,
        pdo: float = 60,
        rate: float = 2,
        base_odds: float = 35,
        base_score: float = 750,
        step: Optional[int] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: str = 'descending',
        decimal: int = 2,
        lr_model: Optional[Any] = None,
        lr_kwargs: Optional[Dict[str, Any]] = None,
        binner: Optional[Any] = None,
        encoder: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        calculate_stats: bool = True,
        verbose: bool = False,
        target: str = 'target',
        **kwargs
    ):
        # 构建父类参数，ScoreCard特有参数不传递给父类
        # 评分相关参数通过kwargs透传，允许用户覆盖默认值
        parent_kwargs = {
            'lower': lower,
            'upper': upper,
            'direction': direction,
            'base_odds': base_odds,
            'base_score': base_score,
            'pdo': pdo,
            'rate': rate,
            'step': step,
            'decimal': decimal,
            'clip': True,
            **kwargs  # 用户传入的kwargs可以覆盖上述默认值
        }


        # 调用父类 StandardScoreTransformer 的初始化
        # ScoreCard 使用 descending 方向（概率越低，分数越高，信用分模式）
        super().__init__(**parent_kwargs)
        
        # ScoreCard 特有属性
        self.lr_model = lr_model
        self.lr_kwargs = lr_kwargs
        self.binner = binner
        self.encoder = encoder
        self.pipeline = pipeline
        self.calculate_stats = calculate_stats
        self.verbose = verbose
        self.target = target
        # 兼容旧版本 pickle：确保 decimal 属性存在
        self.decimal = decimal
        
        # 评分参数通过 _compute_parameters 计算（ScoreCard 重写了此方法）
        # 正确处理 base_odds 的好坏比含义
        # B_ = pdo / ln(rate)
        # A_ = base_score + B_ × ln(actual_odds)
        self.A_, self.B_ = self._compute_parameters()
        
        # 设置方向属性（父类 transform 方法需要）
        self.direction_ = self._determine_direction()

        # 初始化属性
        self.rules_ = {}
        self.base_effect_ = None
        self._feature_names = None
        self.lr_model_ = None
        self._pipeline_components = {}
        
        # 内部标志：binner 是否可以直接作为 WOE 转换器（hscredit 风格）
        self._binner_is_woe_transformer = False
        
        # 检查外部传入的 binner 是否支持直接 WOE 转换
        if self.binner is not None:
            self._check_binner_woe_capability()
        
        # 如果传入了预训练LR模型或pipeline，标记为跳过fit检查
        # 因为可以直接使用预训练模型进行predict
        self._skip_fit_check = (self.lr_model is not None) or (self.pipeline is not None)
        
        # 如果传入了预训练模型，尝试生成规则
        # 即使未提供 binner，也会基于 LR 系数生成回退规则，
        # 确保 scorecard_points 能展示每个变量（而非仅基础分）
        if self.lr_model is not None:
            self._initialize_from_pretrained()
        
        # 如果传入了 pipeline，立即提取组件
        # 这样可以在不调用 fit 的情况下直接使用 predict
        if self.pipeline is not None:
            self._validate_pipeline_components()
            # 提取组件后，如果成功获取了 lr_model_ 和 binner，初始化规则
            if self.lr_model_ is not None and self.binner is not None:
                self.lr_model = self.lr_model_  # 同步到 lr_model
                self._initialize_from_pretrained()
        
        if verbose:
            logger.info(f"ScoreCard 初始化: pdo={pdo}, rate={rate}, base_odds={base_odds}, base_score={base_score}")
            logger.info(f"  - A_ (offset)={self.A_:.4f}, B_ (factor)={self.B_:.4f}")
            if self.binner is not None:
                logger.info(f"  - binner: {self.binner.__class__.__name__}, 支持WOE转换: {self._binner_is_woe_transformer}")

    def _compute_parameters(self) -> Tuple[float, float]:
        """计算评分公式中的参数A和B.

        ScoreCard 重写父类方法，正确处理 base_odds 的含义：
        - base_odds >= 1: 解释为好坏比（好客户:坏客户），例如 35 表示 35:1，
          对应坏样本率 1/36 ≈ 2.8%。内部转换为实际 odds = 1/base_odds。
        - base_odds < 1: 解释为坏样本率或坏好比（P(bad)/P(good)），直接使用。

        标准评分卡公式:
            Score = A - B × ln(odds)
            其中: odds = P(bad) / P(good)
                  B = pdo / ln(rate)
                  A = base_score + B × ln(actual_odds_at_base)

        :return: (A, B)
        """
        B = self.pdo / np.log(self.rate)

        # 将 base_odds 转换为实际的 P(bad)/P(good) odds
        if self.base_odds >= 1:
            # base_odds 是好坏比 (good:bad)，如 35:1
            # 实际 odds = P(bad)/P(good) = 1/base_odds
            actual_odds = 1.0 / self.base_odds
        else:
            # base_odds 是坏样本率或坏好比，直接使用
            actual_odds = self.base_odds

        A = self.base_score + B * np.log(actual_odds)
        return A, B

    def _check_fitted(self) -> None:
        """检查评分卡是否已具备可用的拟合/加载状态."""
        if getattr(self, '_is_fitted', False):
            return

        # 兼容 fit 内部已训练 LR、传入已训练 LR 或已训练 pipeline 后直接使用的路径。
        lr_model = self._get_lr_model()
        if (
            lr_model is not None
            and hasattr(lr_model, 'coef_')
            and hasattr(lr_model, 'intercept_')
        ):
            return

        raise NotFittedError("ScoreCard 尚未拟合，请先调用 fit()、load() 或传入已训练的 lr_model")

    def _initialize_from_pretrained(self):
        """从预训练模型初始化规则和特征名.

        特征名解析优先级：
        1. lr_model.feature_names_in_（最可靠，与系数顺序一致）
        2. binner 的 bin_tables_ / splits_ / toad combiner（特征数匹配时）
        3. feature_0, feature_1, ... 兜底

        规则生成：优先从 binner 还原真实分箱（区间 + WOE + 分数）；
        若无任何分箱信息，则基于 LR 系数生成回退规则，
        保证 scorecard_points 至少能展示每个变量及其系数贡献。
        """
        if hasattr(self.lr_model, 'ensure_positive_woe_coefficients'):
            self.lr_model.ensure_positive_woe_coefficients()

        if not hasattr(self.lr_model, 'coef_'):
            return

        n_features = len(self.lr_model.coef_[0])

        # 1. 解析特征名
        feature_names = None
        if hasattr(self.lr_model, 'feature_names_in_'):
            feature_names = list(self.lr_model.feature_names_in_)
        elif self.binner is not None and hasattr(self.binner, 'bin_tables_') and self.binner.bin_tables_:
            feature_names = list(self.binner.bin_tables_.keys())
        elif self.binner is not None and hasattr(self.binner, 'splits_') and self.binner.splits_:
            feature_names = list(self.binner.splits_.keys())
        elif self._is_toad_like_combiner():
            feature_names = self._extract_external_binner_feature_names()

        if not feature_names or len(feature_names) < n_features:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        self._feature_names = feature_names[:n_features]

        # 2. 生成规则：优先从 binner 还原真实分箱
        self._generate_rules_from_binner()

        # 3. 无分箱信息时回退到系数级规则，避免 scorecard_points 只剩基础分
        if not self.rules_:
            self._generate_fallback_rules_from_lr()

        self._is_fitted = True

    def _generate_fallback_rules_from_lr(self):
        """无分箱信息时，基于 LR 系数为每个特征生成回退规则.

        缺少分箱器/WOE 分布时，无法还原真实分箱区间，
        因此每个特征给出一行「每单位 WOE 对应分数」（-B × coef），
        让 scorecard_points 仍能展示全部变量及其方向与贡献。
        """
        self.rules_ = {}
        for i, col in enumerate(self._feature_names):
            if i >= len(self.coef_):
                break
            coef = float(self.coef_[i]) * self._get_feature_woe_sign(i)
            label = f'每单位WOE(系数={coef:.4f})'
            self.rules_[col] = {
                'bins': None,
                'bin_labels': np.array([label], dtype=object),
                'woe': np.array([np.nan]),
                'scores': np.array([self._woe_to_point(1.0, coef)]),
                'coef': coef,
                'values': None,
            }

        if self.rules_:
            self.base_effect_ = pd.Series(
                np.zeros(len(self._feature_names)), index=self._feature_names
            )

    def _get_lr_model(self) -> Optional[Any]:
        """获取当前生效的 LR 模型."""
        if self.lr_model_ is not None:
            return self.lr_model_
        return self.lr_model

    def _get_feature_woe_sign(self, feature_index: int) -> float:
        """获取指定特征的 WOE 方向调整系数（1 或 -1）."""
        lr_model = self._get_lr_model()
        if lr_model is None:
            return 1.0

        signs = getattr(lr_model, 'woe_coef_signs_', None)
        if signs is None or feature_index >= len(signs):
            return 1.0
        return float(signs[feature_index])

    def _prepare_woe_for_scoring(self, X_woe: pd.DataFrame) -> pd.DataFrame:
        """按 LR 模型的 WOE 方向约定调整输入，保证评分与概率保持一致."""
        lr_model = self._get_lr_model()
        if lr_model is None or not hasattr(lr_model, '_prepare_input_for_model'):
            return X_woe

        prepared = lr_model._prepare_input_for_model(X_woe)
        if isinstance(prepared, pd.DataFrame):
            return prepared

        return pd.DataFrame(prepared, columns=X_woe.columns, index=X_woe.index)
    
    def _generate_rules_from_binner(self):
        """从binner生成评分卡规则（用于预训练模型）."""
        self.rules_ = {}
        
        for i, col in enumerate(self._feature_names):
            if i >= len(self.coef_):
                break
                
            coef = self.coef_[i]
            
            # 从 binner 获取分箱信息
            woe_values = None
            bins = None
            bin_labels = None
            
            # hscredit 风格
            if self.binner is not None and hasattr(self.binner, 'bin_tables_'):
                if col in self.binner.bin_tables_:
                    bin_table = self.binner.bin_tables_[col]
                    if '分档WOE值' in bin_table.columns:
                        woe_values = bin_table['分档WOE值'].values
                        if '分箱标签' in bin_table.columns:
                            bin_labels = bin_table['分箱标签'].values
                            bins = self._parse_bin_labels(bin_labels)

            # toad/scp 风格
            if woe_values is None and self._is_toad_like_combiner():
                ext_labels, ext_woe = self._extract_external_bin_info(col)
                if ext_labels is not None:
                    bin_labels = ext_labels
                    bins = self._parse_bin_labels(bin_labels)
                    if ext_woe is not None:
                        woe_values = ext_woe
            
            if woe_values is None:
                continue
                
            woe_values = np.asarray(woe_values) * self._get_feature_woe_sign(i)
            
            # 计算每个 WOE 对应的分数
            scores = [self._woe_to_point(woe, coef) for woe in woe_values]
            
            self.rules_[col] = {
                'bins': bins,
                'bin_labels': bin_labels,
                'woe': woe_values,
                'scores': np.array(scores),
                'coef': coef,
                'values': None
            }
        
        # 计算基础效应
        if self.rules_:
            self.base_effect_ = pd.Series(np.zeros(len(self._feature_names)), index=self._feature_names)

    @property
    def coef_(self) -> np.ndarray:
        """获取逻辑回归系数."""
        if getattr(self, '_loaded_coef', None) is not None:
            return self._loaded_coef

        # 如果传入了预训练模型但未调用fit，直接返回预训练模型的系数
        # 支持从 lr_model 或 lr_model_ (从pipeline提取) 获取
        if self._skip_fit_check:
            if self.lr_model is not None:
                return self.lr_model.coef_[0]
            if self.lr_model_ is not None:
                return self.lr_model_.coef_[0]
        self._check_fitted()
        if self.lr_model_ is None:
            raise ValueError("lr_model_ 为 None，请先调用fit方法或传入预训练lr_model")
        return self.lr_model_.coef_[0]

    @property
    def intercept_(self) -> float:
        """获取逻辑回归截距."""
        if getattr(self, '_loaded_intercept', None) is not None:
            return self._loaded_intercept

        # 如果传入了预训练模型但未调用fit，直接返回预训练模型的截距
        # 支持从 lr_model 或 lr_model_ (从pipeline提取) 获取
        if self._skip_fit_check:
            if self.lr_model is not None:
                return self.lr_model.intercept_[0]
            if self.lr_model_ is not None:
                return self.lr_model_.intercept_[0]
        self._check_fitted()
        if self.lr_model_ is None:
            raise ValueError("lr_model_ 为 None，请先调用fit方法或传入预训练lr_model")
        return self.lr_model_.intercept_[0]

    @property
    def n_features_(self) -> int:
        """获取非零系数特征数量."""
        return (self.coef_ != 0).sum()

    def get_feature_importances(self, importance_type: str = 'coef') -> pd.Series:
        """获取特征重要性.

        基于底层逻辑回归模型的系数计算特征重要性。

        :param importance_type: 重要性类型，默认'coef'
            - 'coef': 系数绝对值
            - 'score_range': 评分范围（最大-最小分）
        :return: 特征重要性Series
        """
        self._check_fitted()

        # 获取特征名称
        feature_names = self.feature_names_
        if not feature_names:
            n_features = len(self.coef_)
            feature_names = [f'feature_{i}' for i in range(n_features)]

        if importance_type == 'coef':
            # 使用系数绝对值
            importances = np.abs(self.coef_)
        elif importance_type == 'score_range':
            # 使用评分卡中的分数范围（每个特征各分箱分数的极差 max-min）
            if not hasattr(self, 'rules_') or not self.rules_:
                raise ValueError("评分卡规则未生成，无法使用score_range类型")
            importances = []
            for feature in feature_names:
                rule = self.rules_.get(feature) if feature in self.rules_ else None
                scores = np.asarray(rule['scores'], dtype=float) if rule and rule.get('scores') is not None else None
                if scores is not None and len(scores) > 0:
                    importances.append(float(np.max(scores) - np.min(scores)))
                else:
                    importances.append(0.0)
            importances = np.array(importances)
        else:
            raise ValueError(f"不支持的重要性类型: {importance_type}")

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=feature_names,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性属性 (兼容sklearn风格)."""
        self._check_fitted()
        if not hasattr(self, '_feature_importances'):
            self._feature_importances = self.get_feature_importances()
        return self._feature_importances.values

    @property
    def feature_names_(self) -> list:
        """获取特征名列表."""
        # 已拟合状态下，优先从 rules_ 获取（确保与实际分箱数据一致）
        if hasattr(self, 'rules_') and self.rules_:
            return list(self.rules_.keys())
        if self._feature_names is not None:
            return self._feature_names
        # 优先从binner/encoder获取真实特征名（即使lr_model已训练）
        # 这确保使用load导入规则后能正确获取特征名
        if self.binner is not None and hasattr(self.binner, 'splits_'):
            cols = list(self.binner.splits_.keys())
            if len(cols) > 0:
                return cols
        if self.encoder is not None and hasattr(self.encoder, 'mapping_'):
            cols = list(self.encoder.mapping_.keys())
            if len(cols) > 0:
                return cols
        # 如果 lr_model_ 或 lr_model 已设置，从模型获取特征数量
        lr_model = None
        if hasattr(self, 'lr_model_') and self.lr_model_ is not None:
            lr_model = self.lr_model_
        elif hasattr(self, 'lr_model') and self.lr_model is not None:
            lr_model = self.lr_model
        
        if lr_model is not None and hasattr(lr_model, 'coef_'):
            n_features = len(lr_model.coef_[0])
            return [f'feature_{i}' for i in range(n_features)]
        return []

    # score_odds_reference 和 get_score_reference_by_prob 方法
    # 已移至父类 StandardScoreTransformer，通过继承自动获得

    def _validate_pipeline_components(self):
        """验证并提取 pipeline 组件.
        
        识别 pipeline 中的 LR 模型、分箱器和 WOE 转换器。
        """
        # 首先检查外部传入的 binner
        self._check_binner_woe_capability()
        
        if self.pipeline is None:
            return

        steps = getattr(self.pipeline, 'steps', [])
        if not steps:
            raise ValueError("pipeline 不能为空或需要 .steps 属性")

        if self.verbose:
            logger.info(f"从 pipeline 提取组件，共 {len(steps)} 个步骤:")

        for name, obj in steps:
            # 识别 LR 模型
            if self._is_lr_model(obj) and self.lr_model_ is None:
                self.lr_model_ = obj
                if self.verbose:
                    logger.info(f"  - 识别到 LR 模型: {name} ({obj.__class__.__name__})")
                continue

            # 识别 binner（如果尚未传入）
            if self.binner is None and self._is_binner(obj):
                self.binner = obj
                if self.verbose:
                    logger.info(f"  - 识别到分箱器: {name} ({obj.__class__.__name__})")
                continue

            # 识别 encoder
            if self.encoder is None and self._is_woe_encoder(obj):
                self.encoder = obj
                if self.verbose:
                    logger.info(f"  - 识别到 WOE 转换器: {name} ({obj.__class__.__name__})")
                continue

        # 再次检查 binner 是否可以直接作为 WOE 转换器
        self._check_binner_woe_capability()

    def _is_toad_like_combiner(self, obj=None) -> bool:
        """检查对象是否为 toad/scorecardpipeline 风格的 Combiner.

        toad.Combiner 特征：有 rules 属性（dict），有 format_bins 方法
        scorecardpipeline.Combiner 特征：内部持有 combiner 属性（toad.Combiner 实例）
        """
        if obj is None:
            obj = self.binner
        if obj is None:
            return False
        # scorecardpipeline.Combiner wraps toad.Combiner
        if hasattr(obj, 'combiner') and hasattr(obj.combiner, 'format_bins'):
            return True
        # toad.Combiner 直接
        if hasattr(obj, 'format_bins') and hasattr(obj, 'rules') and isinstance(getattr(obj, 'rules', None), dict):
            return True
        return False

    def _get_toad_combiner(self, obj=None):
        """获取底层的 toad.Combiner 实例."""
        if obj is None:
            obj = self.binner
        if obj is None:
            return None
        # scorecardpipeline.Combiner wraps toad.Combiner
        if hasattr(obj, 'combiner') and hasattr(obj.combiner, 'format_bins'):
            return obj.combiner
        # toad.Combiner 直接
        if hasattr(obj, 'format_bins'):
            return obj
        return None

    def _get_toad_woe_transformer(self):
        """获取底层的 toad WOETransformer 实例."""
        enc = self.encoder
        if enc is None:
            return None
        # scorecardpipeline.WOETransformer wraps toad.WOETransformer
        if hasattr(enc, 'transformer') and hasattr(enc.transformer, '_rules'):
            return enc.transformer
        # toad.WOETransformer 直接
        if hasattr(enc, '_rules') or (hasattr(enc, 'rules') and isinstance(getattr(enc, 'rules', None), dict)):
            return enc
        return None

    def _extract_external_binner_feature_names(self) -> list:
        """从 toad/scp binner 中提取特征名列表."""
        combiner = self._get_toad_combiner()
        if combiner is not None:
            return list(combiner.rules.keys())
        return []

    def _extract_external_bin_info(self, col: str):
        """从 toad/scp binner + encoder 中提取指定特征的分箱标签和 WOE 值.

        :return: (bin_labels, woe_values) 或 (None, None) 如果无法提取
        """
        combiner = self._get_toad_combiner()
        if combiner is None or col not in combiner.rules:
            return None, None

        rule = combiner.rules[col]
        if not isinstance(rule, np.ndarray):
            rule = np.array(rule, dtype=object) if not isinstance(rule[0], (int, float)) else np.array(rule)

        # 用 format_bins 获取标签
        try:
            bin_labels = combiner.format_bins(rule)
        except Exception:
            return None, None

        # 获取 WOE 值：优先从 encoder 提取
        woe_values = None
        woe_enc = self._get_toad_woe_transformer()
        if woe_enc is not None and col in woe_enc.rules:
            woe_rule = woe_enc.rules[col]
            if isinstance(woe_rule, dict) and 'value' in woe_rule and 'woe' in woe_rule:
                # value 是 bin index（0,1,2...），woe 是对应的 WOE 值
                val_arr = np.asarray(woe_rule['value'])
                woe_arr = np.asarray(woe_rule['woe'])
                # 按 bin index 排序，使 WOE 与 bin_labels 对齐
                sort_idx = np.argsort(val_arr)
                woe_values = woe_arr[sort_idx]

        return np.asarray(bin_labels), woe_values

    def _check_binner_woe_capability(self):
        """检查 binner 是否可以直接输出 WOE（hscredit 风格）."""
        if self.binner is None:
            return

        # 方法1：检查是否有 bin_tables_ 属性（hscredit 分箱器特征）
        if hasattr(self.binner, 'bin_tables_'):
            self._binner_is_woe_transformer = True
            if self.verbose:
                logger.info("  - 分箱器支持直接 WOE 转换（hscredit 风格）")
            return

        # 方法2：检查 transform 是否支持 metric='woe' 参数
        if hasattr(self.binner, 'transform'):
            try:
                sig = inspect.signature(self.binner.transform)
                params = list(sig.parameters.keys())
                if 'metric' in params:
                    self._binner_is_woe_transformer = True
                    if self.verbose:
                        logger.info("  - 分箱器支持 metric='woe' 参数")
                    return
            except (ValueError, TypeError):
                pass

        # 方法3：检查是否有专门的方法用于 WOE 转换
        if hasattr(self.binner, 'transform_woe') or hasattr(self.binner, 'woe_transform'):
            self._binner_is_woe_transformer = True
            if self.verbose:
                logger.info("  - 分箱器有专门的 WOE 转换方法")
            return

        self._binner_is_woe_transformer = False

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

    def _is_binner(self, obj) -> bool:
        """判断对象是否为分箱器."""
        # 检查类名
        class_name = obj.__class__.__name__.lower()
        binner_keywords = ('combiner', 'binner', 'binning', 'bins', 'chimerge', 
                          'dtreebinner', 'optimalbinning', 'uniformbinning')
        if any(kw in class_name for kw in binner_keywords):
            return True

        # 检查是否有分箱相关属性
        if any(hasattr(obj, attr) for attr in ['bins', 'bin_edges', 'binning_table', 
                                                  'splits_', 'bin_tables_']):
            return True

        return False

    def _is_woe_encoder(self, obj) -> bool:
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
            lr_params.setdefault('positive_woe_coef', True)
            return LogisticRegression(**lr_params)

        # 4. 使用默认参数
        return LogisticRegression(
            calculate_stats=self.calculate_stats,
            positive_woe_coef=True,
            max_iter=1000
        )

    def _transform_to_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """将原始数据转换为 WOE 数据.

        转换优先级：
        1. 如果配置了 binner + encoder，先分箱再转 WOE，保持训练/预测编码口径一致
        2. 如果 binner 支持直接 WOE 转换（hscredit 风格），使用 binner.transform(X, metric='woe')
        3. 如果只有 encoder，直接使用 encoder
        4. 如果没有转换器，假设输入已是 WOE 数据

        :param X: 原始数据
        :return: WOE 数据
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 情况1：既有 binner 又有 encoder，优先使用显式 encoder 的训练口径
        if self.binner is not None and self.encoder is not None:
            try:
                X_binned = self.binner.transform(X, metric='bins')
            except TypeError:
                X_binned = self.binner.transform(X)
            X_woe = self.encoder.transform(X_binned)
            if isinstance(X_woe, pd.DataFrame):
                X_woe.attrs['hscredit_encoding'] = 'woe'
            if self.verbose:
                logger.info(f"使用 binner + encoder 进行 WOE 转换")
            return X_woe

        # 情况2：binner 支持直接 WOE 转换（hscredit 风格）
        if self._binner_is_woe_transformer and self.binner is not None:
            try:
                # 尝试使用 metric='woe' 参数
                X_woe = self.binner.transform(X, metric='woe')
                if isinstance(X_woe, pd.DataFrame):
                    X_woe.attrs['hscredit_encoding'] = 'woe'
                if self.verbose:
                    logger.info(f"使用 binner.transform(X, metric='woe') 进行 WOE 转换")
                return X_woe
            except Exception as e:
                if getattr(self.binner, 'handle_unknown', None) == 'raise':
                    raise
                if self.verbose:
                    logger.info(f"binner.transform(X, metric='woe') 失败: {e}")
                # 尝试其他方法
                try:
                    X_woe = self.binner.transform_woe(X)
                    if isinstance(X_woe, pd.DataFrame):
                        X_woe.attrs['hscredit_encoding'] = 'woe'
                    if self.verbose:
                        logger.info(f"使用 binner.transform_woe(X) 进行 WOE 转换")
                    return X_woe
                except Exception:
                    pass

        # 情况3：仅有 encoder
        if self.encoder is not None:
            X_woe = self.encoder.transform(X)
            if isinstance(X_woe, pd.DataFrame):
                X_woe.attrs['hscredit_encoding'] = 'woe'
            if self.verbose:
                logger.info(f"使用 encoder 进行 WOE 转换")
            return X_woe

        raise ValidationError(
            "原始数据评分缺少可用的分箱器或WOE编码器；请配置 binner/encoder，"
            "或对已转换数据显式使用 input_type='woe'"
        )

    def _setup_rule_based_binner(self) -> None:
        """从加载的规则中设置基于规则的分箱器.

        创建一个虚拟的分箱器，用于基于规则中的分箱信息对原始数据进行分箱。
        这样可以在不提供外部 binner 的情况下，对原始数据进行评分。
        """
        # 创建虚拟的分箱器对象
        class RuleBasedBinner:
            """基于规则的分箱器，用于离线规则评分."""
            def __init__(self, rules_dict, feature_names):
                self.rules_dict = rules_dict
                self.feature_names = feature_names
                self.feature_types_: Dict[str, Any] = {}
                self._cat_bins_: Dict[str, List[List[Any]]] = {}
                self.special_codes: List[Any] = []
                self.handle_unknown: Union[int, str] = -3

            @staticmethod
            def _match_interval(value, label):
                parsed = ScoreCard._parse_interval_label(label)
                if parsed is None:
                    return False

                if pd.isna(value):
                    return False

                left_bracket, lower, upper, right_bracket = parsed
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    return False

                if lower != -np.inf:
                    if left_bracket == '[':
                        if val < lower:
                            return False
                    elif val <= lower:
                        return False

                if upper != np.inf:
                    if right_bracket == ']':
                        if val > upper:
                            return False
                    elif val >= upper:
                        return False

                return True

            @staticmethod
            def _match_category(value, label):
                if pd.isna(value):
                    return False
                if isinstance(label, (list, tuple, np.ndarray)):
                    return any(
                        not pd.isna(category)
                        and type(value) is type(category)
                        and value == category
                        for category in label
                    )
                if ScoreCard._normalize_rule_label(label) in ('missing', 'else'):
                    return False
                value_str = str(value).strip()
                label_str = str(label).strip()
                if value_str == label_str:
                    return True
                candidates = [part.strip() for part in label_str.split(',')]
                return value_str in candidates

            def transform(self, X, metric='bins'):
                # 复制输入
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                # 确保列存在
                X = X[self.feature_names].copy()

                # 对每个特征进行分箱
                for col in self.feature_names:
                    if col not in self.rules_dict:
                        continue

                    rule = self.rules_dict[col]
                    bins = rule.get('bins', [])
                    bin_labels = rule.get('bin_labels', [])

                    if bins is None or bin_labels is None or len(bins) == 0 or len(bin_labels) == 0:
                        continue

                    # 创建分箱函数
                    def get_bin_label(value):
                        if pd.isna(value):
                            if len(bins) == len(bin_labels):
                                for index, descriptor in enumerate(bins):
                                    if isinstance(descriptor, (list, tuple, np.ndarray)) and any(
                                        pd.isna(category) for category in descriptor
                                    ):
                                        return bin_labels[index]
                            for label in bin_labels:
                                if ScoreCard._normalize_rule_label(label) == 'missing':
                                    return label
                            return '缺失值'

                        # 优先按导出的完整标签匹配，避免只依赖解析出的切分点导致中间箱丢失。
                        has_structured_categories = len(bins) == len(bin_labels) and any(
                            isinstance(descriptor, (list, tuple, np.ndarray)) for descriptor in bins
                        )
                        for index, label in enumerate(bin_labels):
                            category_descriptor = bins[index] if has_structured_categories else label
                            if self._match_interval(value, label) or self._match_category(value, category_descriptor):
                                return label

                        for label in bin_labels:
                            if ScoreCard._normalize_rule_label(label) == 'else':
                                return label

                        if self.handle_unknown == 'raise':
                            raise ValueError(f"特征 '{col}' 在 transform 中出现训练期未知类别: {[value]}")
                        return '其他'

                    X[col] = X[col].apply(get_bin_label)

                return X

        # 设置虚拟 binner，并恢复部署导出需要的特征语义。
        self._rule_binner = RuleBasedBinner(self.rules_, self.feature_names_)
        feature_types = dict(getattr(self, '_loaded_feature_types', {}))
        categorical_bins = dict(getattr(self, '_loaded_categorical_bins', {}))
        for feature, rule in self.rules_.items():
            if feature not in feature_types or feature_types[feature] is None:
                labels = rule.get('bin_labels', [])
                bins = rule.get('bins', [])
                has_interval = any(self._parse_interval_label(label) is not None for label in labels)
                has_structured_categories = len(bins) > 0 and isinstance(bins[0], (list, tuple, np.ndarray))
                feature_types[feature] = 'categorical' if has_structured_categories or not has_interval else 'numerical'
            if feature_types[feature] == 'categorical' and feature not in categorical_bins:
                bins = rule.get('bins', [])
                if len(bins) > 0 and all(isinstance(group, (list, tuple, np.ndarray)) for group in bins):
                    categorical_bins[feature] = [list(group) for group in bins]

        self._rule_binner.feature_types_ = feature_types
        self._rule_binner._cat_bins_ = categorical_bins
        self._rule_binner.special_codes = list(getattr(self, '_loaded_special_codes', []))
        self._rule_binner.handle_unknown = getattr(self, '_loaded_handle_unknown', -3)
        self.binner = self._rule_binner
        self._binner_is_woe_transformer = False

    def _has_real_lr_model(self) -> bool:
        """判断当前对象是否持有可直接做 WOE 线性评分的 LR 模型."""
        return self.lr_model_ is not None or self.lr_model is not None

    def _should_use_loaded_rule_scoring(self, input_type: str, is_woe_data: bool) -> bool:
        """判断是否应走离线规则分箱到分数的评分路径."""
        if self._has_real_lr_model() or not self.rules_:
            return False
        if input_type == 'woe' or (input_type == 'auto' and is_woe_data):
            return False
        return True

    def _transform_to_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """将原始数据转换为分箱标签数据."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 如果有外部 binner，优先使用
        if self.binner is not None and not hasattr(self, '_rule_binner'):
            try:
                return self.binner.transform(X, metric='bins')
            except Exception:
                # 如果外部 binner 不支持，尝试基于规则的转换
                pass

        # 如果有基于规则的 binner，使用它
        if hasattr(self, '_rule_binner'):
            try:
                return self._rule_binner.transform(X, metric='bins')
            except Exception as exc:
                if getattr(self._rule_binner, 'handle_unknown', None) == 'raise':
                    raise
                raise ValueError("基于规则的分箱失败") from exc

        raise ValueError("当前评分卡仅加载了规则，predict(input_type='raw') 需要提供支持 transform(metric='bins') 的 binner")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        input_type: str = 'woe',
    ) -> 'ScoreCard':
        """训练评分卡模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: 在__init__中指定target，然后fit(X)

        **输入数据类型**

        fit 方法支持两种输入数据类型，通过 `input_type` 参数控制：
        - 'woe': WOE 转换后的数据（默认）
        - 'raw': 原始数据（需要配置 binner 进行 WOE 转换）

        **使用 WOE 数据（推荐）**:
            >>> binner = OptimalBinning()
            >>> binner.fit(X_train, y_train)
            >>> X_train_woe = binner.transform(X_train, metric='woe')
            >>> scorecard.fit(X_train_woe, y_train)  # 默认 input_type='woe'

        **使用原始数据**:
            >>> scorecard = ScoreCard(binner=binner)  # 需要配置binner
            >>> scorecard.fit(X_train, y_train, input_type='raw')

        :param X: 训练数据（特征矩阵）
            支持 numpy array 或 pandas DataFrame
            如果是DataFrame且y为None，会尝试从X中提取target列作为y
            数据类型由 input_type 参数决定（woe或raw）
        :param y: 目标变量，可选
            如果为None且init中指定了target，则从X中提取
        :param sample_weight: 样本权重，可选
        :param input_type: 输入数据类型，默认为'woe'
            - 'woe': WOE 转换后的数据（默认，推荐）
            - 'raw': 原始数据（需要配置 binner 进行 WOE 转换）
        :return: self
        """
        if self.verbose:
            logger.info("=" * 60)
            logger.info("ScoreCard.fit() 开始训练")
            logger.info(f"输入数据类型: {type(X).__name__}, input_type={input_type}")

        if input_type not in ['woe', 'raw']:
            raise ValueError(f"input_type 必须是 'woe' 或 'raw'，当前为: {input_type}")

        # 转换为 DataFrame
        if not isinstance(X, pd.DataFrame):
            # 如果已配置binner或encoder，优先使用其特征名
            if self.binner is not None and hasattr(self.binner, 'splits_'):
                cols = list(self.binner.splits_.keys())
                if len(cols) == X.shape[1]:
                    X = pd.DataFrame(X, columns=cols)
            elif self.encoder is not None and hasattr(self.encoder, 'mapping_'):
                cols = list(self.encoder.mapping_.keys())
                if len(cols) == X.shape[1]:
                    X = pd.DataFrame(X, columns=cols)
            else:
                X = pd.DataFrame(X)

        # 处理 scorecardpipeline 风格：从 X 中提取 y
        if y is None and self.target is not None:
            if self.target in X.columns:
                y = X[self.target]
                X = X.drop(columns=[self.target])
                if self.verbose:
                    logger.info(f"从X中提取target列 '{self.target}' 作为y")
            else:
                raise ValueError(f"指定的target列 '{self.target}' 不存在于X中")

        if y is None:
            raise ValueError("必须提供y参数或在__init__中指定target参数")

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index if len(y) == len(X) else None)

        # 1. 验证并提取 pipeline 组件
        self._validate_pipeline_components()

        # 2. 根据 input_type 处理数据
        if input_type == 'raw':
            # 需要将原始数据转换为 WOE 数据
            if self.verbose:
                logger.info("将原始数据转换为 WOE 数据...")
            X = self._transform_to_woe(X)
        # else: input_type == 'woe', 直接使用输入数据

        # 3. 记录特征名
        self._feature_names = X.columns.tolist()
        if self.verbose:
            logger.info(f"特征数量: {len(self._feature_names)}")
            logger.info(f"特征列表: {self._feature_names}")

        # 4. 构建并训练/获取 LR 模型
        self.lr_model_ = self._build_lr_model()

        # 如果 LR 模型未训练，则训练
        if not hasattr(self.lr_model_, 'coef_'):
            if self.verbose:
                logger.info("训练 LR 模型...")
            self.lr_model_.fit(X, y, sample_weight=sample_weight)
        else:
            if self.verbose:
                logger.info("使用预训练的 LR 模型")

        if hasattr(self.lr_model_, 'ensure_positive_woe_coefficients'):
            self.lr_model_.ensure_positive_woe_coefficients(X)

        # 5. 生成评分卡规则
        self._generate_rules(X)

        # 6. 计算基础效应
        sub_scores = self._woe_to_score(X)
        self.base_effect_ = pd.Series(
            np.median(sub_scores, axis=0),
            index=self.feature_names_
        )
        
        self._is_fitted = True

        if self.verbose:
            logger.info(f"评分卡训练完成，总分 = 截距分数 + 各特征分数之和")
            logger.info(f"截距分数: {self.A_ - self.B_ * self.intercept_:.2f}")
            logger.info("=" * 60)

        return self

    def _generate_rules(self, X: pd.DataFrame):
        """生成评分卡规则.
        
        支持从分箱器获取完整的分箱信息，包括:
        - 数值特征的正常分箱区间
        - 缺失值分箱（标记为 'missing' 或 np.nan）
        - 特殊值分箱（标记为 'special'）
        """
        self.rules_ = {}

        for i, col in enumerate(self.feature_names_):
            coef = self.coef_[i]

            # 获取该特征的 WOE 值
            woe_values = None
            bins = None
            bin_labels = None
            values = None

            # 从 hscredit 的 binner 获取分箱信息
            if self.binner is not None and hasattr(self.binner, 'bin_tables_'):
                if col in self.binner.bin_tables_:
                    bin_table = self.binner.bin_tables_[col]
                    if '分档WOE值' in bin_table.columns:
                        woe_values = bin_table['分档WOE值'].values
                        if '分箱标签' in bin_table.columns:
                            bin_labels = bin_table['分箱标签'].values
                            bins = self._parse_bin_labels(bin_labels)

            # 从 toad/scp Combiner + WOETransformer 获取分箱和 WOE
            if woe_values is None and self._is_toad_like_combiner():
                ext_labels, ext_woe = self._extract_external_bin_info(col)
                if ext_labels is not None:
                    bin_labels = ext_labels
                    bins = self._parse_bin_labels(bin_labels)
                    if ext_woe is not None:
                        woe_values = ext_woe

            # 从 toad 的 encoder 获取（无 combiner 时兜底）
            if woe_values is None and self.encoder is not None and col in self.encoder:
                encoder_rule = self.encoder[col]
                if isinstance(encoder_rule, dict):
                    woe_raw = encoder_rule.get('woe')
                    values_raw = encoder_rule.get('value')
                    if woe_raw is not None and values_raw is not None:
                        val_arr = np.asarray(values_raw)
                        woe_arr = np.asarray(woe_raw)
                        sort_idx = np.argsort(val_arr)
                        woe_values = woe_arr[sort_idx]
                        values = val_arr[sort_idx]

            # 从训练数据推断
            if woe_values is None:
                unique_woe = X[col].dropna().unique()
                woe_values = sorted(unique_woe)
                bins = None

            woe_values = np.asarray(woe_values) * self._get_feature_woe_sign(i)

            # 最终兜底：如果仍然没有分箱标签，尝试从 binner 获取
            if bin_labels is None and self.binner is not None:
                if hasattr(self.binner, 'bin_tables_') and col in getattr(self.binner, 'bin_tables_', {}):
                    bt = self.binner.bin_tables_[col]
                    if '分箱标签' in bt.columns and '分档WOE值' in bt.columns:
                        bin_labels = bt['分箱标签'].values
                        woe_values = bt['分档WOE值'].values
                        bins = self._parse_bin_labels(bin_labels)

            # 计算每个 WOE 对应的分数
            scores = [self._woe_to_point(woe, coef) for woe in woe_values]

            self.rules_[col] = {
                'bins': bins,
                'bin_labels': bin_labels,
                'woe': woe_values,
                'scores': np.array(scores),
                'coef': coef,
                'values': values
            }

    def _parse_bin_labels(self, bin_labels: np.ndarray) -> list:
        """解析分箱标签为切分点或类别组.
        
        保留完整的分箱标签列表，包括:
        - 数值区间分箱
        - 缺失值分箱（标记为 'missing'）
        - 特殊值分箱（标记为 'special'）
        """
        parsed_labels = []
        
        for label in bin_labels:
            label_str = str(label)
            
            # 检查是否为缺失值或特殊值标记
            if label_str.lower() in ('missing', '缺失', 'nan', 'null'):

                parsed_labels.append('missing')
                continue
            elif label_str.lower() in ('special', '特殊'):
                parsed_labels.append('special')
                continue
            
            # 匹配数值区间: (a, b]、[a ~ b) 或 [负无穷 , 正无穷)
            if self._parse_interval_label(label_str) is not None:
                parsed_labels.append(label_str)
            else:
                # 类别值
                parsed_labels.append(label_str)
        
        return parsed_labels

    def _woe_to_point(self, woe: float, coef: float) -> float:
        """将 WOE 值转换为分数.
        
        基于 StandardScoreTransformer 的参数:
            score = -B_ * coef * woe
        """
        return -self.B_ * coef * woe

    def _woe_to_score(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """将 WOE 数据转换为分数矩阵.
        
        基于 StandardScoreTransformer 的参数:
            score_i = -B_ * coef_i * woe_i
            
        :param X: WOE 数据
        :param feature_names: 特征名列表，默认使用 self.feature_names_
        :return: 分数矩阵
        """
        if feature_names is None:
            feature_names = self.feature_names_

        X_effective = self._prepare_woe_for_scoring(X)
        
        scores = np.zeros((X.shape[0], len(feature_names)))
        
        for i, col in enumerate(feature_names):
            if col in X_effective.columns:
                coef = self.coef_[i]
                scores[:, i] = -self.B_ * coef * X_effective[col].values
        
        return scores

    def _score_flip_constant(self) -> float:
        """返回 ascending 方向用于镜像分数的常数。"""
        if self.lower is not None and self.upper is not None:
            return float(self.lower + self.upper)
        return float(2 * self.base_score)

    def _apply_score_direction(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将统一的 descending 线性分数转换为当前评分方向。"""
        values = np.asarray(scores, dtype=float)
        if self.direction_ == 'ascending':
            return self._score_flip_constant() - values
        return values

    def _remove_score_direction(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将当前方向的分数还原为统一的 descending 线性分数。"""
        return self._apply_score_direction(scores)

    def _finalize_scores(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """应用评分方向和边界裁剪。"""
        return self._clip_scores(self._apply_score_direction(scores))

    def transform(self, proba: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将概率转换为评分，要求评分卡已拟合或已加载."""
        self._check_fitted()
        return self._finalize_scores(super().transform(proba))

    def inverse_transform(self, scores: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """将评分反向转换为概率，要求评分卡已拟合或已加载."""
        self._check_fitted()
        return super().inverse_transform(self._remove_score_direction(scores))

    @staticmethod
    def _normalize_interval_bound(bound: Any) -> float:
        """将外部评分卡区间边界统一为浮点数，支持 toad/scp 的中英文无穷符号."""
        text = str(bound).strip().lower()
        text = text.replace('−', '-').replace('∞', 'inf')
        if text in ('-inf', '-infinity', '-np.inf', '负无穷', '负无穷大'):
            return -np.inf
        if text in ('+inf', 'inf', 'infinity', '+infinity', 'np.inf', '+np.inf', '正无穷', '正无穷大'):
            return np.inf
        return float(text)

    @staticmethod
    def _parse_interval_label(label: Any) -> Optional[Tuple[str, float, float, str]]:
        """解析常见评分卡区间标签.

        支持 hscredit ``[a, b)``、toad ``[a ~ b)``、scorecardpipeline
        ``[负无穷 , b)`` 等格式；非区间标签返回 ``None``。
        """
        label_str = str(label).strip()
        match = re.match(r'^([\[(])\s*(.*?)\s*(?:,|~)\s*(.*?)\s*([\])])$', label_str)
        if not match:
            return None

        left_bracket, lower_raw, upper_raw, right_bracket = match.groups()
        try:
            lower = ScoreCard._normalize_interval_bound(lower_raw)
            upper = ScoreCard._normalize_interval_bound(upper_raw)
        except (TypeError, ValueError):
            return None

        return left_bracket, lower, upper, right_bracket

    @staticmethod
    def _normalize_rule_label(label: Any) -> str:
        """标准化规则标签，便于离线规则映射."""
        label_str = str(label).strip()
        label_lower = label_str.lower()

        if label_lower in ('missing', '缺失值', '缺失', 'nan', 'null'):
            return 'missing'

        if label_lower in ('special', '特殊值', '特殊'):
            return 'special'

        if label_lower in ('else', 'other', 'others', '其他'):
            return 'else'

        if label_str.startswith(('(', '[')):
            interval = ScoreCard._parse_interval_label(label_str)
            if interval is not None:
                _, lower, upper, _ = interval
                lower_text = '-inf' if lower == -np.inf else f'{lower:g}'
                upper_text = '+inf' if upper == np.inf else f'{upper:g}'
                return f'interval:{lower_text}:{upper_text}'

            return re.sub(r',\s*', ', ', label_str)

        if ',' in label_str:
            return ','.join(part.strip() for part in label_str.split(',') if part.strip())

        return label_str

    def _bin_labels_to_score(self, X_bins: pd.DataFrame, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """将分箱标签数据直接映射为分数矩阵."""
        if feature_names is None:
            feature_names = self.feature_names_

        scores = np.zeros((X_bins.shape[0], len(feature_names)))

        for i, col in enumerate(feature_names):
            if col not in X_bins.columns or col not in self.rules_:
                continue

            rule = self.rules_[col]
            rule_labels = rule.get('bin_labels')
            if rule_labels is None:
                rule_labels = rule.get('bins', [])

            score_map = {
                self._normalize_rule_label(label): float(score)
                for label, score in zip(rule_labels, rule['scores'])
            }

            label_series = X_bins[col].map(self._normalize_rule_label)
            scores[:, i] = label_series.map(score_map).fillna(0.0).to_numpy()

        return scores

    def predict_score(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None,
        input_type: str = 'auto',
    ) -> np.ndarray:
        """预测评分（通过LR模型概率）。

        继承自 StandardScoreTransformer 的 predict_score 方法，
        但使用 ScoreCard 内部的 LR 模型来预测概率。

        可通过传入X或proba之一来获取评分。

        :param X: 特征矩阵，用于预测概率
        :param proba: 直接传入预测概率（正类概率）
        :param input_type: X 的输入类型，可选 ``'auto'`` / ``'raw'`` / ``'woe'``，默认 ``'auto'``
        :return: 评分数组

        **参考样例**

        >>> # 通过特征矩阵预测
        >>> scores = scorecard.predict_score(X_test_woe)

        >>> # 通过概率直接转换
        >>> proba = scorecard.lr_model_.predict_proba(X_test_woe)[:, 1]
        >>> scores = scorecard.predict_score(proba=proba)
        """
        if not self._skip_fit_check:
            self._check_fitted()

        if proba is None:
            if X is None:
                raise ValidationError("必须提供X或proba参数之一")
            proba = self.predict_proba(X, input_type=input_type)[:, 1]

        # 调用父类的transform方法将概率转换为评分
        return self.transform(proba)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'raw'
    ) -> np.ndarray:
        """预测评分（基于WOE特征的线性评分卡公式）。

        与 predict_score 不同，此方法使用评分卡公式：
        Score = A_ - B_ * (intercept + sum(coef_i * WOE_i))

        :param X: 输入数据
        :param input_type: 输入数据类型，可选：
            - 'raw': 原始数据，会进行 WOE 转换（默认）
            - 'woe': WOE 数据，直接使用
            - 'auto': 自动检测，通过数据特征推断输入类型

        input_type='auto' 时的判断逻辑：
            1. 数值范围检测：WOE数据通常取值范围在[-5, 5]之间，若所有数值列的min/max
               都在[-10, 10]范围内且主要分布集中在[-5, 5]，则判定为WOE数据
            2. 整数列检测：若存在int64/int32类型的列且唯一值数量>10，判定为原始数据
               （原始数据常包含年龄、收入等整数特征）
            3. 默认策略：当无法明确判断时，为安全起见默认按原始数据处理

            注意：auto检测基于启发式规则，对于边界情况（如原始数据本身就是小数值范围）
            可能误判。生产环境建议显式指定input_type='raw'或'woe'。

        :return: 评分数组

        :raises NotFittedError: 如果未传入lr_model且未调用fit方法
        """
        # 检查是否需要fit
        # 如果未传入预训练模型且未调用fit，则报错
        if not self._skip_fit_check:
            self._check_fitted()
        elif not hasattr(self, '_is_fitted') or not self._is_fitted:
            # 传入了预训练模型但未调用fit，使用预训练模型进行预测
            if self.verbose:
                logger.info("使用预训练模型进行预测（未调用fit）")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 检测输入数据类型
        is_woe_data = self._detect_input_type(X)
        uses_loaded_rule_scoring = self._should_use_loaded_rule_scoring(input_type, is_woe_data)

        if uses_loaded_rule_scoring:
            feature_names = self.feature_names_
            X_bins = self._transform_to_bins(X)
            X_bins = X_bins[feature_names]
            sub_scores = self._bin_labels_to_score(X_bins, feature_names)
        else:
            if input_type == 'auto':
                # 自动检测
                if is_woe_data:
                    if self.verbose:
                        logger.info("检测到输入为 WOE 数据，直接使用")
                    X_woe = X
                else:
                    if self.verbose:
                        logger.info("检测到输入为原始数据，进行 WOE 转换")
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
            # 如果传入了预训练模型但未fit，使用输入数据的列名
            if self._skip_fit_check and not getattr(self, '_is_fitted', False):
                # 未fit，直接使用输入数据的列
                feature_names = X_woe.columns.tolist()
            else:
                # 已fit，使用保存的特征名
                feature_names = self.feature_names_
                X_woe = X_woe[feature_names]

            # 计算每个特征的分数
            sub_scores = self._woe_to_score(X_woe, feature_names)

        # 总分 = 截距分数 + 各特征分数之和
        # intercept_score = A_ - B_ * intercept
        intercept_score = self.A_ - self.B_ * self.intercept_
        total_score = intercept_score + sub_scores.sum(axis=1)

        return self._finalize_scores(total_score)

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

    def _prepare_lr_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'auto',
    ) -> pd.DataFrame:
        """按显式输入类型准备底层 LR 模型所需的 WOE 特征."""
        if input_type not in ['auto', 'raw', 'woe']:
            raise ValueError(f"input_type 必须是 'auto'/'raw'/'woe' 之一，当前为: {input_type}")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if input_type == 'raw':
            X_woe = self._transform_to_woe(X)
        elif input_type == 'woe':
            X_woe = X
        else:
            X_woe = X if self._detect_input_type(X) else self._transform_to_woe(X)

        if not (self._skip_fit_check and not getattr(self, '_is_fitted', False)):
            feature_names = self.feature_names_
            X_woe = X_woe[feature_names]
        return X_woe

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'auto',
    ) -> np.ndarray:
        """预测样本属于各类别的概率（使用底层 LR 模型）。

        支持显式指定输入类型。若为原始数据则先经 binner/encoder 转为 WOE，
        再交由逻辑回归模型输出概率。与 :meth:`predict`（输出分数）相对，本方法输出概率。

        :param X: 输入数据，原始特征或 WOE 数据，DataFrame 或 ndarray
        :param input_type: 输入类型 ``'auto'`` / ``'raw'`` / ``'woe'``，默认 ``'auto'``
        :return: 形状 ``(n_samples, 2)`` 的概率数组，第 1 列为坏样本（正类）概率
        :raises NotFittedError: 评分卡尚未拟合且未传入预训练 LR 模型时

        **参考样例**

        >>> proba_bad = scorecard.predict_proba(X_test)[:, 1]   # 坏样本概率
        """
        self._check_fitted()
        lr_model = self.lr_model_ if hasattr(self, 'lr_model_') and self.lr_model_ is not None else self.lr_model
        if lr_model is None:
            raise NotFittedError("未找到LR模型，当前评分卡无法 predict_proba")

        X_woe = self._prepare_lr_input(X, input_type=input_type)
        return lr_model.predict_proba(X_woe)

    def scorecard_scale(self) -> pd.DataFrame:
        """输出评分卡基础配置（刻度参数）.

        :return: DataFrame，包含 base_odds/base_score/rate/pdo 及推导出的 A、B 刻度参数
        """
        self._check_fitted()

        base_odds_remark = (
            "好坏比（好:坏），内部换算实际 odds = 1/base_odds"
            if self.base_odds >= 1 else "坏样本率 / 坏好比，直接作为实际 odds"
        )
        formula_info = self.score_formula()
        score_direction = "越大越好" if self.direction_ == "descending" else "越小越好"
        return pd.DataFrame([
            {"刻度项": "base_odds", "刻度值": self.base_odds, "备注": base_odds_remark},
            {"刻度项": "base_score", "刻度值": self.base_score,
             "备注": "基础 odds 对应的分数"},
            {"刻度项": "rate", "刻度值": self.rate,
             "备注": "odds 增加的倍率"},
            {"刻度项": "pdo", "刻度值": self.pdo,
             "备注": f"odds 增加 {self.rate} 倍时分数变化量"},
            {"刻度项": "B", "刻度值": self.B_,
             "备注": f"pdo / ln({self.rate})"},
            {"刻度项": "A", "刻度值": self.A_,
             "备注": "base_score + B * ln(实际odds)，实际odds 见 base_odds 备注"},
            {"刻度项": "direction", "刻度值": score_direction,
             "备注": f"direction={self.direction_}"},
            {"刻度项": "formula", "刻度值": formula_info["公式"],
             "备注": "评分卡转换公式，odds = P(坏) / P(好)（同 score_formula 的「公式」）"},
        ])

    def score_formula(self, decimal: int = 4) -> Dict[str, Any]:
        """输出评分卡的评分转换公式（人类可读 + 可编程使用）.

        返回标准评分卡公式 ``Score = A - B × ln(odds)`` 的各项参数与等价的
        WOE 线性表达式，便于复核、文档化与离线部署。

        :param decimal: 公式中数值保留的小数位数，默认 4
        :return: 包含公式字符串与参数的字典，键包括
            ``公式``/``A``/``B``/``截距分数``/``base_odds``/``base_score``/
            ``pdo``/``rate``/``direction``/``WOE线性公式``
        """
        self._check_fitted()

        A = round(float(self.A_), decimal)
        B = round(float(self.B_), decimal)
        intercept_score = round(float(self.A_ - self.B_ * self.intercept_), decimal)

        formula = f"Score = {A} - {B} × ln(odds)，其中 odds = P(坏) / P(好)"
        woe_formula = (
            f"Score = 截距分数({intercept_score}) "
            f"+ Σ(-B × coef_i × WOE_i)，B = {B}"
        )

        return {
            "公式": formula,
            "WOE线性公式": woe_formula,
            "A": A,
            "B": B,
            "截距分数": intercept_score,
            "base_odds": self.base_odds,
            "base_score": self.base_score,
            "pdo": self.pdo,
            "rate": self.rate,
            "direction": self.direction_,
        }

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None,
        decimal: int = 4
    ) -> pd.DataFrame:
        """输出评分卡分箱信息及其对应的分数.
        
        支持从分箱器获取完整的分箱信息，包括:
        - 基础分（截距项对应的分数）
        - 数值特征分箱（区间格式）
        - 类别特征分箱
        - 缺失值分箱（标记为 'missing'）
        - 特殊值分箱（标记为 'special'）
        
        参考 scorecardpipeline 的实现方式，确保与分箱器格式兼容。

        :param feature_map: 特征名到中文含义的映射字典
        :param decimal: 分数保留小数位数，默认 2
        """
        self._check_fitted()

        if feature_map is None:
            feature_map = {}

        rows = []
        
        # 首先添加基础分（截距项）
        # 截距分数 = A_ - B_ * intercept
        intercept_score = self.A_ - self.B_ * self.intercept_
        rows.append({
            '变量名称': '基础分',
            '变量含义': '截距项（基准分数）',
            '变量分箱': '-',
            '对应分数': round(float(intercept_score), decimal),
            'WOE值': None
        })
        
        # 使用 lr 模型的特征名（与训练时一致）
        feature_names = self.feature_names_
        
        for col in feature_names:
            if col not in self.rules_:
                continue
                
            rule = self.rules_[col]
            scores = rule['scores']

            # 优先使用 bin_labels（完整的分箱标签）
            bin_labels = rule.get('bin_labels')
            bins = rule.get('bins')
            # WOE 值可能缺失（如离线规则加载后无 woe），单独处理，
            # 不能放进 zip 否则空/短的 woe 会把整张分箱表截断为 0 行
            woe_values = rule.get('woe')
            woe_list = list(woe_values) if woe_values is not None else []

            # 确定要使用的分箱标签
            if bin_labels is not None and len(bin_labels) > 0:
                labels_to_use = bin_labels
            elif bins is not None and len(bins) > 0:
                labels_to_use = bins
            elif woe_list:
                # 无分箱标签，使用 WOE 值格式化为标签
                labels_to_use = [f'WOE: {w:.4f}' for w in woe_list]
            else:
                # 无标签也无 WOE，用箱序号兜底
                labels_to_use = [f'箱{i}' for i in range(len(scores))]

            if len(labels_to_use) != len(scores):
                # 如果标签和分数数量不匹配，重新生成标签
                labels_to_use = self._format_bin_labels(bins if bins else labels_to_use, len(scores))

            # 处理每个分箱（以 标签/分数 数量为准，WOE 按位置取，缺失则为 None）
            for idx, (bin_label, score) in enumerate(zip(labels_to_use, scores)):
                # 格式化特殊标签
                display_label = self._format_bin_display(bin_label)
                woe = woe_list[idx] if idx < len(woe_list) else None

                rows.append({
                    '变量名称': col,
                    '变量含义': feature_map.get(col, ''),
                    '变量分箱': display_label,
                    '对应分数': round(float(score), decimal),
                    'WOE值': round(float(woe), 4) if woe is not None and not pd.isna(woe) else None
                })

        if not rows:
            return pd.DataFrame(columns=['变量名称', '变量含义', '变量分箱', '对应分数', 'WOE值'])
            
        return pd.DataFrame(rows)
    
    def _format_bin_labels(self, bins, n_scores):
        """根据分箱信息格式化为显示标签."""
        labels = []
        
        for i in range(n_scores):
            if i < len(bins):
                bin_val = bins[i]
                if isinstance(bin_val, str):
                    if bin_val.lower() in ('missing', '缺失'):
                        labels.append('缺失值')
                    elif bin_val.lower() in ('special', '特殊'):
                        labels.append('特殊值')
                    else:
                        labels.append(bin_val)
                else:
                    labels.append(str(bin_val))
            else:
                labels.append(f'箱{i}')
        
        return labels
    
    def _format_bin_display(self, bin_label):
        """格式化分箱标签用于显示."""
        if isinstance(bin_label, str):
            if bin_label.lower() in ('missing', '缺失'):
                return '缺失值'
            elif bin_label.lower() in ('special', '特殊'):
                return '特殊值'
            elif bin_label.lower() in ('nan', 'null'):

                return '缺失值'
        return str(bin_label)

    @staticmethod
    def _format_score_interval(interval: Any, decimal: int = 4, closed: Optional[str] = None) -> str:
        """格式化评分区间，避免 pandas Interval 暴露浮点尾差."""
        if hasattr(interval, 'left') and hasattr(interval, 'right'):
            left_bracket = '[' if getattr(interval, 'closed_left', False) else '('
            right_bracket = ']' if getattr(interval, 'closed_right', False) else ')'
            if closed == 'both':
                left_bracket, right_bracket = '[', ']'
            return f"{left_bracket}{float(interval.left):.{decimal}f}, {float(interval.right):.{decimal}f}{right_bracket}"

        return str(interval)

    def score_to_bad_rate_table(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        method: str = 'quantile',
        score_decimal: int = 4,
    ) -> pd.DataFrame:
        """生成评分分箱对应坏样本率、Odds、KS 的对照表（评分卡校验/划档常用）。

        将分数切成 ``n_bins`` 档，逐档统计样本数、坏样本率、Odds，并累计计算 KS，
        用于检查"分数越高坏率越低"的单调性与整体区分度。

        :param scores: 模型输出的分数数组（如 :meth:`predict` 的结果）
        :param y: 对应的真实标签数组（0=好/1=坏），与 ``scores`` 等长
        :param n_bins: 分数分档数量，默认为 ``10``
        :param method: 分档方式，默认为 ``'quantile'``。可取以下枚举值：

            - ``'quantile'``：等频分档（每档样本量大致相等），用 ``pd.qcut``
            - 其他值（如 ``'uniform'``）：等距分档（按分数范围等宽），用 ``pd.cut``

        :param score_decimal: 评分区间边界保留小数位数，默认 ``4``，用于消除浮点显示尾差
        :return: DataFrame，列含 ``评分区间`` / ``样本数`` / ``坏样本数`` / ``坏样本率`` /
            ``好样本数`` / ``Odds`` / ``累计好样本占比`` / ``累计坏样本占比`` / ``KS``

        **参考样例**

        >>> s = scorecard.predict(X_test)
        >>> scorecard.score_to_bad_rate_table(s, y_test, n_bins=10)
        """
        df = pd.DataFrame({'score': scores, 'y': y})

        if len(df) == 0:
            return pd.DataFrame(columns=[
                '评分区间', '样本数', '坏样本数', '坏样本率',
                '好样本数', 'Odds', '累计好样本占比', '累计坏样本占比', 'KS'
            ])

        if df['score'].nunique(dropna=True) <= 1:
            score_value = float(df['score'].dropna().iloc[0]) if df['score'].notna().any() else np.nan
            df['score_bin'] = f"[{score_value:.{score_decimal}f}, {score_value:.{score_decimal}f}]"
        elif method == 'quantile':
            df['score_bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop')
        else:
            df['score_bin'] = pd.cut(df['score'], bins=n_bins)
        
        stats = df.groupby('score_bin', observed=False).agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        
        stats.columns = ['评分区间', '样本数', '坏样本数', '坏样本率']
        stats['评分区间'] = stats['评分区间'].apply(
            lambda interval: self._format_score_interval(interval, decimal=score_decimal)
        )
        stats['好样本数'] = stats['样本数'] - stats['坏样本数']
        stats['Odds'] = stats['好样本数'] / stats['坏样本数'].replace(0, np.nan)
        stats['累计好样本占比'] = stats['好样本数'].cumsum() / stats['好样本数'].sum()
        stats['累计坏样本占比'] = stats['坏样本数'].cumsum() / stats['坏样本数'].sum()
        stats['KS'] = abs(stats['累计坏样本占比'] - stats['累计好样本占比'])
        
        stats['坏样本率'] = stats['坏样本率'].apply(lambda x: f'{x:.2%}')
        stats['Odds'] = stats['Odds'].apply(lambda x: f'{x:.2f}' if x != np.inf else 'inf')
        stats['KS'] = stats['KS'].apply(lambda x: f'{x:.4f}')
        
        return stats

    def save_pickle(
        self,
        file: str,
        engine: str = 'joblib',
        compression: Optional[str] = None,
        compression_level: Optional[int] = None
    ) -> str:
        """保存模型.

        使用 utils.io.save_pickle 进行持久化存储，支持多种序列化引擎和压缩格式。

        :param file: 文件路径
        :param engine: 序列化引擎，可选 'joblib'/'pickle'/'dill'/'cloudpickle'，默认 'joblib'
        :param compression: 压缩格式，可选 'gzip'/'bz2'/'xz'/'lz4'/'zstd'，默认 None
        :param compression_level: 压缩级别（1-9），默认 None
        :return: 保存的文件路径
        """
        from ....utils.io import save_pickle as _save_pickle

        file_dir = os.path.dirname(file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        _save_pickle(
            self,
            file,
            engine=engine,
            compression=compression,
            compression_level=compression_level
        )

        logger.info("模型已保存至: %s", file)
        return file

    @classmethod
    def load_pickle(
        cls,
        file: str,
        engine: str = 'auto',
        compression: Optional[str] = None
    ) -> 'ScoreCard':
        """加载模型.

        使用 utils.io.load_pickle 进行持久化读取，支持多种序列化引擎和压缩格式。

        :param file: 文件路径
        :param engine: 序列化引擎，可选 'auto'/'joblib'/'pickle'/'dill'/'cloudpickle'，默认 'auto'
        :param compression: 压缩格式，可选 'gzip'/'bz2'/'xz'/'lz4'/'zstd'，默认 None（自动检测）
        :return: 加载的 ScoreCard 模型实例
        """
        from ....utils.io import load_pickle as _load_pickle

        obj = _load_pickle(file, engine=engine, compression=compression)
        if not isinstance(obj, cls):
            raise TypeError(f"加载的对象类型为 {type(obj).__name__}，不是 {cls.__name__}")
        return obj

    def _get_deployment_base_score_and_sign(self) -> Tuple[float, float]:
        """获取部署导出时使用的基础分和分箱分数符号."""
        intercept_score = float(self.A_ - self.B_ * self.intercept_)
        score_sign = 1.0
        base_score = intercept_score

        if self.direction_ == 'ascending':
            flip_constant = self.lower + self.upper if self.lower is not None and self.upper is not None else 2 * self.base_score
            base_score = float(flip_constant - intercept_score)
            score_sign = -1.0

        return base_score, score_sign

    def _get_deployment_rules(self, decimal: int) -> Dict[str, List[Tuple[Any, float]]]:
        """获取部署导出时使用的精确规则定义."""
        deployment_rules: Dict[str, List[Tuple[Any, float]]] = {}
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}
        cat_bins = getattr(self.binner, '_cat_bins_', {}) if self.binner is not None else {}

        for feature in self.feature_names_:
            rule = self.rules_.get(feature)
            if not rule:
                continue

            scores = rule['scores']
            descriptors = None

            if feature_types.get(feature) == 'categorical' and feature in cat_bins and len(cat_bins[feature]) == len(scores):
                descriptors = cat_bins[feature]
            elif rule.get('bin_labels') is not None and len(rule['bin_labels']) == len(scores):
                descriptors = rule['bin_labels']
            elif rule.get('bins') is not None and len(rule['bins']) == len(scores):
                descriptors = rule['bins']

            if descriptors is None:
                continue

            deployment_rules[feature] = [
                (descriptor, round(float(score), decimal))
                for descriptor, score in zip(descriptors, scores)
            ]

        return deployment_rules

    def export_pmml(
        self,
        pmml_file: str = 'scorecard.pmml',
        decimal: int = 12,
        debug: bool = False
    ):
        """导出 PMML 文件.

        :param pmml_file: PMML 文件保存路径，默认 'scorecard.pmml'
        :param decimal: 特征子分保留小数位数，默认 12，确保 PMML 与 predict 精度一致
        :param debug: 是否返回中间对象进行调试，默认 False
        :return: debug=True 时返回 PMMLPipeline，否则返回 None
        """
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
            from sklearn2pmml.preprocessing import (
                AggregateTransformer,
                ConcatTransformer,
                ExpressionTransformer,
                LookupTransformer,
            )
        except ImportError:
            warnings.warn(
                "PMML 导出需要安装依赖: pip install hscredit[pmml] 或安装 sklearn2pmml。"
                "当前环境中相关依赖不可用，PMML 导出已跳过。"
            )
            return

        self._check_fitted()

        base_score, score_sign = self._get_deployment_base_score_and_sign()
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}

        mapper = []
        samples = {}

        deployment_rules = self._get_deployment_rules(decimal=decimal)

        for var, bins in deployment_rules.items():
            feature_type = feature_types.get(var)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0

            if bins is None or len(bins) == 0:
                continue

            if feature_type == 'categorical':
                lookup_mapping, missing_score = self._build_pmml_categorical_lookup_mapping(
                    bins,
                    special_codes=special_codes,
                )
                missing_replacement = '__MISSING__'
                if missing_score is not None:
                    lookup_mapping[missing_replacement] = float(missing_score)

                domain = CategoricalDomain(
                    with_data=False,
                    invalid_value_treatment='as_is',
                    missing_value_treatment='as_value' if missing_score is not None else 'as_is',
                    missing_value_replacement=missing_replacement if missing_score is not None else None,
                )
                transformer = LookupTransformer(lookup_mapping, default_value=float(default_score))
                input_columns = [var]
                transformer_steps = [('prepare', ConcatTransformer())]
            else:
                expression_string = self._build_pmml_expression_from_rules(
                    bins,
                    default_score=default_score,
                    special_codes=special_codes,
                )
                domain = ContinuousDomain(
                    with_data=False,
                    invalid_value_treatment='as_is',
                    missing_value_treatment='as_is',
                )
                transformer = ExpressionTransformer(expression_string)
                # ExpressionTransformer treats a single-column DataFrame as 1D. Aggregating a
                # single numeric value with ``min`` is an identity operation that keeps the
                # intermediate output two-dimensional without duplicating the source column.
                input_columns = [var]
                transformer_steps = [('prepare', AggregateTransformer('min'))]

            mapper.append((
                f'score_{len(mapper)}',
                Pipeline(
                    [('domain', domain)]
                    + transformer_steps
                    + [('score', Alias(transformer, f'__score_{var}', prefit=True))]
                ),
                input_columns,
            ))

            if feature_type == 'categorical':
                sample_value = None
                for lookup_key in lookup_mapping:
                    if lookup_key != '__MISSING__':
                        sample_value = lookup_key
                        break
                samples[var] = [sample_value if sample_value is not None else 'UNKNOWN'] * 20
            else:
                samples[var] = np.random.random(20) * 100

        if not mapper:
            raise ValueError("没有有效的评分规则可以导出")

        scorecard_mapper = ColumnTransformer(
            mapper,
            remainder='drop',
            verbose_feature_names_out=False,
        )

        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=True)),
        ])

        sample_df = pd.DataFrame(samples)
        sample_y = pd.Series(np.random.randint(0, 2, 20), name='score')
        pipeline.fit(sample_df, sample_y)
        
        pipeline.named_steps['scorecard'].coef_ = np.full(len(mapper), score_sign, dtype=float)
        pipeline.named_steps['scorecard'].intercept_ = float(base_score)

        import tempfile

        destination = os.path.abspath(os.fspath(pmml_file))
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        file_descriptor, temporary_file = tempfile.mkstemp(
            prefix=f'.{os.path.basename(destination)}.',
            suffix='.pmml',
            dir=destination_dir,
        )
        os.close(file_descriptor)
        os.unlink(temporary_file)

        try:
            try:
                sklearn2pmml(pipeline, temporary_file, with_repr=True, debug=debug)
            except TypeError as exc:
                # sklearn2pmml 在部分 notebook 环境会在 Java 已成功写出文件后，因解码
                # 子进程输出抛出 NoneType/len TypeError。只接受本次调用生成的临时文件，
                # 绝不能把目标路径中上一次残留的文件误判为成功。
                temporary_exists = os.path.exists(temporary_file) and os.path.getsize(temporary_file) > 0
                is_output_decode_bug = "NoneType" in str(exc) and "len()" in str(exc)
                if not (temporary_exists and is_output_decode_bug):
                    raise
                warnings.warn(
                    "sklearn2pmml reported a subprocess output decoding error after generating the PMML file; "
                    "continuing with the exported artifact.",
                    RuntimeWarning,
                )

            if not os.path.exists(temporary_file) or os.path.getsize(temporary_file) == 0:
                raise RuntimeError("sklearn2pmml 未生成有效的 PMML 文件")

            if self.clip and (self.lower is not None or self.upper is not None):
                self._apply_pmml_score_clipping(temporary_file)

            os.replace(temporary_file, destination)
        finally:
            if os.path.exists(temporary_file):
                os.unlink(temporary_file)
        logger.info("PMML 文件已导出至: %s", pmml_file)

        if debug:
            return pipeline

    def _apply_pmml_score_clipping(self, pmml_file: str) -> None:
        """在 PMML 输出层应用与 :meth:`predict` 一致的分数边界."""
        import xml.etree.ElementTree as ET

        tree = ET.parse(pmml_file)
        root = tree.getroot()
        namespace = root.tag.partition('}')[0].lstrip('{') if root.tag.startswith('{') else ''
        if namespace:
            ET.register_namespace('', namespace)

        def qualified(name: str) -> str:
            return f'{{{namespace}}}{name}' if namespace else name

        model = root.find(f'.//{qualified("RegressionModel")}')
        if model is None:
            raise RuntimeError("PMML 中未找到可应用评分边界的 RegressionModel")

        output = model.find(qualified('Output'))
        if output is None:
            output = ET.Element(qualified('Output'))
            children = list(model)
            mining_schema = model.find(qualified('MiningSchema'))
            insert_at = children.index(mining_schema) + 1 if mining_schema is not None else 0
            model.insert(insert_at, output)

        reserved_names = {'__hscredit_raw_score', 'predicted_score'}
        for field in list(output):
            if field.get('name') in reserved_names:
                output.remove(field)

        ET.SubElement(
            output,
            qualified('OutputField'),
            {
                'name': '__hscredit_raw_score',
                'optype': 'continuous',
                'dataType': 'double',
                'feature': 'predictedValue',
                'targetField': 'score',
            },
        )
        clipped_field = ET.SubElement(
            output,
            qualified('OutputField'),
            {
                'name': 'predicted_score',
                'optype': 'continuous',
                'dataType': 'double',
                'feature': 'transformedValue',
            },
        )

        expression = ET.Element(qualified('FieldRef'), {'field': '__hscredit_raw_score'})
        if self.lower is not None:
            lower_apply = ET.Element(qualified('Apply'), {'function': 'max'})
            lower_constant = ET.SubElement(lower_apply, qualified('Constant'), {'dataType': 'double'})
            lower_constant.text = repr(float(self.lower))
            lower_apply.append(expression)
            expression = lower_apply
        if self.upper is not None:
            upper_apply = ET.Element(qualified('Apply'), {'function': 'min'})
            upper_constant = ET.SubElement(upper_apply, qualified('Constant'), {'dataType': 'double'})
            upper_constant.text = repr(float(self.upper))
            upper_apply.append(expression)
            expression = upper_apply
        clipped_field.append(expression)

        tree.write(pmml_file, encoding='utf-8', xml_declaration=True)

    def export_deployment_code(
        self,
        language: str = 'python',
        output_file: Optional[str] = None,
        function_name: str = 'calculate_score',
        decimal: int = 12,
    ) -> str:
        """导出评分卡部署代码.

        支持生成 SQL、Python、Java 格式的评分卡计算代码，可直接用于生产部署。

        :param language: 目标语言，可选 'sql'/'python'/'java'，默认 'python'
        :param output_file: 输出文件路径，为 None 时仅返回字符串
        :param function_name: 函数/存储过程名称，默认 'calculate_score'
        :param decimal: 分数保留小数位数，默认 4
        :return: 生成的部署代码字符串

        **参考样例**

        >>> sc = ScoreCard(...)
        >>> sc.fit(X_train, y_train)
        >>> # 生成 SQL
        >>> sql = sc.export_deployment_code(language='sql', output_file='scorecard.sql')
        >>> # 生成 Python
        >>> py = sc.export_deployment_code(language='python', output_file='scorecard.py')
        """
        self._check_fitted()
        language_normalized = language.lower()
        if language_normalized in ('python', 'java'):
            self._validate_deployment_function_name(function_name, language_normalized)

        card = self._get_deployment_rules(decimal=decimal)
        base_score, score_sign = self._get_deployment_base_score_and_sign()
        base_score = round(float(base_score), decimal)

        if language_normalized == 'sql':
            code = self._generate_sql(card, base_score, function_name, score_sign=score_sign)
        elif language_normalized == 'python':
            code = self._generate_python(card, base_score, function_name, score_sign=score_sign)
        elif language_normalized == 'java':
            code = self._generate_java(card, base_score, function_name, score_sign=score_sign)
        else:
            raise ValueError(f"不支持的语言: {language}，可选: sql/python/java")

        if output_file:
            import os
            dir_path = os.path.dirname(output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(code)

        return code

    @staticmethod
    def _validate_deployment_function_name(function_name: str, language: str) -> None:
        """验证会直接写入生成代码的函数名."""
        name = str(function_name)
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) is None:
            raise ValueError(f"{language} 函数名必须是合法标识符: {function_name!r}")
        if language == 'python':
            import keyword

            if keyword.iskeyword(name):
                raise ValueError(f"python 函数名不能使用关键字: {function_name!r}")
        elif language == 'java':
            java_keywords = {
                'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
                'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final',
                'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int',
                'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public',
                'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
                'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null',
            }
            if name in java_keywords:
                raise ValueError(f"java 函数名不能使用关键字: {function_name!r}")

    @staticmethod
    def _format_deployment_score(score: float, score_sign: float) -> float:
        """格式化部署导出时的特征分数符号."""
        adjusted = float(score_sign) * float(score)
        return 0.0 if adjusted == -0.0 else adjusted

    def _generate_sql(self, card: dict, base_score: float, func_name: str, score_sign: float = 1.0) -> str:
        """生成 SQL CASE WHEN 评分卡代码."""
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}

        expression_lines = [f"    {base_score}"]

        for feature, bins in card.items():
            feature_type = feature_types.get(feature)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0
            expression_lines.append(f"    + CASE")
            for bin_descriptor, score in bins:
                cond = self._bin_label_to_sql_condition(feature, bin_descriptor, special_codes=special_codes)
                adjusted_score = self._format_deployment_score(score, score_sign)
                expression_lines.append(f"        WHEN {cond} THEN {adjusted_score}")
            expression_lines.append(f"        ELSE {self._format_deployment_score(default_score, score_sign)}")
            feature_comment = str(feature).replace('\r', ' ').replace('\n', ' ')
            expression_lines.append(f"      END  -- {feature_comment}")

        raw_query = ["SELECT", *expression_lines, "    AS raw_score", "FROM your_table"]
        lines = ["-- 评分卡 SQL 部署代码（自动生成）", f"-- base_score = {base_score}", ""]
        if not self.clip or (self.lower is None and self.upper is None):
            raw_query[-2] = "    AS score"
            lines.extend(raw_query)
            lines[-1] += ";"
            return '\n'.join(lines)

        lines.extend(["SELECT", "    CASE"])
        if self.lower is not None:
            lines.append(f"        WHEN raw_score < {float(self.lower)!r} THEN {float(self.lower)!r}")
        if self.upper is not None:
            lines.append(f"        WHEN raw_score > {float(self.upper)!r} THEN {float(self.upper)!r}")
        lines.extend(["        ELSE raw_score", "    END AS score", "FROM ("])
        lines.extend(f"    {line}" for line in raw_query)
        lines.append(") AS scorecard_raw;")
        return '\n'.join(lines)

    def _generate_python(self, card: dict, base_score: float, func_name: str, score_sign: float = 1.0) -> str:
        """生成 Python 评分卡函数代码."""
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}
        feature_names = list(self.feature_names_)
        direction = getattr(self, 'direction_', self.direction)

        lines = [
            f'"""评分卡 Python 部署代码（自动生成）"""',
            f'import numpy as np',
            f'import pandas as pd',
            f'',
            f'# 评分卡元数据，可在 import/exec 部署代码后直接读取',
            f'feature_name_in_ = {feature_names!r}',
            f'feature_names_in_ = feature_name_in_',
            f'n_features_in_ = {len(feature_names)}',
            f'pdo = {self.pdo!r}',
            f'rate = {self.rate!r}',
            f'base_odds = {self.base_odds!r}',
            f'base_score = {self.base_score!r}',
            f'step = {self.step!r}',
            f'lower = {self.lower!r}',
            f'upper = {self.upper!r}',
            f'direction = {direction!r}',
            f'decimal = {self.decimal!r}',
            f'A_ = {float(self.A_)!r}',
            f'B_ = {float(self.B_)!r}',
            f'intercept_score = {base_score!r}',
            f'deployment_base_score = intercept_score',
            f'score_sign = {float(score_sign)!r}',
            f'',
            f'def {func_name}(row: dict) -> float:',
            f'    """计算单条样本的评分卡分数.',
            f'',
            f'    :param row: 样本特征字典',
            f'    :return: 评分',
            f'    """',
            f'    score = {base_score}  # base_score',
        ]

        for feature, bins in card.items():
            feature_type = feature_types.get(feature)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0
            lines.append(f'')
            feature_comment = str(feature).replace('\r', ' ').replace('\n', ' ')
            lines.append(f'    # {feature_comment}')
            lines.append(f'    val = row.get({feature!r})')
            first = True
            for bin_descriptor, sc in bins:
                prefix = 'if' if first else 'elif'
                cond = self._bin_label_to_python_condition('val', bin_descriptor, special_codes=special_codes)
                adjusted_score = self._format_deployment_score(sc, score_sign)
                lines.append(f'    {prefix} {cond}:')
                lines.append(f'        score += {adjusted_score}')
                first = False
            lines.append(f'    else:')
            lines.append(f'        score += {self._format_deployment_score(default_score, score_sign)}')

        if self.clip and self.lower is not None:
            lines.append(f'    score = max({float(self.lower)!r}, score)')
        if self.clip and self.upper is not None:
            lines.append(f'    score = min({float(self.upper)!r}, score)')
        lines.append(f'')
        lines.append(f'    return score')
        lines.append(f'')
        lines.append(f'')
        lines.append(f'def batch_{func_name}(df: pd.DataFrame) -> pd.Series:')
        lines.append(f'    """批量计算评分."""')
        lines.append(f'    return df.apply(lambda row: {func_name}(row.to_dict()), axis=1)')
        return '\n'.join(lines)

    def _generate_java(self, card: dict, base_score: float, func_name: str, score_sign: float = 1.0) -> str:
        """生成 Java 评分卡方法代码."""
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}

        lines = [
            f'import java.util.Map;',
            f'',
            f'/**',
            f' * 评分卡 Java 部署代码（自动生成）',
            f' */',
            f'public class ScoreCard {{',
            f'',
            f'    public static double {func_name}(Map<String, Object> row) {{',
            f'        double score = {base_score};  // base_score',
        ]

        for feature_index, (feature, bins) in enumerate(card.items()):
            feature_type = feature_types.get(feature)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0
            java_var = self._safe_java_var(f'feature_{feature_index}_{feature}')
            lines.append(f'')
            feature_comment = str(feature).replace('\r', ' ').replace('\n', ' ')
            lines.append(f'        // {feature_comment}')
            lines.append(
                f'        Object {java_var} = row.get({self._java_string_literal(feature)});'
            )
            first = True
            for bin_descriptor, sc in bins:
                prefix = 'if' if first else 'else if'
                cond = self._bin_label_to_java_condition(
                    java_var, bin_descriptor, special_codes=special_codes
                )
                adjusted_score = self._format_deployment_score(sc, score_sign)
                lines.append(f'        {prefix} ({cond}) {{')
                lines.append(f'            score += {adjusted_score};')
                lines.append(f'        }}')
                first = False
            lines.append(f'        else {{')
            lines.append(f'            score += {self._format_deployment_score(default_score, score_sign)};')
            lines.append(f'        }}')

        if self.clip and self.lower is not None:
            lines.append(f'        score = Math.max({float(self.lower)!r}, score);')
        if self.clip and self.upper is not None:
            lines.append(f'        score = Math.min({float(self.upper)!r}, score);')
        lines.append(f'')
        lines.append(f'        return score;')
        lines.append(f'    }}')
        lines.append(f'}}')
        return '\n'.join(lines)

    @staticmethod
    def _bin_label_to_sql_condition(feature: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 SQL CASE WHEN 条件."""
        feature_sql = ScoreCard._quote_sql_identifier(feature)
        if isinstance(label, (list, np.ndarray)):
            values = [value.item() if isinstance(value, np.generic) else value for value in label if not pd.isna(value)]
            contains_missing = any(pd.isna(value) for value in label)
            if not values:
                return f"{feature_sql} IS NULL"
            if len(values) > 1:
                value_condition = f"{feature_sql} IN ({', '.join(ScoreCard._sql_literal(value) for value in values)})"
            else:
                value_condition = f"{feature_sql} = {ScoreCard._sql_literal(values[0])}"
            if contains_missing:
                return f"({feature_sql} IS NULL OR {value_condition})"
            return value_condition

        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{feature_sql} IS NULL"
        if label in ('特殊值', 'special'):
            if special_codes:
                comparisons = []
                for code in special_codes:
                    if pd.isna(code):
                        comparisons.append(f"{feature_sql} IS NULL")
                    else:
                        comparisons.append(f"{feature_sql} = {ScoreCard._sql_literal(code)}")
                return ' OR '.join(comparisons)
            return '1=0 /* no special codes configured */'
        interval = ScoreCard._parse_interval_label(label)
        if interval is not None:
            left_bracket, lower, upper, right_bracket = interval
            conds = [f"{feature_sql} IS NOT NULL"]
            if np.isfinite(lower):
                op = '>=' if left_bracket == '[' else '>'
                conds.append(f"{feature_sql} {op} {float(lower)!r}")
            if np.isfinite(upper):
                op = '<=' if right_bracket == ']' else '<'
                conds.append(f"{feature_sql} {op} {float(upper)!r}")
            return ' AND '.join(conds)
        return f"{feature_sql} = {ScoreCard._sql_literal(label)}"

    @staticmethod
    def _bin_label_to_python_condition(var: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 Python 条件表达式."""
        if isinstance(label, (list, np.ndarray)):
            # 全 NaN 的 list：缺失值描述符应返回 pd.isna()，不参与后续 value_exprs 匹配
            if len(label) > 0 and all(pd.isna(v) for v in label):
                return f"pd.isna({var})"
            value_exprs = [
                repr(value.item() if isinstance(value, np.generic) else value)
                for value in label
                if not pd.isna(value)
            ]
            missing_cond = f"pd.isna({var})" if any(pd.isna(value) for value in label) else None
            if value_exprs and missing_cond:
                if len(value_exprs) == 1:
                    return f"({missing_cond}) or ({var} == {value_exprs[0]})"
                return f"({missing_cond}) or ({var} in ({', '.join(value_exprs)}))"
            if not value_exprs:
                return f"pd.isna({var})"
            if len(value_exprs) > 1:
                return f"{var} in ({', '.join(value_exprs)})"
            return f"{var} == {value_exprs[0]}"

        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"pd.isna({var})"
        if label in ('特殊值', 'special'):
            if special_codes:
                comparisons = []
                for code in special_codes:
                    if pd.isna(code):
                        comparisons.append(f"pd.isna({var})")
                    else:
                        comparisons.append(f"{var} == {code!r}")
                return ' or '.join(comparisons)
            return 'False'
        interval = ScoreCard._parse_interval_label(label)
        if interval is not None:
            left_bracket, lower, upper, right_bracket = interval
            conds = [f"not pd.isna({var})"]
            if np.isfinite(lower):
                op = '>=' if left_bracket == '[' else '>'
                conds.append(f"{var} {op} {float(lower)!r}")
            if np.isfinite(upper):
                op = '<=' if right_bracket == ']' else '<'
                conds.append(f"{var} {op} {float(upper)!r}")
            return ' and '.join(conds)
        return f"{var} == {label!r}"

    @staticmethod
    def _bin_label_to_java_condition(var: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 Java 条件表达式."""
        if isinstance(label, (list, np.ndarray)):
            # 全 NaN 的 list：缺失值描述符应返回 {var} == null，不参与后续值匹配
            if len(label) > 0 and all(pd.isna(v) for v in label):
                return f"{var} == null"
            values = [value.item() if isinstance(value, np.generic) else value for value in label if not pd.isna(value)]
            contains_missing = any(pd.isna(value) for value in label)
            if not values:
                return f"{var} == null"
            conditions = [ScoreCard._java_value_condition(var, value) for value in values]
            if contains_missing:
                conditions.insert(0, f"{var} == null")
            return '(' + ' || '.join(conditions) + ')' if len(conditions) > 1 else conditions[0]

        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{var} == null"
        if label in ('特殊值', 'special'):
            if special_codes:
                conditions = []
                for code in special_codes:
                    if pd.isna(code):
                        conditions.append(f"{var} == null")
                    elif isinstance(code, str):
                        escaped_code = code.replace('\\', '\\\\').replace('"', '\\"')
                        conditions.append(f"\"{escaped_code}\".equals({var})")
                    else:
                        conditions.append(
                            f"({var} instanceof Number && ((Number){var}).doubleValue() == {float(code)!r})"
                        )
                return '(' + ' || '.join(conditions) + ')'
            return 'false'
        interval = ScoreCard._parse_interval_label(label)
        if interval is not None:
            left_bracket, lower, upper, right_bracket = interval
            conds = [f"{var} instanceof Number"]
            if np.isfinite(lower):
                op = '>=' if left_bracket == '[' else '>'
                conds.append(f"((Number){var}).doubleValue() {op} {float(lower)!r}")
            if np.isfinite(upper):
                op = '<=' if right_bracket == ']' else '<'
                conds.append(f"((Number){var}).doubleValue() {op} {float(upper)!r}")
            return ' && '.join(conds)
        escaped_label = label.replace('\\', '\\\\').replace('"', '\\"')
        return f"\"{escaped_label}\".equals({var})"

    @staticmethod
    def _quote_sql_identifier(name: Any) -> str:
        """使用 ANSI 双引号安全引用 SQL 标识符."""
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _sql_literal(value: Any) -> str:
        """将类别值格式化为保留基础类型的 SQL 字面量."""
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            return 'NULL'
        if isinstance(value, (bool, np.bool_)):
            return 'TRUE' if value else 'FALSE'
        if isinstance(value, (int, float, np.integer, np.floating)):
            return repr(value)
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _java_string_literal(value: Any) -> str:
        """生成安全的 Java 字符串字面量."""
        escaped = str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\r', '\\r').replace('\n', '\\n')
        return f'"{escaped}"'

    @staticmethod
    def _java_value_condition(var: str, value: Any) -> str:
        """生成保留类别基础类型的 Java 相等条件."""
        if isinstance(value, (bool, np.bool_)):
            return f"Boolean.valueOf({str(bool(value)).lower()}).equals({var})"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"({var} instanceof Number && ((Number){var}).doubleValue() == {float(value)!r})"
        return f"{ScoreCard._java_string_literal(value)}.equals({var})"

    @staticmethod
    def _safe_java_var(name: str) -> str:
        """将特征名转为合法的 Java 变量名."""
        import re
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if not safe:
            safe = 'feature'
        if safe[0].isdigit():
            safe = 'f_' + safe
        return safe

    def _get_deployment_special_codes(self) -> List[Any]:
        """获取部署导出时需要识别的特殊值编码."""
        if self.binner is None:
            return []
        special_codes = getattr(self.binner, 'special_codes', None)
        return list(special_codes) if special_codes else []

    @staticmethod
    def _is_missing_descriptor(descriptor: Any) -> bool:
        """判断部署规则描述符是否表示缺失值箱."""
        if isinstance(descriptor, (list, np.ndarray)):
            return len(descriptor) == 0 or all(pd.isna(value) for value in descriptor)
        label = str(descriptor).strip().lower()
        return label in ('missing', '缺失值', '缺失', 'nan', 'null')


    @staticmethod
    def _is_special_descriptor(descriptor: Any) -> bool:
        """判断部署规则描述符是否表示特殊值箱."""
        if isinstance(descriptor, (list, np.ndarray)):
            return False
        label = str(descriptor).strip().lower()
        return label in ('special', '特殊值', '特殊')

    def _get_deployment_default_score(self, bins: List[Tuple[Any, float]]) -> float:
        """获取部署规则的默认回退分数.

        显式 ``else`` 规则优先；否则未知类别沿用分箱器的 WOE=0 语义，贡献分为 0。
        """
        for descriptor, score in bins:
            if self._normalize_rule_label(descriptor) == 'else':
                return float(score)
        return 0.0

    def _build_pmml_expression_from_rules(
        self,
        bins: List[Tuple[Any, float]],
        default_score: float,
        special_codes: Optional[List[Any]] = None,
    ) -> str:
        """基于部署规则构建 PMML ExpressionTransformer 表达式."""
        expression = repr(float(default_score))

        for descriptor, score in reversed(bins):
            condition = self._bin_label_to_pmml_condition('X[0]', descriptor, special_codes=special_codes)
            expression = f"({float(score)!r}) if ({condition}) else ({expression})"

        return expression

    def _build_pmml_categorical_lookup_mapping(
        self,
        bins: List[Tuple[Any, float]],
        special_codes: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """为类别变量构建 PMML LookupTransformer 映射."""
        mapping: Dict[str, float] = {}
        missing_score: Optional[float] = None

        for descriptor, score in bins:
            if isinstance(descriptor, (list, np.ndarray)):
                contains_missing = False
                for value in descriptor:
                    if pd.isna(value):
                        contains_missing = True
                    else:
                        mapping[str(value)] = float(score)
                if contains_missing:
                    missing_score = float(score)
                continue

            if self._is_missing_descriptor(descriptor):
                missing_score = float(score)
                continue

            if self._is_special_descriptor(descriptor):
                for code in special_codes or []:
                    if pd.isna(code):
                        if missing_score is None:
                            missing_score = float(score)
                    else:
                        mapping[str(code)] = float(score)
                continue

            mapping[str(descriptor)] = float(score)

        return mapping, missing_score

    @staticmethod
    def _bin_label_to_pmml_condition(var: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将部署规则描述符转为 PMML ExpressionTransformer 使用的条件表达式."""
        if isinstance(label, (list, np.ndarray)):
            # 全 NaN 的 list：缺失值描述符应返回 pandas.isnull()，不参与后续 value_exprs 匹配
            if len(label) > 0 and all(pd.isna(v) for v in label):
                return f"pandas.isnull({var})"
            value_exprs = [
                repr(value.item() if isinstance(value, np.generic) else value)
                for value in label
                if not pd.isna(value)
            ]
            missing_cond = f"pandas.isnull({var})" if any(pd.isna(value) for value in label) else None
            if value_exprs and missing_cond:
                if len(value_exprs) == 1:
                    return f"({missing_cond}) or ({var} == {value_exprs[0]})"
                return f"({missing_cond}) or ({var} in [{', '.join(value_exprs)}])"
            if not value_exprs:
                return f"pandas.isnull({var})"
            if len(value_exprs) == 1:
                return f"{var} == {value_exprs[0]}"
            return f"{var} in [{', '.join(value_exprs)}]"

        label = str(label).strip()
        label_lower = label.lower()

        if label_lower in ('missing', '缺失值', '缺失', 'nan', 'null'):
            return f"pandas.isnull({var})"


        if label_lower in ('special', '特殊值', '特殊'):
            if special_codes:
                comparisons = []
                for code in special_codes:
                    if pd.isna(code):
                        comparisons.append(f"pandas.isnull({var})")
                    else:
                        comparisons.append(f"{var} == {code!r}")
                return ' or '.join(comparisons)
            return 'False'

        interval = ScoreCard._parse_interval_label(label)
        if interval is not None:
            left_bracket, lower, upper, right_bracket = interval
            conditions = [f"not pandas.isnull({var})"]
            if np.isfinite(lower):
                operator = '>=' if left_bracket == '[' else '>'
                conditions.append(f"{var} {operator} {float(lower)!r}")
            if np.isfinite(upper):
                operator = '<=' if right_bracket == ']' else '<'
                conditions.append(f"{var} {operator} {float(upper)!r}")
            return ' and '.join(conditions)

        return f"{var} == {label!r}"

    @staticmethod
    def _build_pmml_lookup_mapping(
        bins: Union[np.ndarray, list],
        scores: Union[np.ndarray, list],
    ) -> Tuple[Dict[str, float], float]:
        """为类别分箱构建 PMML LookupTransformer 所需的映射."""
        mapping: Dict[str, float] = {}
        default_value = 0.0
        default_labels = {
            'missing', '缺失值', '缺失', 'nan', 'null',

            'special', '特殊值', '特殊',
        }

        for bin_value, score in zip(bins, scores):
            values = bin_value if isinstance(bin_value, (list, np.ndarray)) else [bin_value]
            for value in values:
                if pd.isna(value):
                    default_value = float(score)
                    continue

                label = str(value).strip()
                if label.lower() in default_labels:
                    default_value = float(score)
                else:
                    mapping[label] = float(score)

        return mapping, default_value

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
        """获取评分卡的特征重要性表（基于 LR 系数）。

        以逻辑回归系数绝对值衡量重要性。与 :meth:`get_feature_importances`（返回 Series，
        支持 ``coef`` / ``score_range`` 两种口径）不同，本方法返回明细 DataFrame。

        :return: DataFrame，含 ``feature`` / ``coef`` / ``importance`` 三列，按
            ``importance`` 降序
        :raises NotFittedError: 评分卡尚未拟合时

        **参考样例**

        >>> scorecard.get_feature_importance().head()
        """
        self._check_fitted()

        return pd.DataFrame({
            'feature': self.feature_names_,
            'coef': self.coef_,
            'importance': np.abs(self.coef_)
        }).sort_values('importance', ascending=False)

    def get_reason(self, X: Union[pd.DataFrame, np.ndarray], keep: int = 3) -> pd.DataFrame:
        """输出每个样本评分的主要驱动原因（reason codes / 拒绝原因）。

        对每个样本，按"该特征得分相对其基准效应的偏离"排序，取偏离最负（最拉低分数）的
        前 ``keep`` 个特征作为不利原因，可用于授信拒绝理由说明（adverse action）与可解释性。

        :param X: 输入数据，原始或 WOE 数据（自动识别），DataFrame 或 ndarray
        :param keep: 每个样本保留的主要原因（特征）个数，默认为 ``3``
        :return: DataFrame，每行对应一个样本，列为排序后的 Top-``keep`` 不利特征及其影响
        :raises NotFittedError: 评分卡尚未拟合时

        **参考样例**

        >>> scorecard.get_reason(X_test, keep=3)
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 转换数据
        is_woe = self._detect_input_type(X)
        if not is_woe:
            X = self._transform_to_woe(X)

        sub_scores = self._woe_to_score(X[self.feature_names_])
        # 兼容 base_effect_ 为 numpy array 或 pandas Series 的情况
        if isinstance(self.base_effect_, pd.Series):
            base_effect_values = self.base_effect_.values
        else:
            base_effect_values = np.asarray(self.base_effect_)
        effect_diff = sub_scores - base_effect_values
        
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
        """生成评分区间与理论逾期率（坏样本概率）对照表。

        将分数分档后，依据评分卡刻度公式由各档分数中位数反推理论 odds 与坏样本概率
        （``prob = odds/(1+odds)``，``odds = exp((A - score)/B)``），可叠加真实标签 ``y``
        对比理论与实际逾期率，用于评分卡校准核验与划档定价。

        :param scores: 分数数组；若为 ``None`` 则用 ``X`` 经 :meth:`predict` 计算
        :param X: 输入数据，当 ``scores`` 未提供时用于预测分数
        :param y: 可选真实标签，提供后表中追加各档实际坏样本率以对比理论值
        :param n_bins: 分数分档数量，默认为 ``10``
        :param method: 分档方式，默认为 ``'quantile'``。可取以下枚举值：

            - ``'quantile'``：等频分档（``pd.qcut``）
            - ``'uniform'``：等距分档（``pd.cut``）
            - ``'custom'``：使用 ``score_bins`` 指定的自定义分档边界

        :param score_bins: 自定义分档边界列表，仅当 ``method='custom'`` 时生效
        :return: DataFrame，含评分区间及对应理论 odds / 理论坏样本概率（提供 ``y`` 时含实际值）
        :raises ValueError: ``scores`` 与 ``X`` 均未提供时

        **参考样例**

        >>> scorecard.score_to_probability_table(X=X_test, y=y_test, n_bins=10)
        """
        self._check_fitted()

        if scores is None:
            if X is None:
                raise ValueError("必须提供 scores 或 X 参数之一")
            scores = self.predict(X)

        scores = np.asarray(scores)
        score_series = pd.Series(scores)

        if len(scores) == 0:
            return pd.DataFrame(columns=[
                '评分区间', '评分中位数', '理论逾期率', '理论Odds', '样本数'
            ])
        if method == 'custom' and score_bins is not None:
            bins = pd.IntervalIndex.from_breaks(score_bins)
            score_bin = pd.cut(score_series, bins=bins, include_lowest=True)
        elif score_series.nunique(dropna=True) <= 1:
            score_value = float(score_series.dropna().iloc[0]) if score_series.notna().any() else np.nan
            score_bin = pd.Series([f"[{score_value:.2f}, {score_value:.2f}]"] * len(score_series))
        elif method == 'uniform':
            score_bin = pd.cut(score_series, bins=n_bins, include_lowest=True)
        else:
            score_bin = pd.qcut(score_series, q=n_bins, duplicates='drop')

        result = []
        categories = score_bin.cat.categories if hasattr(score_bin, 'cat') else pd.Series(score_bin).unique()

        for interval in categories:
            mask = score_bin == interval
            bin_scores = scores[mask]

            if len(bin_scores) == 0:
                continue

            score_median = np.median(bin_scores)
            odds_theoretical = np.exp((self.A_ - score_median) / self.B_)
            prob_theoretical = odds_theoretical / (1 + odds_theoretical)
            interval_label = (
                f"[{interval.left:.0f}, {interval.right:.0f})"
                if hasattr(interval, 'left') and hasattr(interval, 'right')
                else str(interval)
            )

            row = {
                '评分区间': interval_label,
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
        """输出样本级评分明细：基础分 + 各特征贡献分 + 总分（可附主要原因）。

        将总分拆解为截距基础分与每个特征的贡献分，便于逐样本审视分数构成、做可解释性展示。

        :param X: 输入数据，原始或 WOE 数据（自动识别），DataFrame 或 ndarray
        :param sample_idx: 仅输出指定样本的明细，可为单个下标或下标列表；``None`` 表示全部
        :param include_reason: 是否附带主要驱动原因列（同 :meth:`get_reason`），默认为 ``True``
        :return: DataFrame，每行一个样本，列含各特征贡献分、基础分与总分
        :raises NotFittedError: 评分卡尚未拟合时

        **参考样例**

        >>> scorecard.get_detailed_score(X_test, sample_idx=0)
        """
        self._check_fitted()

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
        intercept_score = self.A_ - self.B_ * self.intercept_
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
        # 兼容 base_effect_ 为 numpy array 或 pandas Series 的情况
        if isinstance(self.base_effect_, pd.Series):
            base_effect_values = self.base_effect_.values
        else:
            base_effect_values = np.asarray(self.base_effect_)
        effect_diff = sub_scores - base_effect_values

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

    def export(
        self,
        to_json: Optional[str] = None,
        to_frame: bool = False,
        decimal: int = 2,
        include_meta: bool = True,
        compatibility: Optional[str] = None,
    ) -> Union[Dict, pd.DataFrame]:
        """导出评分卡规则，兼容 toad/scorecardpipeline 格式.

        导出格式与 toad.ScoreCard.export() 和 scorecardpipeline.ScoreCard.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :param to_frame: 是否返回 DataFrame 格式，默认为 False
        :param decimal: 分数保留小数位数，默认为 2
        :param include_meta: 是否额外导出重建评分所需元数据，默认为 True
        :param compatibility: 外部兼容格式；设为 ``'toad'`` 时等价于 ``include_meta=False``
        :return: 评分卡规则字典或 DataFrame
            - 字典格式: {'feature': {'bin_label': score, ...}, ...}
            - DataFrame格式: columns=['name', 'value', 'score']

        **参考样例**

        >>> card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> card.fit(X_woe, y, binner=binner)
        >>>
        >>> # 导出为字典
        >>> rules = card.export()
        >>>
        >>> # 导出并保存到 JSON 文件
        >>> rules = card.export(to_json='scorecard_rules.json')
        >>>
        >>> # 导出为 DataFrame
        >>> df = card.export(to_frame=True)

        **与 toad/scorecardpipeline 的兼容性**

        导出的规则可以直接被 toad 和 scorecardpipeline 加载:

        >>> # toad 加载（显式导出不含 hscredit 元数据的兼容格式）
        >>> import toad
        >>> toad_rules = card.export(compatibility='toad')
        >>> toad_card = toad.ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> toad_card.load(toad_rules)
        >>>
        >>> # scorecardpipeline 加载
        >>> from scorecardpipeline import ScoreCard
        >>> scp_rules = card.export(compatibility='scorecardpipeline')
        >>> scp_card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scp_card.load(scp_rules)
        """
        import json
        
        self._check_fitted()

        if compatibility is not None:
            if str(compatibility).lower() not in ('toad', 'scorecardpipeline'):
                raise ValueError("compatibility 仅支持 'toad' 或 'scorecardpipeline'")
            include_meta = False

        # 构建与 toad 兼容的格式
        card: Dict[str, Any] = {}
        for col in self.feature_names_:
            rule = self.rules_[col]
            bins = rule['bins']
            bin_labels = rule.get('bin_labels')
            scores = rule['scores']

            if (bins is None or len(bins) == 0) and (bin_labels is None or len(bin_labels) == 0):
                continue

            feature_rules = {}
            if bin_labels is not None and len(bin_labels) == len(scores):
                for bin_label, score in zip(bin_labels, scores):
                    feature_rules[str(bin_label)] = round(float(score), decimal)
            elif isinstance(bins[0], (list, np.ndarray)):
                # 类别特征
                for bin_vals, score in zip(bins, scores):
                    bin_label = ', '.join([str(v) for v in bin_vals])
                    feature_rules[bin_label] = round(float(score), decimal)
            else:
                # 数值特征 - 格式化为区间标签
                has_string_bins = (len(bins) > 0 and isinstance(bins[0], str) and
                                 ('[' in str(bins[0]) or '(' in str(bins[0])))
                
                if has_string_bins:
                    # 已经是格式化的标签
                    for bin_label, score in zip(bins, scores):
                        feature_rules[str(bin_label)] = round(float(score), decimal)
                else:
                    # 数值切分点，格式化为区间
                    for i, score in enumerate(scores):
                        if i == 0:
                            bin_label = f'[-inf, {bins[0]})' if len(bins) > 0 else '[-inf, +inf)'
                        elif i == len(scores) - 1:
                            bin_label = f'[{bins[-1]}, +inf)' if len(bins) > 0 else '[-inf, +inf)'
                        else:
                            bin_label = f'[{bins[i-1]}, {bins[i]})'
                        feature_rules[bin_label] = round(float(score), decimal)

            card[col] = feature_rules

        if include_meta:
            intercept_score = float(self.A_ - self.B_ * self.intercept_)
            feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}
            special_codes = self._get_deployment_special_codes()
            handle_unknown = getattr(self.binner, 'handle_unknown', -3) if self.binner is not None else -3
            card['__meta__'] = {
                'format': 'hscredit-scorecard-rules',
                'version': 1,
                'intercept_score': intercept_score,
                'base_score': float(self.base_score),
                'direction': self.direction_,
                'pdo': self.pdo,
                'rate': self.rate,
                'base_odds': self.base_odds,
                'step': self.step,
                'lower': self.lower,
                'upper': self.upper,
                'clip': self.clip,
                'decimal': self.decimal,
                'score_decimal': decimal,
                'A': float(self.A_),
                'B': float(self.B_),
                'feature_names': list(self.feature_names_),
                'feature_types': {feature: feature_types.get(feature) for feature in self.feature_names_},
                'coef': [
                    float(coef * self._get_feature_woe_sign(i))
                    for i, coef in enumerate(self.coef_)
                ],
                'categorical_bins': self._get_export_categorical_bins(),
                'special_codes': [
                    value.item() if isinstance(value, np.generic) else value
                    for value in special_codes
                ],
                'handle_unknown': handle_unknown,
            }

        # 保存到 JSON 文件
        if to_json is not None:
            import os
            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(to_json, 'w', encoding='utf-8') as f:
                json.dump(card, f, ensure_ascii=False, indent=2)

        # 返回 DataFrame 格式
        if to_frame:
            rows = []
            for name in card:
                if name == '__meta__':
                    continue
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })
            frame = pd.DataFrame(rows)
            if include_meta:
                frame.attrs['scorecard_meta'] = dict(card['__meta__'])
            return frame

        return card

    def _apply_export_metadata(self, meta: Dict[str, Any]) -> None:
        """应用导出文件中的评分卡元数据."""
        format_name = meta.get('format')
        if format_name == 'hscredit-scorecard-rules' and meta.get('version', 1) != 1:
            raise ValueError(f"不支持的评分卡规则版本: {meta.get('version')}")

        self.pdo = meta.get('pdo', self.pdo)
        self.rate = meta.get('rate', self.rate)
        self.base_odds = meta.get('base_odds', self.base_odds)
        self.base_score = meta.get('base_score', self.base_score)
        self.step = meta.get('step', self.step)
        self.lower = meta.get('lower', self.lower)
        self.upper = meta.get('upper', self.upper)
        self.clip = meta.get('clip', self.clip)
        self.decimal = meta.get('decimal', self.decimal)
        self.precision = self.decimal

        direction = meta.get('direction')
        if direction is not None:
            self.direction = direction

        self.A_, self.B_ = self._compute_parameters()
        self.direction_ = self._determine_direction()

        intercept_score = meta.get('intercept_score')
        if intercept_score is not None:
            self._loaded_intercept = (self.A_ - float(intercept_score)) / self.B_

        coef = meta.get('coef')
        if coef is not None:
            self._loaded_coef = np.asarray(coef, dtype=float)

        feature_names = meta.get('feature_names')
        if feature_names is not None:
            self._feature_names = list(feature_names)

        self._loaded_feature_types = dict(meta.get('feature_types', {}))
        self._loaded_categorical_bins = dict(meta.get('categorical_bins', {}))
        self._loaded_special_codes = list(meta.get('special_codes', []))
        self._loaded_handle_unknown = meta.get('handle_unknown', -3)

    def _get_export_categorical_bins(self) -> Dict[str, List[List[Any]]]:
        """获取可 JSON 序列化且不丢失逗号类别的结构化类别规则。"""
        cat_bins = getattr(self.binner, '_cat_bins_', {}) if self.binner is not None else {}
        result = {}
        for feature, groups in cat_bins.items():
            if feature not in self.feature_names_:
                continue
            result[feature] = [
                [value.item() if isinstance(value, np.generic) else value for value in group]
                for group in groups
            ]
        return result

    @staticmethod
    def _coerce_scorecard_rules(data: Any) -> Dict[str, Any]:
        """将外部导出的评分卡数据统一为 feature -> {bin: score} 字典."""
        if isinstance(data, pd.DataFrame):
            card = ScoreCard._scorecard_records_to_dict(data.to_dict('records'))
            meta = data.attrs.get('scorecard_meta')
            if meta is not None:
                card['__meta__'] = dict(meta)
            return card

        if isinstance(data, list):
            return ScoreCard._scorecard_records_to_dict(data)

        if isinstance(data, dict):
            if {'name', 'value', 'score'}.issubset(data.keys()) or {'变量名称', '变量分箱', '对应分数'}.issubset(data.keys()):
                return ScoreCard._scorecard_records_to_dict(pd.DataFrame(data).to_dict('records'))
            if {'columns', 'data'}.issubset(data.keys()):
                try:
                    records = pd.DataFrame(data['data'], columns=data['columns']).to_dict('records')
                    return ScoreCard._scorecard_records_to_dict(records)
                except Exception:
                    pass
            return dict(data)

        raise ValueError("评分卡规则必须是字典、DataFrame 或包含 name/value/score 的记录列表")

    @staticmethod
    def _scorecard_records_to_dict(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """将 export(to_frame=True) 或 DataFrame JSON records 转为规则字典."""
        card: Dict[str, Dict[str, float]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("评分卡记录列表中的每一项都必须是字典")

            feature = record.get('name', record.get('变量名称', record.get('变量名')))
            value = record.get('value', record.get('变量分箱', record.get('分箱')))
            score = record.get('score', record.get('对应分数', record.get('分数')))

            if feature is None or value is None or score is None:
                raise ValueError("评分卡记录必须包含 name/value/score 或 变量名称/变量分箱/对应分数")

            if str(feature) == '基础分':
                continue

            card.setdefault(str(feature), {})[str(value)] = float(score)

        return card

    @staticmethod
    def _coerce_feature_rules(feature: str, feature_rules: Any) -> Dict[str, float]:
        """兼容 feature -> list[record] 形式的单特征规则."""
        if isinstance(feature_rules, dict):
            return feature_rules

        if isinstance(feature_rules, list):
            converted: Dict[str, float] = {}
            for item in feature_rules:
                if not isinstance(item, dict):
                    raise ValueError(f"特征 {feature} 的规则列表必须由字典组成")
                value = item.get('value', item.get('变量分箱', item.get('分箱')))
                score = item.get('score', item.get('对应分数', item.get('分数')))
                if value is None or score is None:
                    raise ValueError(f"特征 {feature} 的规则记录必须包含 value/score")
                converted[str(value)] = float(score)
            return converted

        raise ValueError(f"特征 {feature} 的评分卡规则必须是字典")

    @staticmethod
    def _parse_loaded_bin_descriptor(bin_label: Any) -> Tuple[bool, Any]:
        """解析 load 输入的分箱标签，返回 (是否区间, 规则描述)."""
        interval = ScoreCard._parse_interval_label(bin_label)
        if interval is not None:
            _, _, upper, _ = interval
            return True, None if upper == np.inf else upper

        normalized = ScoreCard._normalize_rule_label(bin_label)
        if normalized in ('missing', 'special', 'else'):
            return False, normalized

        label_str = str(bin_label)
        if ',' in label_str:
            return False, [v.strip() for v in label_str.split(',') if v.strip()]

        return False, [label_str]

    def load_rules(
        self,
        from_json: Union[str, os.PathLike, Dict],
        update: bool = False,
        binner: Optional[Any] = None
    ) -> 'ScoreCard':
        """加载评分卡规则，兼容 hscredit/toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载评分卡规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: 评分卡规则字典或 JSON 文件路径
            - 字典: {'feature': {'bin_label': score, ...}, ...}
            - 文件路径: 'scorecard_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :param binner: 可选的分箱器，用于对原始数据进行分箱后评分。
            - 如果提供，将用于 predict(input_type='raw') 时的数据转换
            - 如果不提供，将基于规则中的分箱信息进行转换
        :return: self，支持链式调用

        **参考样例**

        >>> card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>>
        >>> # 从字典加载
        >>> rules = {'age': {'[18, 25)': 50, '[25, 35)': 45}}
        >>> card.load(rules)
        >>>
        >>> # 从 JSON 文件加载
        >>> card.load('scorecard_rules.json')
        >>>
        >>> # 更新现有规则
        >>> card.load({'new_feature': {'bin1': 10, 'bin2': 20}}, update=True)

        **与 toad/scorecardpipeline 的兼容性**

        可以直接加载 toad 和 scorecardpipeline 导出的规则:

        >>> # toad 导出
        >>> import toad
        >>> toad_card = toad.ScoreCard()
        >>> toad_card.fit(X, y, combiner=combiner, transer=transformer)
        >>> rules = toad_card.export()
        >>>
        >>> # hscredit 加载
        >>> from hscredit.core.models import ScoreCard
        >>> card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> card.load(rules)
        """
        import json

        if isinstance(from_json, (str, os.PathLike)):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                card = self._coerce_scorecard_rules(json.load(f))
        else:
            # 直接使用字典、DataFrame 或 records
            card = self._coerce_scorecard_rules(from_json)

        meta = None
        if isinstance(card, dict):
            meta = card.pop('__meta__', None)

        if not update:
            self.rules_ = {}
            self._feature_names = []
            self.base_effect_ = None
            # 替换规则时不能继续保留旧模型链，否则 predict 会优先使用旧 LR/binner，
            # 新加载的规则只影响 export 而不参与评分。需要复用外部分箱器时应通过
            # load_rules(..., binner=...) 显式传入。
            self.lr_model = None
            self.lr_model_ = None
            self.encoder = None
            self.pipeline = None
            self._pipeline_components = {}
            self.binner = None
            self._binner_is_woe_transformer = False
            if hasattr(self, '_rule_binner'):
                del self._rule_binner
            self._loaded_intercept = None
            self._loaded_coef = None
            self._loaded_feature_types = {}
            self._loaded_categorical_bins = {}
            self._loaded_special_codes = []
            self._loaded_handle_unknown = -3
        else:
            # update=True 同样表示后续评分以规则为准；保留现有分箱器用于产生箱标签，
            # 但必须移除旧 LR，否则更新后的分值不会被 predict 使用。
            self.lr_model = None
            self.lr_model_ = None
            self.pipeline = None
            self._pipeline_components = {}

        categorical_bins = meta.get('categorical_bins', {}) if meta else {}
        if meta:
            self._apply_export_metadata(meta)

        # 解析规则
        for feature, feature_rules in card.items():
            # 兼容历史导出文件：'基础分'（截距项）不是特征，跳过以免污染特征列表
            if feature == '基础分':
                continue
            feature_rules = self._coerce_feature_rules(str(feature), feature_rules)
            if self._feature_names is None:
                self._feature_names = []
            if feature not in self._feature_names:
                self._feature_names.append(feature)

            bins = []
            numeric_splits = []
            has_interval_rule = False
            scores = []
            bin_labels = []

            for bin_label, score in feature_rules.items():
                bin_labels.append(str(bin_label))
                scores.append(float(score))

                is_interval, descriptor = self._parse_loaded_bin_descriptor(bin_label)
                if is_interval:
                    has_interval_rule = True
                    if descriptor is not None:
                        numeric_splits.append(float(descriptor))
                else:
                    bins.append(descriptor)

            if has_interval_rule:
                # 数值型：保留切分点；全量箱无切分点时保留标签，避免离线分箱被跳过。
                splits = sorted(list(set(numeric_splits))) if numeric_splits else list(bin_labels)
            else:
                # 类别型：保持列表格式
                structured = categorical_bins.get(feature)
                splits = (
                    structured
                    if structured is not None and len(structured) == len(scores)
                    else bins if bins else list(bin_labels)
                )

            self.rules_[feature] = {
                'bins': splits,
                'bin_labels': np.array(bin_labels, dtype=object),
                'scores': np.array(scores),
            }

        if meta is None and getattr(self, '_loaded_intercept', None) is None and self.rules_:
            # toad/scorecardpipeline export 的分箱分数已经包含截距分摊，导入后直接累加各分箱分。
            self._loaded_intercept = self.A_ / self.B_

        # 计算基础效应
        if not hasattr(self, 'base_effect_') or self.base_effect_ is None:
            self.base_effect_ = np.zeros(len(self.feature_names_))

        # 如果提供了 binner，保存并设置标志
        if binner is not None:
            self.binner = binner
            self._binner_is_woe_transformer = True

        # 如果没有 binner 但有规则（来自 export JSON），尝试从规则中恢复分箱能力
        if self.binner is None and self.rules_:
            self._setup_rule_based_binner()

        self._is_fitted = True
        return self

    load = _ScoreCardLoadDispatcher()


class RoundScoreCard(ScoreCard):
    """按分箱分数精度进行一致性计分的评分卡模型.

    与 :class:`ScoreCard` 不同，``RoundScoreCard`` 会先将基础分和各特征分箱分数
    按初始化指定的 ``decimal`` 精度进行取整，再基于这份取整后的评分卡完成
    预测、原因分析与部署导出，确保对外结果与 :meth:`scorecard_points` 完全一致
    （避免"展示分"与"实际计分"因四舍五入产生偏差，便于落地核对）。

    构造参数与 :class:`ScoreCard` 完全一致，仅评分计分口径不同；下列方法
    （:meth:`fit` / :meth:`scorecard_points` / :meth:`predict` / :meth:`predict_score` /
    :meth:`get_reason` / :meth:`get_detailed_score` / :meth:`export`）语义同父类，
    区别仅在于使用"取整后分数"参与计算与导出。

    **参数**

    :param decimal: 评分卡分数保留小数位数，默认 2
    :param scorecard: 可选，已训练的 :class:`ScoreCard` 实例，传入后直接复用其规则转为整数计分卡

    **参考样例**

    >>> from hscredit.core.models import RoundScoreCard
    >>> sc = RoundScoreCard(decimal=0, binner=binner)   # 整数分评分卡
    >>> sc.fit(X_train, y_train, input_type='raw')
    >>> sc.scorecard_points()        # 各分箱整数分
    >>> sc.predict(X_test)           # 与 scorecard_points 完全一致的计分

    **引用**

    取整一致性计分对应评分卡工程落地实践，刻度公式同 :class:`ScoreCard`
    （Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley）。
    """

    def __init__(
        self,
        pdo: float = 60,
        rate: float = 2,
        base_odds: float = 35,
        base_score: float = 750,
        step: Optional[int] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        direction: str = 'descending',
        decimal: int = 2,
        lr_model: Optional[Any] = None,
        lr_kwargs: Optional[Dict[str, Any]] = None,
        binner: Optional[Any] = None,
        encoder: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        scorecard: Optional['ScoreCard'] = None,
        calculate_stats: bool = True,
        verbose: bool = False,
        target: str = 'target',
        **kwargs
    ):
        super().__init__(
            pdo=pdo,
            rate=rate,
            base_odds=base_odds,
            base_score=base_score,
            step=step,
            lower=lower,
            upper=upper,
            direction=direction,
            decimal=decimal,
            lr_model=lr_model,
            lr_kwargs=lr_kwargs,
            binner=binner,
            encoder=encoder,
            pipeline=pipeline,
            calculate_stats=calculate_stats,
            verbose=verbose,
            target=target,
            **kwargs
        )
        self.decimal = decimal
        self.scorecard = scorecard

        # 如果传入了 ScoreCard 对象，从中复制配置
        if scorecard is not None:
            self._copy_from_scorecard(scorecard)

    def _copy_from_scorecard(self, scorecard: 'ScoreCard') -> None:
        """从传入的 ScoreCard 对象复制必要属性，实现即插即用.

        复制以下属性：
        - rules_: 特征与分箱规则
        - feature_names_: 特征名称列表
        - intercept_: 截距项
        - coef_: 系数向量
        - A_, B_: 缩放参数
        - direction_: 评分方向
        - base_effect_: 基础效应（用于 get_detailed_score）
        - binner: 分箱器（如果有）
        - _is_fitted: 拟合标志
        - _binner_is_woe_transformer: 分箱器类型标志
        - lr_model_: 逻辑回归模型（如果有）

        设置以下标志：
        - _skip_fit_check: 跳过拟合检查
        - _uses_woe_input: 使用 WOE 输入
        - _binner_is_woe_transformer: 基于 binner 类型
        """
        scorecard._check_fitted()

        # 复制基础属性
        self.rules_ = scorecard.rules_
        self._feature_names = scorecard._feature_names  # 使用内部属性
        # intercept_ 和 coef_ 是只读属性，通过 lr_model 或 _loaded_intercept 设置
        # 尝试从 lr_model 获取
        if hasattr(scorecard, 'lr_model') and scorecard.lr_model is not None:
            self.lr_model = scorecard.lr_model
        elif hasattr(scorecard, 'lr_model_') and scorecard.lr_model_ is not None:
            self.lr_model_ = scorecard.lr_model_
        # 如果有 _loaded_intercept（从 load 方法设置），也会被 intercept_ property 正确获取
        if hasattr(scorecard, '_loaded_intercept'):
            self._loaded_intercept = scorecard._loaded_intercept

        self.A_ = scorecard.A_
        self.B_ = scorecard.B_
        self.direction_ = scorecard.direction_
        self.base_effect_ = scorecard.base_effect_
        self.lower = scorecard.lower
        self.upper = scorecard.upper

        # 复制分箱器（如果有）
        if hasattr(scorecard, 'binner') and scorecard.binner is not None:
            self.binner = scorecard.binner
            # 根据 binner 类型设置标志
            if hasattr(scorecard, '_binner_is_woe_transformer'):
                self._binner_is_woe_transformer = scorecard._binner_is_woe_transformer
            else:
                # 默认为 WOE 转换器模式
                self._binner_is_woe_transformer = True

        # 复制逻辑回归模型（如果有）
        if hasattr(scorecard, 'lr_model_') and scorecard.lr_model_ is not None:
            self.lr_model_ = scorecard.lr_model_
        elif hasattr(scorecard, 'lr_model') and scorecard.lr_model is not None:
            self.lr_model_ = scorecard.lr_model

        # 设置必要标志
        self._is_fitted = True
        self._skip_fit_check = True

        # 设置评分方向相关属性
        if hasattr(scorecard, 'base_score'):
            self.base_score = scorecard.base_score
        if hasattr(scorecard, 'pdo'):
            self.pdo = scorecard.pdo
        if hasattr(scorecard, 'rate'):
            self.rate = scorecard.rate
        if hasattr(scorecard, 'base_odds'):
            self.base_odds = scorecard.base_odds
        if hasattr(scorecard, 'step'):
            self.step = scorecard.step
        if hasattr(scorecard, 'target'):
            self.target = scorecard.target

    def _round_score_value(self, value: float, decimal: Optional[int] = None) -> float:
        """对单个分数值按指定精度进行四舍五入."""
        digits = self.decimal if decimal is None else decimal
        rounded = round(float(value), digits)
        return 0.0 if rounded == -0.0 else float(rounded)

    def _round_score_array(self, values: Union[np.ndarray, pd.Series], decimal: Optional[int] = None) -> np.ndarray:
        """对分数数组按指定精度进行四舍五入."""
        digits = self.decimal if decimal is None else decimal
        arr = np.round(np.asarray(values, dtype=float), digits)
        arr[np.isclose(arr, 0.0)] = 0.0
        return arr

    def _format_score_text(self, value: float, decimal: Optional[int] = None) -> str:
        """格式化分数字符串显示."""
        digits = self.decimal if decimal is None else decimal
        rounded = self._round_score_value(value, digits)
        if digits <= 0:
            return str(int(round(rounded)))
        return f"{rounded:.{digits}f}"

    def _get_score_sign(self) -> float:
        """获取当前评分方向对应的子分符号."""
        return -1.0 if self.direction_ == 'ascending' else 1.0

    def _get_rounded_base_score(self, decimal: Optional[int] = None) -> float:
        """获取按评分卡精度调整后的基础分."""
        base_score = float(self.A_ - self.B_ * self.intercept_)
        if self.direction_ == 'ascending':
            flip_constant = self.lower + self.upper if self.lower is not None and self.upper is not None else 2 * self.base_score
            base_score = float(flip_constant - base_score)
        return self._round_score_value(base_score, decimal)

    def _get_rounded_rule_scores(self, rule: Dict[str, Any], decimal: Optional[int] = None) -> np.ndarray:
        """获取按评分卡精度调整后的规则分数."""
        score_sign = self._get_score_sign()
        return np.array([
            self._round_score_value(score_sign * float(score), decimal)
            for score in rule.get('scores', [])
        ], dtype=float)

    def _round_sub_scores_from_woe(
        self,
        X_woe: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        decimal: Optional[int] = None
    ) -> np.ndarray:
        """基于 WOE 数据计算按评分卡精度调整后的子分矩阵."""
        if feature_names is None:
            feature_names = self.feature_names_

        X_effective = self._prepare_woe_for_scoring(X_woe)
        scores = np.zeros((X_effective.shape[0], len(feature_names)))
        score_sign = self._get_score_sign()

        for i, col in enumerate(feature_names):
            if col not in X_effective.columns:
                continue
            raw_scores = -self.B_ * self.coef_[i] * X_effective[col].to_numpy(dtype=float)
            scores[:, i] = self._round_score_array(score_sign * raw_scores, decimal)

        return scores

    def _bin_labels_to_rounded_score(
        self,
        X_bins: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        decimal: Optional[int] = None
    ) -> np.ndarray:
        """将分箱标签数据映射为按评分卡精度调整后的子分矩阵."""
        if feature_names is None:
            feature_names = self.feature_names_

        scores = np.zeros((X_bins.shape[0], len(feature_names)))

        for i, col in enumerate(feature_names):
            if col not in X_bins.columns or col not in self.rules_:
                continue

            rule = self.rules_[col]
            rule_labels = rule.get('bin_labels')
            if rule_labels is None:
                rule_labels = rule.get('bins', [])

            rounded_scores = self._get_rounded_rule_scores(rule, decimal=decimal)
            score_map = {
                self._normalize_rule_label(label): float(score)
                for label, score in zip(rule_labels, rounded_scores)
            }

            label_series = X_bins[col].map(self._normalize_rule_label)
            scores[:, i] = label_series.map(score_map).fillna(0.0).to_numpy(dtype=float)

        return scores

    def _resolve_round_scoring_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'raw'
    ) -> Dict[str, Any]:
        """统一解析 RoundScoreCard 评分时需要的输入数据."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        is_woe_data = self._detect_input_type(X)
        uses_loaded_rule_scoring = self._should_use_loaded_rule_scoring(input_type, is_woe_data)

        if uses_loaded_rule_scoring:
            feature_names = self.feature_names_
            X_bins = self._transform_to_bins(X)[feature_names]
            sub_scores = self._bin_labels_to_rounded_score(X_bins, feature_names)
            return {
                'X_raw': X,
                'X_woe': None,
                'X_bins': X_bins,
                'feature_names': feature_names,
                'sub_scores': sub_scores,
            }

        if input_type == 'auto':
            X_woe = X if is_woe_data else self._transform_to_woe(X)
        elif input_type == 'raw':
            X_woe = self._transform_to_woe(X)
        elif input_type == 'woe':
            X_woe = X
        else:
            raise ValueError(f"input_type 必须是 'auto'/'raw'/'woe' 之一，当前为: {input_type}")

        if self._skip_fit_check and not getattr(self, '_is_fitted', False):
            feature_names = X_woe.columns.tolist()
        else:
            feature_names = self.feature_names_
            X_woe = X_woe[feature_names]

        X_bins = None
        if input_type == 'raw' or (input_type == 'auto' and not is_woe_data):
            try:
                X_bins = self._transform_to_bins(X)
                X_bins = X_bins[feature_names]
            except Exception:
                X_bins = None

        sub_scores = self._round_sub_scores_from_woe(X_woe, feature_names)

        return {
            'X_raw': X,
            'X_woe': X_woe,
            'X_bins': X_bins,
            'feature_names': feature_names,
            'sub_scores': sub_scores,
        }

    def _get_base_effect_values(self, feature_names: List[str]) -> np.ndarray:
        """获取与当前特征顺序对齐的基础效应值."""
        if self.base_effect_ is None:
            return np.zeros(len(feature_names), dtype=float)

        if isinstance(self.base_effect_, pd.Series):
            return self.base_effect_.reindex(feature_names).fillna(0.0).to_numpy(dtype=float)

        base_effect = np.asarray(self.base_effect_, dtype=float)
        if base_effect.shape[0] != len(feature_names):
            return np.zeros(len(feature_names), dtype=float)
        return base_effect

    def _find_bin_label_from_woe(self, feature: str, woe_value: float) -> str:
        """根据 WOE 值查找对应的分箱标签."""
        rule = self.rules_.get(feature, {})
        woe_values = rule.get('woe')
        bin_labels = rule.get('bin_labels')
        if woe_values is None or bin_labels is None:
            return '未知'

        candidates = [woe_value]
        if feature in self.feature_names_:
            feature_index = self.feature_names_.index(feature)
            sign = self._get_feature_woe_sign(feature_index)
            candidates.append(float(woe_value) * sign)

        for label, stored_woe in zip(bin_labels, woe_values):
            if any(pd.isna(candidate) and pd.isna(stored_woe) for candidate in candidates):
                return self._format_bin_display(label)
            for candidate in candidates:
                try:
                    if np.isclose(float(candidate), float(stored_woe), atol=1e-12, rtol=0):
                        return self._format_bin_display(label)
                except (TypeError, ValueError):
                    continue
        return '未知'


    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        input_type: str = 'woe',
    ) -> 'RoundScoreCard':
        """训练评分卡（计分口径取整一致）。

        流程同 :meth:`ScoreCard.fit`（参数 ``X`` / ``y`` / ``sample_weight`` / ``input_type``
        含义一致），但额外按 ``decimal`` 精度对各特征分箱分数取整，并据此重算基础效应，
        确保后续计分与 :meth:`scorecard_points` 完全一致。

        :return: self
        """
        X_for_base_effect = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        result = super().fit(X, y=y, sample_weight=sample_weight, input_type=input_type)

        if isinstance(X_for_base_effect, pd.DataFrame) and y is None and self.target is not None and self.target in X_for_base_effect.columns:
            X_for_base_effect = X_for_base_effect.drop(columns=[self.target])

        if not isinstance(X_for_base_effect, pd.DataFrame):
            X_for_base_effect = pd.DataFrame(X_for_base_effect)

        if input_type == 'raw':
            X_woe = self._transform_to_woe(X_for_base_effect)
        else:
            X_woe = X_for_base_effect

        X_woe = X_woe[self.feature_names_]
        rounded_sub_scores = self._round_sub_scores_from_woe(X_woe, self.feature_names_)
        self.base_effect_ = pd.Series(np.median(rounded_sub_scores, axis=0), index=self.feature_names_)

        return result

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None,
        decimal: Optional[int] = None
    ) -> pd.DataFrame:
        """输出取整后的评分卡明细表（变量/分箱/分数/WOE）。

        同 :meth:`ScoreCard.scorecard_points`，但各分箱分数按 ``decimal``（默认取实例的
        ``self.decimal``）四舍五入，与本类实际计分完全一致，可直接交付落地。

        :param feature_map: 变量名到业务含义的映射，用于填充"变量含义"列，可选
        :param decimal: 覆盖分数保留小数位数，默认为 ``None``（用实例的 ``self.decimal``）
        :return: 含 ``变量名称`` / ``变量含义`` / ``变量分箱`` / ``对应分数`` / ``WOE值`` 的 DataFrame，
            首行为基础分
        """
        self._check_fitted()

        digits = self.decimal if decimal is None else decimal
        feature_map = feature_map or {}
        rows = [{
            '变量名称': '基础分',
            '变量含义': '截距项（基准分数）',
            '变量分箱': '-',
            '对应分数': self._get_rounded_base_score(digits),
            'WOE值': None,
        }]

        for col in self.feature_names_:
            if col not in self.rules_:
                continue

            rule = self.rules_[col]
            rounded_scores = self._get_rounded_rule_scores(rule, decimal=digits)
            bin_labels = rule.get('bin_labels')
            bins = rule.get('bins')
            # WOE 可能缺失（离线规则加载），单独取值，避免 zip 截断分箱行
            woe_values = rule.get('woe')
            woe_list = list(woe_values) if woe_values is not None else []

            if bin_labels is not None and len(bin_labels) > 0:
                labels_to_use = bin_labels
            elif bins is not None and len(bins) > 0:
                labels_to_use = bins
            elif woe_list:
                labels_to_use = [f'WOE: {w:.4f}' for w in woe_list]
            else:
                labels_to_use = [f'箱{i}' for i in range(len(rounded_scores))]

            if len(labels_to_use) != len(rounded_scores):
                labels_to_use = self._format_bin_labels(bins if bins else labels_to_use, len(rounded_scores))

            for idx, (bin_label, score) in enumerate(zip(labels_to_use, rounded_scores)):
                woe = woe_list[idx] if idx < len(woe_list) else None
                rows.append({
                    '变量名称': col,
                    '变量含义': feature_map.get(col, ''),
                    '变量分箱': self._format_bin_display(bin_label),
                    '对应分数': self._round_score_value(score, digits),
                    'WOE值': round(float(woe), 4) if woe is not None and not pd.isna(woe) else None,
                })

        return pd.DataFrame(rows)

    def predict_score(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None,
        input_type: str = 'auto'
    ) -> np.ndarray:
        """预测评分（取整一致）。

        同 :meth:`ScoreCard.predict_score`，但分数按 ``decimal`` 精度取整；传入 ``X`` 时走
        评分卡规则计分（与 :meth:`predict` 一致），传入 ``proba`` 时由概率经刻度公式换算后取整。

        :param X: 特征数据（原始或 WOE，由 ``input_type`` 控制），与 ``proba`` 二选一
        :param proba: 坏样本概率数组，与 ``X`` 二选一
        :param input_type: 输入类型 ``'auto'`` / ``'raw'`` / ``'woe'``，默认为 ``'auto'``
        :return: 取整后的评分数组
        :raises ValidationError: ``X`` 与 ``proba`` 均未提供时
        """
        if X is not None:
            return self.predict(X, input_type=input_type)

        if proba is None:
            raise ValidationError("必须提供X或proba参数之一")

        scores = self.transform(proba)
        scores = self._clip_scores(scores)
        return self._round_score_array(scores)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        input_type: str = 'raw'
    ) -> np.ndarray:
        """预测评分（基于取整后的评分卡规则计分）。

        同 :meth:`ScoreCard.predict`，但以"基础分 + 各特征取整后分箱分"求和得到总分，
        结果与 :meth:`scorecard_points` 展示的分值逐项一致。

        :param X: 输入数据（原始或 WOE）
        :param input_type: 输入类型 ``'raw'`` / ``'woe'`` / ``'auto'``，默认为 ``'raw'``
        :return: 取整后的评分数组
        """
        if not self._skip_fit_check:
            self._check_fitted()
        elif not hasattr(self, '_is_fitted') or not self._is_fitted:
            if self.verbose:
                logger.info("使用预训练模型进行预测（未调用fit）")

        resolved = self._resolve_round_scoring_inputs(X, input_type=input_type)
        total_score = self._get_rounded_base_score() + resolved['sub_scores'].sum(axis=1)
        total_score = self._clip_scores(total_score)
        return self._round_score_array(total_score)

    def get_reason(self, X: Union[pd.DataFrame, np.ndarray], keep: int = 3) -> pd.DataFrame:
        """输出主要评分原因（基于取整后评分卡）。

        同 :meth:`ScoreCard.get_reason`，但用取整后的各特征分数计算偏离，原因与实际计分一致。

        :param X: 输入数据（原始或 WOE，自动识别）
        :param keep: 每个样本保留的主要原因个数，默认为 ``3``
        :return: 每行一个样本、含 ``reason`` 列的 DataFrame
        """
        self._check_fitted()
        resolved = self._resolve_round_scoring_inputs(X, input_type='auto')

        sub_scores = resolved['sub_scores']
        feature_names = resolved['feature_names']
        effect_diff = sub_scores - self._get_base_effect_values(feature_names)

        reasons_list = []
        for row_diff in effect_diff:
            top_indices = np.argsort(np.abs(row_diff))[::-1][:keep]
            reasons = []
            for idx in top_indices:
                feature = feature_names[idx]
                diff = row_diff[idx]
                direction = '降低' if diff < 0 else '提升'
                reasons.append(f"{feature}({direction}{self._format_score_text(abs(diff))}分)")
            reasons_list.append('; '.join(reasons))

        return pd.DataFrame({'reason': reasons_list})

    def get_detailed_score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: Optional[Union[int, list]] = None,
        include_reason: bool = True
    ) -> pd.DataFrame:
        """输出样本级评分明细（基于取整后评分卡）。

        同 :meth:`ScoreCard.get_detailed_score`，但各特征贡献分与总分均为取整后值，
        与 :meth:`scorecard_points` 及 :meth:`predict` 完全一致。

        :param X: 输入数据（原始或 WOE，自动识别）
        :param sample_idx: 仅输出指定样本（单下标或下标列表）；``None`` 表示全部
        :param include_reason: 是否附带主要原因列，默认为 ``True``
        :return: 多层列结构 DataFrame，含各特征原始值/分箱/WOE/分数与样本总分
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if sample_idx is not None:
            if isinstance(sample_idx, int):
                sample_idx = [sample_idx]
            X = X.iloc[sample_idx]

        resolved = self._resolve_round_scoring_inputs(X, input_type='auto')
        feature_names = resolved['feature_names']
        X_raw = resolved['X_raw']
        X_woe = resolved['X_woe']
        X_bins = resolved['X_bins']
        sub_scores = resolved['sub_scores']
        base_score = self._get_rounded_base_score()
        total_scores = self._round_score_array(base_score + sub_scores.sum(axis=1))

        data_dict = {
            ('样本信息', '样本索引'): [],
            ('样本信息', '总分'): [],
            ('样本信息', '截距分数'): [],
        }

        for col in feature_names:
            data_dict[(col, '原始值')] = []
            data_dict[(col, '分箱')] = []
            data_dict[(col, 'WOE')] = []
            data_dict[(col, '分数')] = []

        for i, (idx, row) in enumerate(X_raw.iterrows()):
            data_dict[('样本信息', '样本索引')].append(idx)
            data_dict[('样本信息', '总分')].append(self._round_score_value(total_scores[i]))
            data_dict[('样本信息', '截距分数')].append(self._get_rounded_base_score())

            for j, col in enumerate(feature_names):
                raw_value = row[col] if col in row.index else np.nan
                if X_woe is not None and col in X_woe.columns:
                    bin_label = self._find_bin_label_from_woe(col, X_woe.iloc[i][col])
                elif X_bins is not None and col in X_bins.columns:
                    bin_label = self._format_bin_display(X_bins.iloc[i][col])
                else:
                    bin_label = '未知'


                woe_value = X_woe.iloc[i][col] if X_woe is not None and col in X_woe.columns else None
                score = sub_scores[i, j]

                data_dict[(col, '原始值')].append(raw_value)
                data_dict[(col, '分箱')].append(bin_label)
                data_dict[(col, 'WOE')].append(round(float(woe_value), 4) if woe_value is not None else None)
                data_dict[(col, '分数')].append(self._round_score_value(score))

        result_df = pd.DataFrame(data_dict)
        result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)

        if include_reason:
            reasons = self._generate_reasons(X_woe, sub_scores, n_reasons=3)
            result_df[('评分分析', '评分原因')] = reasons

        return result_df

    def _generate_reasons(self, X_woe: pd.DataFrame, sub_scores: np.ndarray, n_reasons: int = 3) -> list:
        """基于调整精度后的子分生成评分原因."""
        feature_names = self.feature_names_
        effect_diff = sub_scores - self._get_base_effect_values(feature_names)

        reasons_list = []
        for i in range(len(sub_scores)):
            row_diff = effect_diff[i]
            top_indices = np.argsort(np.abs(row_diff))[::-1][:n_reasons]

            reasons = []
            for idx in top_indices:
                feature = feature_names[idx]
                diff = row_diff[idx]
                score = sub_scores[i, idx]

                if diff < 0:
                    reasons.append(
                        f"{feature}拉低{self._format_score_text(abs(diff))}分(当前{self._format_score_text(score)}分)"
                    )
                else:
                    reasons.append(
                        f"{feature}提升{self._format_score_text(abs(diff))}分(当前{self._format_score_text(score)}分)"
                    )

            reasons_list.append('; '.join(reasons))

        return reasons_list

    def _get_deployment_base_score_and_sign(self) -> Tuple[float, float]:
        """获取基于调整后评分卡的部署基础分和分数符号."""
        return self._get_rounded_base_score(), 1.0

    def _get_deployment_rules(self, decimal: int) -> Dict[str, List[Tuple[Any, float]]]:
        """获取基于调整后评分卡的部署规则定义."""
        deployment_rules: Dict[str, List[Tuple[Any, float]]] = {}
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}
        cat_bins = getattr(self.binner, '_cat_bins_', {}) if self.binner is not None else {}

        for feature in self.feature_names_:
            rule = self.rules_.get(feature)
            if not rule:
                continue

            descriptors = None
            rounded_scores = self._get_rounded_rule_scores(rule, decimal=self.decimal)

            if feature_types.get(feature) == 'categorical' and feature in cat_bins and len(cat_bins[feature]) == len(rounded_scores):
                descriptors = cat_bins[feature]
            elif rule.get('bin_labels') is not None and len(rule['bin_labels']) == len(rounded_scores):
                descriptors = rule['bin_labels']
            elif rule.get('bins') is not None and len(rule['bins']) == len(rounded_scores):
                descriptors = rule['bins']

            if descriptors is None:
                continue

            deployment_rules[feature] = [
                (descriptor, float(score))
                for descriptor, score in zip(descriptors, rounded_scores)
            ]

        return deployment_rules

    def export(
        self,
        to_json: Optional[str] = None,
        to_frame: bool = False,
        decimal: Optional[int] = None,
        include_meta: bool = True,
        compatibility: Optional[str] = None,
    ) -> Union[Dict, pd.DataFrame]:
        """导出取整后评分卡规则（用于落地部署/留档）。

        同 :meth:`ScoreCard.export`，但导出的分值为按 ``decimal`` 取整后的分数，与
        :meth:`scorecard_points` / :meth:`predict` 完全一致。

        :param to_json: JSON 文件保存路径，提供则同时落盘，默认为 ``None``
        :param to_frame: 为 ``True`` 时返回扁平 DataFrame，否则返回规则字典，默认为 ``False``
        :param decimal: 覆盖分数保留小数位，默认为 ``None``（用实例的 ``self.decimal``）
        :param include_meta: 是否在结果中包含刻度参数等元信息，默认为 ``True``
        :param compatibility: 显式导出 ``'toad'`` 或 ``'scorecardpipeline'`` 兼容格式
        :return: 规则字典或 DataFrame（取决于 ``to_frame``）
        """
        digits = self.decimal if decimal is None else decimal
        exported = super().export(
            to_json=None,
            to_frame=to_frame,
            decimal=digits,
            include_meta=include_meta,
            compatibility=compatibility,
        )

        # 规则文件必须保存“方向变换前”的取整分数。RoundScoreCard 在评分时会按
        # direction_ 应用一次符号/基础分变换；若导出 scorecard_points 中已经翻转的
        # 展示分，load 后会在 ascending 模式下发生第二次翻转。
        if include_meta and compatibility is None:
            if isinstance(exported, pd.DataFrame):
                exported.attrs['scorecard_meta']['rounded_scorecard'] = True
            else:
                exported['__meta__']['rounded_scorecard'] = True

        if to_json is not None:
            import json

            directory = os.path.dirname(to_json)
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = exported
            if isinstance(exported, pd.DataFrame):
                payload = self._coerce_scorecard_rules(exported)
            with open(to_json, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

        return exported
