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

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import inspect

from ....exceptions import DependencyError, NotFittedError, ValidationError
from ..classical.logistic_regression import LogisticRegression
from .score_transformer import StandardScoreTransformer


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

    参考:
        - toad.ScoreCard
        - scorecardpipeline.ScoreCard
        - optbinning.Scorecard
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
        
        # 如果传入了预训练模型和binner，尝试生成规则
        if self.lr_model is not None and self.binner is not None:
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
            print(f"ScoreCard 初始化: pdo={pdo}, rate={rate}, base_odds={base_odds}, base_score={base_score}")
            print(f"  - A_ (offset)={self.A_:.4f}, B_ (factor)={self.B_:.4f}")
            if self.binner is not None:
                print(f"  - binner: {self.binner.__class__.__name__}, 支持WOE转换: {self._binner_is_woe_transformer}")

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

    def _initialize_from_pretrained(self):
        """从预训练模型初始化规则和特征名."""
        if hasattr(self.lr_model, 'ensure_positive_woe_coefficients'):
            self.lr_model.ensure_positive_woe_coefficients()

        # 从lr_model获取特征数量
        if hasattr(self.lr_model, 'coef_'):
            n_features = len(self.lr_model.coef_[0])
            # 尝试从binner获取特征名
            if hasattr(self.binner, 'bin_tables_') and self.binner.bin_tables_:
                # 使用分箱器中的特征名（优先）— hscredit 风格
                feature_names = list(self.binner.bin_tables_.keys())
                # 如果特征数量匹配，使用这些特征名
                if len(feature_names) >= n_features:
                    # 尝试匹配lr_model的特征名（如果存储了）
                    if hasattr(self.lr_model, 'feature_names_in_'):
                        self._feature_names = list(self.lr_model.feature_names_in_)
                    else:
                        # 使用前n_features个特征名
                        self._feature_names = feature_names[:n_features]
                else:
                    self._feature_names = [f'feature_{i}' for i in range(n_features)]
            elif hasattr(self.binner, 'splits_') and self.binner.splits_:
                # load() 导入规则后，bin_tables_ 可能为空，但 splits_ 有数据
                # 此时从 splits_ 获取特征名
                feature_names = list(self.binner.splits_.keys())
                if len(feature_names) >= n_features:
                    if hasattr(self.lr_model, 'feature_names_in_'):
                        self._feature_names = list(self.lr_model.feature_names_in_)
                    else:
                        self._feature_names = feature_names[:n_features]
                else:
                    self._feature_names = [f'feature_{i}' for i in range(n_features)]
            elif self._is_toad_like_combiner():
                # toad/scp 风格：从 combiner.rules 获取特征名
                feature_names = self._extract_external_binner_feature_names()
                if len(feature_names) >= n_features:
                    if hasattr(self.lr_model, 'feature_names_in_'):
                        self._feature_names = list(self.lr_model.feature_names_in_)
                    else:
                        self._feature_names = feature_names[:n_features]
                else:
                    self._feature_names = [f'feature_{i}' for i in range(n_features)]
            else:
                self._feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # 生成规则
            self._generate_rules_from_binner()
            self._is_fitted = True

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
        # 如果传入了预训练模型但未调用fit，直接返回预训练模型的系数
        # 支持从 lr_model 或 lr_model_ (从pipeline提取) 获取
        if self._skip_fit_check:
            if self.lr_model is not None:
                return self.lr_model.coef_[0]
            if self.lr_model_ is not None:
                return self.lr_model_.coef_[0]
        check_is_fitted(self)
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
        check_is_fitted(self)
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
        check_is_fitted(self)

        # 获取特征名称
        feature_names = self.feature_names_
        if not feature_names:
            n_features = len(self.coef_)
            feature_names = [f'feature_{i}' for i in range(n_features)]

        if importance_type == 'coef':
            # 使用系数绝对值
            importances = np.abs(self.coef_)
        elif importance_type == 'score_range':
            # 使用评分卡中的分数范围
            if not hasattr(self, 'rules_') or not self.rules_:
                raise ValueError("评分卡规则未生成，无法使用score_range类型")
            importances = []
            for feature in feature_names:
                if feature in self.rules_:
                    scores = [v['score'] for v in self.rules_[feature].values() if isinstance(v, dict) and 'score' in v]
                    if scores:
                        importances.append(max(scores) - min(scores))
                    else:
                        importances.append(0)
                else:
                    importances.append(0)
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
        check_is_fitted(self)
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
            print(f"从 pipeline 提取组件，共 {len(steps)} 个步骤:")

        for name, obj in steps:
            # 识别 LR 模型
            if self._is_lr_model(obj) and self.lr_model_ is None:
                self.lr_model_ = obj
                if self.verbose:
                    print(f"  - 识别到 LR 模型: {name} ({obj.__class__.__name__})")
                continue

            # 识别 binner（如果尚未传入）
            if self.binner is None and self._is_binner(obj):
                self.binner = obj
                if self.verbose:
                    print(f"  - 识别到分箱器: {name} ({obj.__class__.__name__})")
                continue

            # 识别 encoder
            if self.encoder is None and self._is_woe_encoder(obj):
                self.encoder = obj
                if self.verbose:
                    print(f"  - 识别到 WOE 转换器: {name} ({obj.__class__.__name__})")
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
                print("  - 分箱器支持直接 WOE 转换（hscredit 风格）")
            return

        # 方法2：检查 transform 是否支持 metric='woe' 参数
        if hasattr(self.binner, 'transform'):
            try:
                sig = inspect.signature(self.binner.transform)
                params = list(sig.parameters.keys())
                if 'metric' in params:
                    self._binner_is_woe_transformer = True
                    if self.verbose:
                        print("  - 分箱器支持 metric='woe' 参数")
                    return
            except (ValueError, TypeError):
                pass

        # 方法3：检查是否有专门的方法用于 WOE 转换
        if hasattr(self.binner, 'transform_woe') or hasattr(self.binner, 'woe_transform'):
            self._binner_is_woe_transformer = True
            if self.verbose:
                print("  - 分箱器有专门的 WOE 转换方法")
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
        1. 如果 binner 支持直接 WOE 转换（hscredit 风格），使用 binner.transform(X, metric='woe')
        2. 如果配置了 binner + encoder（toad 风格），先分箱再转 WOE
        3. 如果只有 encoder，直接使用 encoder
        4. 如果没有转换器，假设输入已是 WOE 数据

        :param X: 原始数据
        :return: WOE 数据
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 情况1：binner 支持直接 WOE 转换（hscredit 风格）
        if self._binner_is_woe_transformer and self.binner is not None:
            try:
                # 尝试使用 metric='woe' 参数
                X_woe = self.binner.transform(X, metric='woe')
                if isinstance(X_woe, pd.DataFrame):
                    X_woe.attrs['hscredit_encoding'] = 'woe'
                if self.verbose:
                    print(f"使用 binner.transform(X, metric='woe') 进行 WOE 转换")
                return X_woe
            except Exception as e:
                if self.verbose:
                    print(f"binner.transform(X, metric='woe') 失败: {e}")
                # 尝试其他方法
                try:
                    X_woe = self.binner.transform_woe(X)
                    if isinstance(X_woe, pd.DataFrame):
                        X_woe.attrs['hscredit_encoding'] = 'woe'
                    if self.verbose:
                        print(f"使用 binner.transform_woe(X) 进行 WOE 转换")
                    return X_woe
                except:
                    pass

        # 情况2：既有 binner 又有 encoder（toad/scp 风格）
        if self.binner is not None and self.encoder is not None:
            X_binned = self.binner.transform(X)
            X_woe = self.encoder.transform(X_binned)
            if isinstance(X_woe, pd.DataFrame):
                X_woe.attrs['hscredit_encoding'] = 'woe'
            if self.verbose:
                print(f"使用 binner + encoder 进行 WOE 转换")
            return X_woe

        # 情况3：仅有 encoder
        if self.encoder is not None:
            X_woe = self.encoder.transform(X)
            if isinstance(X_woe, pd.DataFrame):
                X_woe.attrs['hscredit_encoding'] = 'woe'
            if self.verbose:
                print(f"使用 encoder 进行 WOE 转换")
            return X_woe

        # 情况4：无转换器，假设输入已是 WOE 数据
        if self.verbose:
            print(f"无转换器配置，假设输入已是 WOE 数据")
        return X

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

                    if not bins or not bin_labels:
                        continue

                    # 创建分箱函数
                    def get_bin_label(value):
                        if pd.isna(value):
                            return '缺失值'

                        # 尝试匹配数值区间
                        try:
                            val = float(value)
                            for i, b in enumerate(bins):
                                if pd.isna(b):
                                    continue
                                b = float(b)
                                if i == 0 and val < b:
                                    return bin_labels[i]
                                elif i > 0:
                                    prev = float(bins[i-1])
                                    if not pd.isna(prev) and prev <= val < b:
                                        return bin_labels[i]
                            # 最后一个区间
                            if not pd.isna(bins[-1]) and val >= float(bins[-1]):
                                return bin_labels[-1]
                        except (TypeError, ValueError):
                            pass

                        # 尝试匹配类别值
                        for b in bins:
                            if isinstance(b, list):
                                if str(value) in [str(v) for v in b]:
                                    idx = bins.index(b)
                                    if idx < len(bin_labels):
                                        return bin_labels[idx]
                            elif str(value).strip() == str(b).strip():
                                idx = bins.index(b)
                                if idx < len(bin_labels):
                                    return bin_labels[idx]

                        return '其他'

                    X[col] = X[col].apply(get_bin_label)

                return X

        # 设置虚拟 binner
        self._rule_binner = RuleBasedBinner(self.rules_, self.feature_names_)
        self.binner = self._rule_binner
        self._binner_is_woe_transformer = False

    def _transform_to_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """将原始数据转换为分箱标签数据."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 如果有外部 binner，优先使用
        if self.binner is not None and not hasattr(self, '_rule_binner'):
            try:
                return self.binner.transform(X, metric='bins')
            except Exception as exc:
                # 如果外部 binner 不支持，尝试基于规则的转换
                pass

        # 如果有基于规则的 binner，使用它
        if hasattr(self, '_rule_binner'):
            try:
                return self._rule_binner.transform(X, metric='bins')
            except Exception as exc:
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
            print("=" * 60)
            print("ScoreCard.fit() 开始训练")
            print(f"输入数据类型: {type(X).__name__}, input_type={input_type}")

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
                    print(f"从X中提取target列 '{self.target}' 作为y")
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
                print("将原始数据转换为 WOE 数据...")
            X = self._transform_to_woe(X)
        # else: input_type == 'woe', 直接使用输入数据

        # 3. 记录特征名
        self._feature_names = X.columns.tolist()
        if self.verbose:
            print(f"特征数量: {len(self._feature_names)}")
            print(f"特征列表: {self._feature_names}")

        # 4. 构建并训练/获取 LR 模型
        self.lr_model_ = self._build_lr_model()

        # 如果 LR 模型未训练，则训练
        if not hasattr(self.lr_model_, 'coef_'):
            if self.verbose:
                print("训练 LR 模型...")
            self.lr_model_.fit(X, y, sample_weight=sample_weight)
        else:
            if self.verbose:
                print("使用预训练的 LR 模型")

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
            print(f"评分卡训练完成，总分 = 截距分数 + 各特征分数之和")
            print(f"截距分数: {self.A_ - self.B_ * self.intercept_:.2f}")
            print("=" * 60)

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
            
            # 匹配数值区间: (a, b] 或 (a, +inf)
            match = re.match(
                r'\((-inf|[-\d.]+),\s*(\+?inf|[-\d.]+)[)\]]',
                label_str
            )
            if match:
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

    @staticmethod
    def _normalize_rule_label(label: Any) -> str:
        """标准化规则标签，便于离线规则映射."""
        label_str = str(label).strip()
        label_lower = label_str.lower()

        if label_lower in ('missing', '缺失值', '缺失', 'nan', 'null'):
            return 'missing'

        if label_lower in ('special', '特殊值', '特殊'):
            return 'special'

        if label_str.startswith(('(', '[')):
            interval_match = re.match(r'^[\[(]\s*([^,]+)\s*,\s*([^\])]+)\s*[\])]$', label_str)
            if interval_match:
                lower = interval_match.group(1).strip().lower()
                upper = interval_match.group(2).strip().lower()

                lower = '-inf' if lower in ('-infinity', '-inf') else lower
                upper = '+inf' if upper in ('infinity', 'inf', '+inf') else upper

                return f'interval:{lower}:{upper}'

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
        proba: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """预测评分（通过LR模型概率）。

        继承自 StandardScoreTransformer 的 predict_score 方法，
        但使用 ScoreCard 内部的 LR 模型来预测概率。

        可通过传入X或proba之一来获取评分。

        :param X: 特征矩阵（WOE数据），用于预测概率
        :param proba: 直接传入预测概率（正类概率）
        :return: 评分数组

        **示例**

        >>> # 通过特征矩阵预测
        >>> scores = scorecard.predict_score(X_test_woe)

        >>> # 通过概率直接转换
        >>> proba = scorecard.lr_model_.predict_proba(X_test_woe)[:, 1]
        >>> scores = scorecard.predict_score(proba=proba)
        """
        if not self._skip_fit_check:
            check_is_fitted(self)

        if proba is None:
            if X is None:
                raise ValidationError("必须提供X或proba参数之一")
            # 使用内部LR模型预测概率
            lr_model = self.lr_model_ if hasattr(self, 'lr_model_') and self.lr_model_ is not None else self.lr_model
            if lr_model is None:
                raise NotFittedError("未找到LR模型，请先调用fit()或传入预训练lr_model")
            proba = lr_model.predict_proba(X)[:, 1]

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
            check_is_fitted(self)
        elif not hasattr(self, '_is_fitted') or not self._is_fitted:
            # 传入了预训练模型但未调用fit，使用预训练模型进行预测
            if self.verbose:
                print("使用预训练模型进行预测（未调用fit）")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        uses_loaded_rule_scoring = (
            getattr(self, '_loaded_intercept', None) is not None
            and self.lr_model_ is None
            and self.lr_model is None
        )

        # 检测输入数据类型
        is_woe_data = self._detect_input_type(X)

        if uses_loaded_rule_scoring:
            if input_type == 'woe' or (input_type == 'auto' and is_woe_data):
                raise ValueError("当前评分卡由离线规则加载，请传入原始数据并设置 input_type='raw'")

            feature_names = self.feature_names_
            X_bins = self._transform_to_bins(X)
            X_bins = X_bins[feature_names]
            sub_scores = self._bin_labels_to_score(X_bins, feature_names)
        else:
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

        # 如果方向是 ascending（欺诈分模式），翻转评分
        # descending: 概率越高分越低（信用分，默认）
        # ascending: 概率越高分越高（欺诈分）
        if self.direction_ == 'ascending':
            if self.lower is not None and self.upper is not None:
                total_score = self.lower + self.upper - total_score
            else:
                # 如果没有边界，围绕 base_score 翻转
                total_score = 2 * self.base_score - total_score

        # 应用边界限制（继承自父类）
        total_score = self._clip_scores(total_score)

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
             "备注": "odds 增加的倍率"},
            {"刻度项": "pdo", "刻度值": self.pdo,
             "备注": f"odds 增加 {self.rate} 倍时分数变化量"},
            {"刻度项": "B (pdo/ln(rate))", "刻度值": round(self.B_, 4),
             "备注": f"pdo / ln({self.rate})"},
            {"刻度项": "A (offset)", "刻度值": round(self.A_, 4),
             "备注": "base_score + B * ln(base_odds)"},
        ])

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None,
        decimal: int = 2
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
        check_is_fitted(self)

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
            woe_values = rule.get('woe', [])
            
            # 优先使用 bin_labels（完整的分箱标签）
            bin_labels = rule.get('bin_labels')
            bins = rule.get('bins')
            woe_values = rule.get('woe', [])
            
            # 确定要使用的分箱标签
            if bin_labels is not None and len(bin_labels) > 0:
                labels_to_use = bin_labels
            elif bins is not None and len(bins) > 0:
                labels_to_use = bins
            else:
                # 无分箱标签，使用 WOE 值格式化为标签
                labels_to_use = [f'WOE: {w:.4f}' for w in woe_values]
            
            if len(labels_to_use) != len(scores):
                # 如果标签和分数数量不匹配，重新生成标签
                labels_to_use = self._format_bin_labels(bins if bins else labels_to_use, len(scores))
            
            # 处理每个分箱
            for bin_label, score, woe in zip(labels_to_use, scores, woe_values):
                # 格式化特殊标签
                display_label = self._format_bin_display(bin_label)
                
                rows.append({
                    '变量名称': col,
                    '变量含义': feature_map.get(col, ''),
                    '变量分箱': display_label,
                    '对应分数': round(float(score), decimal),
                    'WOE值': round(float(woe), 4) if woe is not None else None
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

    def export(
        self,
        to_json: Optional[str] = None,
        to_frame: bool = False,
        decimal: int = 2,
        include_meta: bool = False
    ) -> Union[Dict, pd.DataFrame]:
        """导出评分卡规则，兼容 toad/scorecardpipeline 格式.

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :param to_frame: 是否返回 DataFrame 格式，默认为 False
        :param decimal: 分数保留小数位数，默认为 2
        :param include_meta: 是否包含元数据（用于 hscredit 的 load 方法）。
            - True: 导出完整格式，包含 __meta__ 信息，可被 ScoreCard.load() 完整恢复
            - False: 导出 toad/scorecardpipeline 兼容的简洁格式
        :return: 评分卡规则字典或 DataFrame
            - 字典格式(默认): {'feature': {'bin_label': score, ...}, ...}
            - 完整格式(include_meta=True): {'__meta__': {...}, 'feature': {...}, ...}
            - DataFrame格式: columns=['name', 'value', 'score']
        """
        import json

        check_is_fitted(self)

        # 使用 scorecard_points 获取完整信息
        points_df = self.scorecard_points(decimal=decimal)

        # 构建与 toad/scorecardpipeline 兼容的格式
        card = {}
        for _, row in points_df.iterrows():
            feature = row['变量名称']
            bin_label = row['变量分箱']
            score = row['对应分数']

            if feature not in card:
                card[feature] = {}
            card[feature][bin_label] = round(float(score), decimal)

        # 如果需要包含元数据（用于 hscredit load 方法）
        if include_meta:
            # 计算截距分数
            intercept_score = float(self.A_ - self.B_ * self.intercept_)

            card['__meta__'] = {
                'intercept_score': intercept_score,
                'base_score': float(self.base_score),
                'direction': self.direction_,
                'pdo': self.pdo,
                'rate': self.rate,
                'base_odds': self.base_odds,
                'lower': self.lower,
                'upper': self.upper,
                'A': float(self.A_),
                'B': float(self.B_),
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
                    continue  # 跳过元数据
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })
            return pd.DataFrame(rows)

        return card

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

        print(f"模型已保存至: {file}")
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

        return _load_pickle(file, engine=engine, compression=compression)

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
        decimal: int = 2,
        debug: bool = False
    ):
        """导出 PMML 文件.

        :param pmml_file: PMML 文件保存路径，默认 'scorecard.pmml'
        :param decimal: 特征子分保留小数位数，默认 2
        :param debug: 是否返回中间对象进行调试，默认 False
        :return: debug=True 时返回 PMMLPipeline，否则返回 None
        """
        try:
            from sklearn_pandas import DataFrameMapper
            from sklearn.linear_model import LinearRegression
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
            from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer
        except ImportError as e:
            raise DependencyError(
                "导出 PMML 需要安装 hscredit[pmml]，或至少安装 sklearn-pandas 和 sklearn2pmml"
            ) from e

        check_is_fitted(self)

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

            mapper.append(([var], [domain, Alias(transformer, f'__score_{var}', prefit=True)]))

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

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=True)),
        ])

        sample_df = pd.DataFrame(samples)
        sample_y = pd.Series(np.random.randint(0, 2, 20), name='score')
        pipeline.fit(sample_df, sample_y)
        
        pipeline.named_steps['scorecard'].coef_ = np.full(len(mapper), score_sign, dtype=float)
        pipeline.named_steps['scorecard'].intercept_ = float(base_score)

        try:
            sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=debug)
        except TypeError as exc:
            # sklearn2pmml can raise a spurious TypeError("object of type 'NoneType' has no len()")
            # in notebook environments while decoding Java subprocess output, even though the PMML
            # artifact has already been written successfully.
            pmml_exists = os.path.exists(pmml_file) and os.path.getsize(pmml_file) > 0
            is_output_decode_bug = "NoneType" in str(exc) and "len()" in str(exc)
            if not (pmml_exists and is_output_decode_bug):
                raise
            warnings.warn(
                "sklearn2pmml reported a subprocess output decoding error after generating the PMML file; "
                "continuing with the exported artifact.",
                RuntimeWarning,
            )
        print(f"PMML 文件已导出至: {pmml_file}")

        if debug:
            return pipeline

    def export_deployment_code(
        self,
        language: str = 'python',
        output_file: Optional[str] = None,
        function_name: str = 'calculate_score',
        decimal: int = 4,
    ) -> str:
        """导出评分卡部署代码.

        支持生成 SQL、Python、Java 格式的评分卡计算代码，可直接用于生产部署。

        :param language: 目标语言，可选 'sql'/'python'/'java'，默认 'python'
        :param output_file: 输出文件路径，为 None 时仅返回字符串
        :param function_name: 函数/存储过程名称，默认 'calculate_score'
        :param decimal: 分数保留小数位数，默认 4
        :return: 生成的部署代码字符串

        示例::

            >>> sc = ScoreCard(...)
            >>> sc.fit(X_train, y_train)
            >>> # 生成 SQL
            >>> sql = sc.export_deployment_code(language='sql', output_file='scorecard.sql')
            >>> # 生成 Python
            >>> py = sc.export_deployment_code(language='python', output_file='scorecard.py')
        """
        check_is_fitted(self)

        card = self._get_deployment_rules(decimal=decimal)
        base_score, score_sign = self._get_deployment_base_score_and_sign()
        base_score = round(float(base_score), decimal)

        if language.lower() == 'sql':
            code = self._generate_sql(card, base_score, function_name, score_sign=score_sign)
        elif language.lower() == 'python':
            code = self._generate_python(card, base_score, function_name, score_sign=score_sign)
        elif language.lower() == 'java':
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
    def _format_deployment_score(score: float, score_sign: float) -> float:
        """格式化部署导出时的特征分数符号."""
        adjusted = float(score_sign) * float(score)
        return 0.0 if adjusted == -0.0 else adjusted

    def _generate_sql(self, card: dict, base_score: float, func_name: str, score_sign: float = 1.0) -> str:
        """生成 SQL CASE WHEN 评分卡代码."""
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}

        lines = [f"-- 评分卡 SQL 部署代码（自动生成）", f"-- base_score = {base_score}", ""]
        lines.append(f"SELECT")
        lines.append(f"    {base_score}")

        for feature, bins in card.items():
            feature_type = feature_types.get(feature)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0
            lines.append(f"    + CASE")
            for bin_descriptor, score in bins:
                cond = self._bin_label_to_sql_condition(feature, bin_descriptor, special_codes=special_codes)
                adjusted_score = self._format_deployment_score(score, score_sign)
                lines.append(f"        WHEN {cond} THEN {adjusted_score}")
            lines.append(f"        ELSE {self._format_deployment_score(default_score, score_sign)}")
            lines.append(f"      END  -- {feature}")

        lines.append(f"    AS score")
        lines.append(f"FROM your_table;")
        return '\n'.join(lines)

    def _generate_python(self, card: dict, base_score: float, func_name: str, score_sign: float = 1.0) -> str:
        """生成 Python 评分卡函数代码."""
        special_codes = self._get_deployment_special_codes()
        feature_types = getattr(self.binner, 'feature_types_', {}) if self.binner is not None else {}

        lines = [
            f'"""评分卡 Python 部署代码（自动生成）"""',
            f'import numpy as np',
            f'import pandas as pd',
            f'',
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
            lines.append(f'    # {feature}')
            lines.append(f'    val = row.get("{feature}")')
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
            f'/**',
            f' * 评分卡 Java 部署代码（自动生成）',
            f' */',
            f'public class ScoreCard {{',
            f'',
            f'    public static double {func_name}(Map<String, Object> row) {{',
            f'        double score = {base_score};  // base_score',
        ]

        for feature, bins in card.items():
            feature_type = feature_types.get(feature)
            default_score = self._get_deployment_default_score(bins) if feature_type == 'categorical' else 0.0
            lines.append(f'')
            lines.append(f'        // {feature}')
            lines.append(f'        Object {self._safe_java_var(feature)} = row.get("{feature}");')
            first = True
            for bin_descriptor, sc in bins:
                prefix = 'if' if first else 'else if'
                cond = self._bin_label_to_java_condition(
                    self._safe_java_var(feature), bin_descriptor, special_codes=special_codes
                )
                adjusted_score = self._format_deployment_score(sc, score_sign)
                lines.append(f'        {prefix} ({cond}) {{')
                lines.append(f'            score += {adjusted_score};')
                lines.append(f'        }}')
                first = False
            lines.append(f'        else {{')
            lines.append(f'            score += {self._format_deployment_score(default_score, score_sign)};')
            lines.append(f'        }}')

        lines.append(f'')
        lines.append(f'        return score;')
        lines.append(f'    }}')
        lines.append(f'}}')
        return '\n'.join(lines)

    @staticmethod
    def _bin_label_to_sql_condition(feature: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 SQL CASE WHEN 条件."""
        if isinstance(label, (list, np.ndarray)):
            values = [str(value) for value in label if not pd.isna(value)]
            if not values:
                return f"{feature} IS NULL"
            escaped_values = [value.replace("'", "''") for value in values]
            if len(escaped_values) > 1:
                quoted_values = ', '.join(f"'{value}'" for value in escaped_values)
                return f"{feature} IN ({quoted_values})"
            return f"{feature} = '{escaped_values[0]}'"

        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{feature} IS NULL"
        if label in ('特殊值', 'special'):
            if special_codes:
                comparisons = []
                for code in special_codes:
                    if pd.isna(code):
                        comparisons.append(f"{feature} IS NULL")
                    elif isinstance(code, str):
                        escaped = code.replace("'", "''")
                        comparisons.append(f"{feature} = '{escaped}'")
                    else:
                        comparisons.append(f"{feature} = {code}")
                return ' OR '.join(comparisons)
            return '1=0 /* no special codes configured */'
        # 区间格式: [a, b) 或 (a, b] 或 (-inf, b) 等
        import re
        m = re.match(r'[\[\(]([-\d.inf+]+)\s*,\s*([-\d.inf+]+)[\]\)]', label)
        if m:
            lo, hi = m.group(1), m.group(2)
            conds = []
            left_closed = label[0] == '['
            right_closed = label[-1] == ']'
            if lo not in ('-inf', '-Infinity', '-inf'):
                op = '>=' if left_closed else '>'
                conds.append(f"{feature} {op} {lo}")
            if hi not in ('inf', 'Infinity', '+inf'):
                op = '<=' if right_closed else '<'
                conds.append(f"{feature} {op} {hi}")
            if conds:
                return ' AND '.join(conds)
            return f"1=1"
        escaped_label = label.replace("'", "''")
        return f"{feature} = '{escaped_label}'"

    @staticmethod
    def _bin_label_to_python_condition(var: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 Python 条件表达式."""
        if isinstance(label, (list, np.ndarray)):
            value_exprs = [repr(str(value)) for value in label if not pd.isna(value)]
            missing_cond = f"pd.isna({var})" if any(pd.isna(value) for value in label) else None
            if value_exprs and missing_cond:
                if len(value_exprs) == 1:
                    return f"({missing_cond}) or ({var} == {value_exprs[0]})"
                return f"({missing_cond}) or ({var} in {{{', '.join(value_exprs)}}})"
            if not value_exprs:
                return f"pd.isna({var})"
            if len(value_exprs) > 1:
                return f"{var} in {{{', '.join(value_exprs)}}}"
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
        import re
        m = re.match(r'[\[\(]([-\d.inf+]+)\s*,\s*([-\d.inf+]+)[\]\)]', label)
        if m:
            lo, hi = m.group(1), m.group(2)
            conds = []
            left_closed = label[0] == '['
            right_closed = label[-1] == ']'
            if lo not in ('-inf', '-Infinity', '-inf'):
                op = '>=' if left_closed else '>'
                conds.append(f"{var} {op} {lo}")
            if hi not in ('inf', 'Infinity', '+inf'):
                op = '<=' if right_closed else '<'
                conds.append(f"{var} {op} {hi}")
            if conds:
                return ' and '.join(conds)
            return 'True'
        return f"{var} == {label!r}"

    @staticmethod
    def _bin_label_to_java_condition(var: str, label: Any, special_codes: Optional[List[Any]] = None) -> str:
        """将分箱标签转为 Java 条件表达式."""
        if isinstance(label, (list, np.ndarray)):
            values = [str(value) for value in label if not pd.isna(value)]
            if not values:
                return f"{var} == null"
            escaped_values = [value.replace('\\', '\\\\').replace('"', '\\"') for value in values]
            if len(escaped_values) > 1:
                conditions = ' || '.join(f'"{value}".equals({var})' for value in escaped_values)
                return f"({conditions})"
            return f"\"{escaped_values[0]}\".equals({var})"

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
                        conditions.append(f"((Number){var}).doubleValue() == {float(code)}")
                return '(' + ' || '.join(conditions) + ')'
            return 'false'
        import re
        m = re.match(r'[\[\(]([-\d.inf+]+)\s*,\s*([-\d.inf+]+)[\]\)]', label)
        if m:
            lo, hi = m.group(1), m.group(2)
            conds = []
            left_closed = label[0] == '['
            right_closed = label[-1] == ']'
            if lo not in ('-inf', '-Infinity', '-inf'):
                op = '>=' if left_closed else '>'
                conds.append(f"((Number){var}).doubleValue() {op} {lo}")
            if hi not in ('inf', 'Infinity', '+inf'):
                op = '<=' if right_closed else '<'
                conds.append(f"((Number){var}).doubleValue() {op} {hi}")
            if conds:
                return ' && '.join(conds)
            return 'true'
        escaped_label = label.replace('\\', '\\\\').replace('"', '\\"')
        return f"\"{escaped_label}\".equals({var})"

    @staticmethod
    def _safe_java_var(name: str) -> str:
        """将特征名转为合法的 Java 变量名."""
        import re
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
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

        类别变量在分箱器里默认落到第 0 箱，因此部署导出也需要保持一致。
        """
        for descriptor, score in bins:
            if not self._is_missing_descriptor(descriptor) and not self._is_special_descriptor(descriptor):
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
            value_exprs = [repr(str(value)) for value in label if not pd.isna(value)]
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

        interval_match = re.match(r'^([\[(])\s*([^,]+)\s*,\s*([^\])]+)\s*([\])])$', label)
        if interval_match:
            left_bracket, lower, upper, right_bracket = interval_match.groups()
            conditions = []
            lower = lower.strip()
            upper = upper.strip()
            if lower not in ('-inf', '-Infinity', '-INF'):
                operator = '>=' if left_bracket == '[' else '>'
                conditions.append(f"{var} {operator} {lower}")
            if upper not in ('+inf', 'inf', 'Infinity', '+INF', 'INF'):
                operator = '<=' if right_bracket == ']' else '<'
                conditions.append(f"{var} {operator} {upper}")
            return ' and '.join(conditions) if conditions else 'True'

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
            odds_theoretical = np.exp((self.A_ - score_median) / self.B_)
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
        include_meta: bool = False
    ) -> Union[Dict, pd.DataFrame]:
        """导出评分卡规则，兼容 toad/scorecardpipeline 格式.

        导出格式与 toad.ScoreCard.export() 和 scorecardpipeline.ScoreCard.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :param to_frame: 是否返回 DataFrame 格式，默认为 False
        :param decimal: 分数保留小数位数，默认为 2
        :param include_meta: 是否额外导出重建评分所需元数据，默认为 False
        :return: 评分卡规则字典或 DataFrame
            - 字典格式: {'feature': {'bin_label': score, ...}, ...}
            - DataFrame格式: columns=['name', 'value', 'score']

        **示例**

        >>> card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> card.fit(X_woe, y, binner=binner)
        >>> 
        >>> # 导出为字典
        >>> rules = card.export()
        >>> # 返回格式: {'age': {'[18, 25)': 50, '[25, 35)': 45, ...}, ...}
        >>> 
        >>> # 导出并保存到 JSON 文件
        >>> rules = card.export(to_json='scorecard_rules.json')
        >>> 
        >>> # 导出为 DataFrame
        >>> df = card.export(to_frame=True)
        
        **与 toad/scorecardpipeline 的兼容性**

        导出的规则可以直接被 toad 和 scorecardpipeline 加载:
        
        >>> # toad 加载
        >>> import toad
        >>> toad_card = toad.ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> toad_card.load(rules)
        >>> 
        >>> # scorecardpipeline 加载
        >>> from scorecardpipeline import ScoreCard
        >>> scp_card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scp_card.load(rules)
        """
        import json
        
        check_is_fitted(self)

        # 构建与 toad 兼容的格式
        card = {}
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
            card['__meta__'] = {
                'intercept_score': intercept_score,
                'base_score': float(self.base_score),
                'direction': self.direction_,
                'pdo': self.pdo,
                'rate': self.rate,
                'base_odds': self.base_odds,
                'base_score': self.base_score,
                'lower': self.lower,
                'upper': self.upper,
                'A': float(self.A_),
                'B': float(self.B_),
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
            return pd.DataFrame(rows)

        return card

    def _apply_export_metadata(self, meta: Dict[str, Any]) -> None:
        """应用导出文件中的评分卡元数据."""
        self.pdo = meta.get('pdo', self.pdo)
        self.rate = meta.get('rate', self.rate)
        self.base_odds = meta.get('base_odds', self.base_odds)
        self.base_score = meta.get('base_score', self.base_score)
        self.lower = meta.get('lower', self.lower)
        self.upper = meta.get('upper', self.upper)

        direction = meta.get('direction')
        if direction is not None:
            self.direction = direction

        self.A_, self.B_ = self._compute_parameters()
        self.direction_ = self._determine_direction()

        intercept_score = meta.get('intercept_score')
        if intercept_score is not None:
            self._loaded_intercept = (self.A_ - float(intercept_score)) / self.B_

    def load(
        self,
        from_json: Union[str, Dict],
        update: bool = False,
        binner: Optional[Any] = None
    ) -> 'ScoreCard':
        """加载评分卡规则，兼容 toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载评分卡规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: 评分卡规则字典或 JSON 文件路径
            - 字典: {'feature': {'bin_label': score, ...}, ...}
            - 文件路径: 'scorecard_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :param binner: 可选的分箱器，用于对原始数据进行分箱后评分。
            - 如果提供，将用于 predict(input_type='raw') 时的数据转换
            - 如果不提供，将基于规则中的分箱信息进行转换
        :return: self，支持链式调用

        **示例**

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
        >>> toad_card.fit(X, y, combiner=combiner, transer=transformer)  # toad 保持原参数名
        >>> rules = toad_card.export()
        >>> 
        >>> # hscredit 加载
        >>> from hscredit.core.models import ScoreCard
        >>> card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> card.load(rules)
        """
        import json
        import re

        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                card = json.load(f)
        else:
            # 直接使用字典
            card = dict(from_json)

        meta = None
        if isinstance(card, dict):
            meta = card.pop('__meta__', None)

        if not update:
            self.rules_ = {}
            self._feature_names = []
            self.base_effect_ = None
            self._loaded_intercept = None

        if meta:
            self._apply_export_metadata(meta)

        # 解析规则
        for feature, feature_rules in card.items():
            if self._feature_names is None:
                self._feature_names = []
            if feature not in self._feature_names:
                self._feature_names.append(feature)

            bins = []
            scores = []
            bin_labels = []

            for bin_label, score in feature_rules.items():
                bin_labels.append(str(bin_label))
                scores.append(float(score))

                # 尝试解析区间标签
                # 格式如: [-inf, 25), [25, 35), [35, +inf)
                # 或类别: 'A, B', 'C, D'
                if ',' in bin_label and ('[' in bin_label or '(' in bin_label):
                    # 数值区间
                    try:
                        # 提取数字
                        nums = re.findall(r'[-+]?(?:\d*\.?\d+|inf)', bin_label)
                        if len(nums) == 2:
                            lower, upper = nums
                            # 只保留上界作为切分点（除了第一个区间）
                            if upper == '+inf':
                                pass  # 最后一个区间，不添加切分点
                            elif lower == '-inf':
                                bins.append(float(upper) if upper != 'inf' else np.inf)
                    except (ValueError, TypeError):
                        # 解析失败，作为类别处理
                        bins.append([bin_label])
                else:
                    # 类别值
                    if isinstance(bin_label, str) and ',' in bin_label:
                        # 多个类别值
                        vals = [v.strip() for v in bin_label.split(',')]
                        bins.append(vals)
                    else:
                        bins.append([bin_label])

            # 区分数值型和类别型
            is_numeric = False
            if bins and len(bins) > 0:
                if isinstance(bins[0], (int, float, np.number)):
                    is_numeric = True
                elif isinstance(bins[0], list) and len(bins[0]) > 0:
                    # 检查是否是数值
                    try:
                        float(bins[0][0])
                        is_numeric = True
                    except (ValueError, TypeError):
                        is_numeric = False

            if is_numeric and bins:
                # 数值型：转换为切分点列表
                numeric_bins = [b for b in bins if isinstance(b, (int, float, np.number))]
                splits = sorted(list(set(numeric_bins)))
            else:
                # 类别型：保持列表格式
                splits = bins

            self.rules_[feature] = {
                'bins': splits,
                'bin_labels': np.array(bin_labels, dtype=object),
                'scores': np.array(scores),
            }

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


class RoundScoreCard(ScoreCard):
    """按分箱分数精度进行一致性计分的评分卡模型.

    与 `ScoreCard` 不同，`RoundScoreCard` 会先将基础分和各特征分箱分数
    按初始化指定的 `decimal` 精度进行调整，再基于这份调整后的评分卡完成
    预测、原因分析与部署导出，确保对外结果与 `scorecard_points()` 完全一致。

    **参数**

    :param decimal: 评分卡分数保留小数位数默认 2
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

        uses_loaded_rule_scoring = (
            getattr(self, '_loaded_intercept', None) is not None
            and self.lr_model_ is None
            and self.lr_model is None
        )

        is_woe_data = self._detect_input_type(X)

        if uses_loaded_rule_scoring:
            if input_type == 'woe' or (input_type == 'auto' and is_woe_data):
                raise ValueError("当前 RoundScoreCard 由离线规则加载，请传入原始数据并设置 input_type='raw'")

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
        """训练按评分卡精度一致计分的评分卡模型."""
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
        """输出按初始化精度调整后的评分卡分箱信息及其分数."""
        check_is_fitted(self)

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
            woe_values = rule.get('woe', [])
            bin_labels = rule.get('bin_labels')
            bins = rule.get('bins')

            if bin_labels is not None and len(bin_labels) > 0:
                labels_to_use = bin_labels
            elif bins is not None and len(bins) > 0:
                labels_to_use = bins
            else:
                labels_to_use = [f'WOE: {w:.4f}' for w in woe_values]

            if len(labels_to_use) != len(rounded_scores):
                labels_to_use = self._format_bin_labels(bins if bins else labels_to_use, len(rounded_scores))

            for bin_label, score, woe in zip(labels_to_use, rounded_scores, woe_values):
                rows.append({
                    '变量名称': col,
                    '变量含义': feature_map.get(col, ''),
                    '变量分箱': self._format_bin_display(bin_label),
                    '对应分数': self._round_score_value(score, digits),
                    'WOE值': round(float(woe), 4) if woe is not None else None,
                })

        return pd.DataFrame(rows)

    def predict_score(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None,
        input_type: str = 'auto'
    ) -> np.ndarray:
        """预测评分，优先基于调整精度后的评分卡规则进行计算."""
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
        """基于调整精度后的评分卡规则预测评分."""
        if not self._skip_fit_check:
            check_is_fitted(self)
        elif not hasattr(self, '_is_fitted') or not self._is_fitted:
            if self.verbose:
                print("使用预训练模型进行预测（未调用fit）")

        resolved = self._resolve_round_scoring_inputs(X, input_type=input_type)
        total_score = self._get_rounded_base_score() + resolved['sub_scores'].sum(axis=1)
        total_score = self._clip_scores(total_score)
        return self._round_score_array(total_score)

    def get_reason(self, X: Union[pd.DataFrame, np.ndarray], keep: int = 3) -> pd.DataFrame:
        """获取基于调整精度后评分卡的主要评分原因."""
        check_is_fitted(self)
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
        """获取按调整后评分卡计算的样本详细评分信息."""
        check_is_fitted(self)

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
        include_meta: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """导出按调整后评分卡计算的规则定义."""
        import json

        check_is_fitted(self)
        digits = self.decimal if decimal is None else decimal
        points_df = self.scorecard_points(decimal=digits)

        card = {}
        for _, row in points_df.iterrows():
            feature = row['变量名称']
            bin_label = row['变量分箱']
            score = self._round_score_value(row['对应分数'], digits)
            card.setdefault(feature, {})[bin_label] = score

        if include_meta:
            card['__meta__'] = {
                'direction': self.direction_,
                'pdo': self.pdo,
                'rate': self.rate,
                'base_odds': self.base_odds,
                'base_score': self.base_score,
                'lower': self.lower,
                'upper': self.upper,
                'decimal': digits,
                'rounded_scorecard': True,
                'intercept_score': self._get_rounded_base_score(digits),
            }

        if to_json is not None:
            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(to_json, 'w', encoding='utf-8') as f:
                json.dump(card, f, ensure_ascii=False, indent=2)

        if to_frame:
            rows = []
            for name, feature_rules in card.items():
                if name == '__meta__':
                    continue
                for value, score in feature_rules.items():
                    rows.append({'name': name, 'value': value, 'score': score})
            return pd.DataFrame(rows)

        return card

