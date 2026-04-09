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

from .logistic_regression import LogisticRegression
from .probability_to_score import StandardScoreTransformer


class ScoreCard(StandardScoreTransformer):
    """评分卡模型.

    将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。
    继承 StandardScoreTransformer 实现评分计算，统一参数命名。

    **参数**

    :param pdo: Point of Double Odds，odds增加rate倍时分数变化量，默认 60
    :param rate: 倍率，默认 2
        - odds增加的倍数
    :param base_odds: 基础 odds（违约概率/正常概率），默认 35
        - 表示在base_score对应的坏样本率
        - 例如：35:1 => 坏样本率 ~2.8%
    :param base_score: 基础 odds 对应的分数，默认 750
    :param step: score_odds_reference的步长，默认None(自动计算为pdo/10)
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

    :ivar A_: 刻度参数 A = base_score + B × ln(base_odds)
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
        其中: odds = P / (1 - P)
              A = base_score + B × ln(base_odds)
              B = pdo / ln(rate)

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
            'direction': 'descending',
            'base_odds': base_odds,
            'base_score': base_score,
            'pdo': pdo,
            'rate': rate,
            'step': step,
            'precision': 2,  # 评分卡保留2位小数
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
        
        # 评分参数通过父类 _compute_parameters 计算
        # A_ = base_score + B_ * ln(base_odds)
        # B_ = pdo / ln(rate)
        self.A_, self.B_ = self._compute_parameters()
        
        # 设置方向属性（父类 transform 方法需要）
        self.direction_ = self._determine_direction()
        
        # 保留旧参数名兼容（factor->B_, offset->A_）
        self.factor = self.B_
        self.offset = self.A_

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
    
    def _initialize_from_pretrained(self):
        """从预训练模型初始化规则和特征名."""
        # 从lr_model获取特征数量
        if hasattr(self.lr_model, 'coef_'):
            n_features = len(self.lr_model.coef_[0])
            # 尝试从binner获取特征名
            if hasattr(self.binner, 'bin_tables_') and self.binner.bin_tables_:
                # 使用分箱器中的特征名（优先）
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
            else:
                self._feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # 生成规则
            self._generate_rules_from_binner()
            self._is_fitted = True
    
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
            
            if self.binner is not None and hasattr(self.binner, 'bin_tables_'):
                if col in self.binner.bin_tables_:
                    bin_table = self.binner.bin_tables_[col]
                    if '分档WOE值' in bin_table.columns:
                        woe_values = bin_table['分档WOE值'].values
                        if '分箱标签' in bin_table.columns:
                            bin_labels = bin_table['分箱标签'].values
                            bins = self._parse_bin_labels(bin_labels)
            
            if woe_values is None:
                continue
                
            woe_values = np.asarray(woe_values)
            
            # 计算每个 WOE 对应的分数
            scores = [self._woe_to_point(woe, coef) for woe in woe_values]
            
            self.rules_[col] = {
                'bins': bins if bins is not None else woe_values,
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
            return LogisticRegression(**lr_params)

        # 4. 使用默认参数
        return LogisticRegression(
            calculate_stats=self.calculate_stats,
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
                if self.verbose:
                    print(f"使用 binner.transform(X, metric='woe') 进行 WOE 转换")
                return X_woe
            except Exception as e:
                if self.verbose:
                    print(f"binner.transform(X, metric='woe') 失败: {e}")
                # 尝试其他方法
                try:
                    X_woe = self.binner.transform_woe(X)
                    if self.verbose:
                        print(f"使用 binner.transform_woe(X) 进行 WOE 转换")
                    return X_woe
                except:
                    pass

        # 情况2：既有 binner 又有 encoder（toad/scp 风格）
        if self.binner is not None and self.encoder is not None:
            X_binned = self.binner.transform(X)
            X_woe = self.encoder.transform(X_binned)
            if self.verbose:
                print(f"使用 binner + encoder 进行 WOE 转换")
            return X_woe

        # 情况3：仅有 encoder
        if self.encoder is not None:
            X_woe = self.encoder.transform(X)
            if self.verbose:
                print(f"使用 encoder 进行 WOE 转换")
            return X_woe

        # 情况4：无转换器，假设输入已是 WOE 数据
        if self.verbose:
            print(f"无转换器配置，假设输入已是 WOE 数据")
        return X

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

            # 从 toad 的 encoder 获取
            elif self.encoder is not None and hasattr(self.encoder, 'get'):
                if col in self.encoder:
                    encoder_rule = self.encoder[col]
                    if isinstance(encoder_rule, dict):
                        woe_values = encoder_rule.get('woe')
                        values = encoder_rule.get('value')
                        # 尝试从 binner 获取 bins
                        if self.binner is not None and hasattr(self.binner, 'get'):
                            if col in self.binner:
                                binner_rule = self.binner[col]
                                if isinstance(binner_rule, dict):
                                    bins = binner_rule.get('bins')

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
                'bin_labels': bin_labels,  # 保存原始分箱标签
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
            if label_str.lower() in ('missing', '缺失', 'nan', 'none', 'null'):
                parsed_labels.append('missing')
                continue
            elif label_str.lower() in ('special', '特殊'):
                parsed_labels.append('special')
                continue
            
            # 匹配数值区间
            match = re.match(r'\((-inf|[\d.-]+),\s*([\d.]+)\]|\[([\d.]+),\s*(inf|[\d.]+)\)', label_str)
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
        
        scores = np.zeros((X.shape[0], len(feature_names)))
        
        for i, col in enumerate(feature_names):
            if col in X.columns:
                coef = self.coef_[i]
                scores[:, i] = -self.B_ * coef * X[col].values
        
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
                raise ValueError("必须提供X或proba参数之一")
            # 使用内部LR模型预测概率
            lr_model = self.lr_model_ if hasattr(self, 'lr_model_') and self.lr_model_ is not None else self.lr_model
            if lr_model is None:
                raise ValueError("未找到LR模型，请先调用fit()或传入预训练lr_model")
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
        feature_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """输出评分卡分箱信息及其对应的分数.
        
        支持从分箱器获取完整的分箱信息，包括:
        - 基础分（截距项对应的分数）
        - 数值特征分箱（区间格式）
        - 类别特征分箱
        - 缺失值分箱（标记为 'missing'）
        - 特殊值分箱（标记为 'special'）
        
        参考 scorecardpipeline 的实现方式，确保与分箱器格式兼容。
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
            '对应分数': round(float(intercept_score), 2),
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
            bins = rule.get('bins', [])
            
            # 确定要使用的分箱标签
            if bin_labels is not None and len(bin_labels) > 0:
                labels_to_use = bin_labels
            elif bins is not None and len(bins) > 0:
                labels_to_use = bins
            else:
                labels_to_use = [f'箱{i}' for i in range(len(scores))]
            
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
                    '对应分数': round(float(score), 2),
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
            elif bin_label.lower() in ('nan', 'none', 'null'):
                return '缺失值'
        return str(bin_label)

    def export(
        self,
        to_json: Optional[str] = None,
        to_frame: bool = False,
        decimal: int = 2
    ) -> Union[Dict, pd.DataFrame]:
        """导出评分卡规则，兼容 toad/scorecardpipeline 格式.

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :param to_frame: 是否返回 DataFrame 格式，默认为 False
        :param decimal: 分数保留小数位数，默认为 2
        :return: 评分卡规则字典或 DataFrame
            - 字典格式: {'feature': {'bin_label': score, ...}, ...}
            - DataFrame格式: columns=['name', 'value', 'score']
        """
        import json
        
        check_is_fitted(self)

        # 使用 scorecard_points 获取完整信息
        points_df = self.scorecard_points()
        
        # 构建与 toad/scorecardpipeline 兼容的格式
        card = {}
        for _, row in points_df.iterrows():
            feature = row['变量名称']
            bin_label = row['变量分箱']
            score = row['对应分数']
            
            if feature not in card:
                card[feature] = {}
            card[feature][bin_label] = round(float(score), decimal)

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
        from ...utils.io import save_pickle as _save_pickle

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
        from ...utils.io import load_pickle as _load_pickle

        return _load_pickle(file, engine=engine, compression=compression)

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

        intercept_score = self.A_ - self.B_ * self.intercept_

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

    def export_deployment_code(
        self,
        language: str = 'python',
        output_file: Optional[str] = None,
        function_name: str = 'calculate_score',
    ) -> str:
        """导出评分卡部署代码.

        支持生成 SQL、Python、Java 格式的评分卡计算代码，可直接用于生产部署。

        :param language: 目标语言，可选 'sql'/'python'/'java'，默认 'python'
        :param output_file: 输出文件路径，为 None 时仅返回字符串
        :param function_name: 函数/存储过程名称，默认 'calculate_score'
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

        card = self.export()
        base_score = float(self.base_score_) if hasattr(self, 'base_score_') else 0.0

        if language.lower() == 'sql':
            code = self._generate_sql(card, base_score, function_name)
        elif language.lower() == 'python':
            code = self._generate_python(card, base_score, function_name)
        elif language.lower() == 'java':
            code = self._generate_java(card, base_score, function_name)
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

    def _generate_sql(self, card: dict, base_score: float, func_name: str) -> str:
        """生成 SQL CASE WHEN 评分卡代码."""
        lines = [f"-- 评分卡 SQL 部署代码（自动生成）", f"-- base_score = {base_score}", ""]
        lines.append(f"SELECT")
        lines.append(f"    {base_score}")

        for feature, bins in card.items():
            lines.append(f"    + CASE")
            for bin_label, score in bins.items():
                cond = self._bin_label_to_sql_condition(feature, bin_label)
                lines.append(f"        WHEN {cond} THEN {score}")
            lines.append(f"        ELSE 0")
            lines.append(f"      END  -- {feature}")

        lines.append(f"    AS score")
        lines.append(f"FROM your_table;")
        return '\n'.join(lines)

    def _generate_python(self, card: dict, base_score: float, func_name: str) -> str:
        """生成 Python 评分卡函数代码."""
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
            lines.append(f'')
            lines.append(f'    # {feature}')
            lines.append(f'    val = row.get("{feature}")')
            first = True
            for bin_label, sc in bins.items():
                prefix = 'if' if first else 'elif'
                cond = self._bin_label_to_python_condition('val', bin_label)
                lines.append(f'    {prefix} {cond}:')
                lines.append(f'        score += {sc}')
                first = False

        lines.append(f'')
        lines.append(f'    return score')
        lines.append(f'')
        lines.append(f'')
        lines.append(f'def batch_{func_name}(df: pd.DataFrame) -> pd.Series:')
        lines.append(f'    """批量计算评分."""')
        lines.append(f'    return df.apply(lambda row: {func_name}(row.to_dict()), axis=1)')
        return '\n'.join(lines)

    def _generate_java(self, card: dict, base_score: float, func_name: str) -> str:
        """生成 Java 评分卡方法代码."""
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
            lines.append(f'')
            lines.append(f'        // {feature}')
            lines.append(f'        Object {self._safe_java_var(feature)} = row.get("{feature}");')
            first = True
            for bin_label, sc in bins.items():
                prefix = 'if' if first else 'else if'
                cond = self._bin_label_to_java_condition(self._safe_java_var(feature), bin_label)
                lines.append(f'        {prefix} ({cond}) {{')
                lines.append(f'            score += {sc};')
                lines.append(f'        }}')
                first = False

        lines.append(f'')
        lines.append(f'        return score;')
        lines.append(f'    }}')
        lines.append(f'}}')
        return '\n'.join(lines)

    @staticmethod
    def _bin_label_to_sql_condition(feature: str, label: str) -> str:
        """将分箱标签转为 SQL CASE WHEN 条件."""
        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{feature} IS NULL"
        if label in ('特殊值', 'special'):
            return f"1=1 /* special */"
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
        # 类别值
        return f"{feature} = '{label}'"

    @staticmethod
    def _bin_label_to_python_condition(var: str, label: str) -> str:
        """将分箱标签转为 Python 条件表达式."""
        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{var} is None or (isinstance({var}, float) and np.isnan({var}))"
        if label in ('特殊值', 'special'):
            return f"True  # special"
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
        return f"{var} == '{label}'"

    @staticmethod
    def _bin_label_to_java_condition(var: str, label: str) -> str:
        """将分箱标签转为 Java 条件表达式."""
        label = str(label).strip()
        if label in ('缺失值', 'missing', 'nan', 'null', 'None'):
            return f"{var} == null"
        if label in ('特殊值', 'special'):
            return f"true /* special */"
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
        return f"\"{label}\".equals({var})"

    @staticmethod
    def _safe_java_var(name: str) -> str:
        """将特征名转为合法的 Java 变量名."""
        import re
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if safe[0].isdigit():
            safe = 'f_' + safe
        return safe

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

    def export(
        self,
        to_json: Optional[str] = None,
        to_frame: bool = False,
        decimal: int = 2
    ) -> Union[Dict, pd.DataFrame]:
        """导出评分卡规则，兼容 toad/scorecardpipeline 格式.

        导出格式与 toad.ScoreCard.export() 和 scorecardpipeline.ScoreCard.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :param to_frame: 是否返回 DataFrame 格式，默认为 False
        :param decimal: 分数保留小数位数，默认为 2
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
            scores = rule['scores']

            if bins is None or len(bins) == 0:
                continue

            feature_rules = {}
            if isinstance(bins[0], (list, np.ndarray)):
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
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })
            return pd.DataFrame(rows)

        return card

    def load(
        self,
        from_json: Union[str, Dict],
        update: bool = False
    ) -> 'ScoreCard':
        """加载评分卡规则，兼容 toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载评分卡规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: 评分卡规则字典或 JSON 文件路径
            - 字典: {'feature': {'bin_label': score, ...}, ...}
            - 文件路径: 'scorecard_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
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
            card = from_json

        if not update:
            self.rules_ = {}
            self._feature_names = []

        # 解析规则
        for feature, feature_rules in card.items():
            if self._feature_names is None:
                self._feature_names = []
            if feature not in self._feature_names:
                self._feature_names.append(feature)

            bins = []
            scores = []

            for bin_label, score in feature_rules.items():
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
                'scores': np.array(scores),
            }

        # 计算基础效应
        if not hasattr(self, 'base_effect_') or self.base_effect_ is None:
            self.base_effect_ = np.zeros(len(self.feature_names_))

        self._is_fitted = True
        return self
