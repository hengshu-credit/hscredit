"""统一分箱接口 - 整合所有分箱方法.

提供统一的 fit/transform 接口，支持所有分箱方法：
- 基础方法: uniform, quantile, tree, chi
- 优化方法: best_ks, best_iv, mdlp
- 运筹规划: or_tools
- 高级方法: cart, kmeans, monotonic, genetic, smooth, kernel_density, best_lift, target_bad_rate

支持指定切割点 (user_splits) 和单调性约束。
使用 core.metrics 中的指标计算方法。
"""

from typing import Union, List, Dict, Optional, Any, Callable
import numpy as np
import pandas as pd
import warnings
from scipy import stats

from .base import BaseBinning
from .uniform_binning import UniformBinning
from .quantile_binning import QuantileBinning
from .tree_binning import TreeBinning
from .chi_merge_binning import ChiMergeBinning
from .best_ks_binning import BestKSBinning
from .best_iv_binning import BestIVBinning
from .mdlp_binning import MDLPBinning
from .or_binning import ORBinning, ORTOOLS_AVAILABLE
from .cart_binning import CartBinning
from .kmeans_binning import KMeansBinning
from .genetic_binning import GeneticBinning
from .smooth_binning import SmoothBinning
from .kernel_density_binning import KernelDensityBinning
from .best_lift_binning import BestLiftBinning
from .target_bad_rate_binning import TargetBadRateBinning
from .monotonic_binning import MonotonicBinning

# 从 metrics 导入指标计算方法
from ..metrics._binning import (
    woe_iv_vectorized,
    iv_for_splits,
    ks_for_splits,
    compare_splits_iv,
    compare_splits_ks,
)


class OptimalBinning(BaseBinning):
    """统一分箱接口 - 整合所有分箱方法.

    提供统一的 fit/transform 接口，支持所有分箱方法。
    融合 MonotonicBinning 的单调性约束功能。
    支持指定切割点 (user_splits) 和预分箱。

    **架构设计原则**

    1. **OptimalBinning 作为统一入口**：集成所有分箱方法，支持预分箱功能
    2. **独立分箱模块保持简单**：各个具体分箱类（如BestIVBinning、MDLPBinning等）
       只执行一次分箱，不包含预分箱逻辑
    3. **预分箱在 OptimalBinning 层面实现**：通过 prebinning 参数在统一接口层实现
       预分箱+二次分箱的两阶段分箱流程

    :param target: 目标变量列名，默认为'target'
    :param method: 分箱方法，可选:
        - 基础方法: 'uniform', 'quantile', 'tree', 'chi'
        - 优化方法: 'best_ks', 'best_iv', 'mdlp'
        - 运筹规划: 'or_tools'
        - 高级方法: 'cart', 'kmeans', 'monotonic', 'genetic',
                   'smooth', 'kernel_density', 'best_lift', 'target_bad_rate'
        默认为'mdlp'
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param monotonic: 坏样本率单调性约束，默认为False
        - False: 不要求单调性
        - True 或 'auto': 自动检测最佳趋势（允许单增、单减、正U、倒U）
        - 'auto_asc_desc': 自动检测，但只允许单增或单减（不允许U型）
        - 'auto_heuristic': 使用启发式方法自动检测
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
        - 'peak': 允许单峰形态(先升后降，倒U型)
        - 'valley': 允许单谷形态(先降后升，正U型)
        - 'peak_heuristic': 使用启发式方法检测峰值
        - 'valley_heuristic': 使用启发式方法检测谷值
    :param user_splits: 用户指定的切分点，支持:
        - Dict[str, List]: 每个特征的切分点，如 {'age': [25, 35, 45]}
        - Callable: 函数返回切分点
    :param prebinning: 预分箱方法，支持:
        - str: 预分箱方法名（所有VALID_METHODS中的方法都可作为预分箱方法）
        - BaseBinning: 预分箱器实例
        - Dict: 预分箱配置，如 {'method': 'cart', 'max_n_bins': 20}
        默认为None（不进行预分箱）
    :param prebinning_params: 预分箱参数，当prebinning为str时使用
    :param special_codes: 特殊值列表，默认为None
    :param cat_cutoff: 类别型变量处理阈值，默认为None
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False
    :param decimal: 数值型切分点保留的小数位数，默认为4
    :param kwargs: 其他分箱方法特定参数

    **示例**

    >>> from hscredit.core.binning import OptimalBinning
    >>> # 使用MDLP分箱（默认，直接分箱）
    >>> binner = OptimalBinning(method='mdlp', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> # 使用最优IV分箱（直接分箱）
    >>> binner = OptimalBinning(method='best_iv', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> # 使用CART分箱（直接分箱）
    >>> binner = OptimalBinning(method='cart', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> # 使用预分箱（MDLP先进行CART预分箱成20箱，再优化为5箱）
    >>> binner = OptimalBinning(method='mdlp', prebinning='cart', prebinning_params={'max_n_bins': 20})
    >>> binner.fit(X, y)
    >>> # 使用预分箱器实例
    >>> pre_binner = OptimalBinning(method='cart', max_n_bins=20)
    >>> binner = OptimalBinning(method='best_iv', prebinning=pre_binner)
    >>> binner.fit(X, y)
    >>> # 使用quantile预分箱（先将数据分成20等份，再进行MDLP分箱）
    >>> binner = OptimalBinning(method='mdlp', prebinning='quantile', prebinning_params={'max_n_bins': 20})
    >>> binner.fit(X, y)
    """

    # 所有支持的分箱方法
    VALID_METHODS = [
        'uniform', 'quantile', 'tree', 'chi',
        'best_ks', 'best_iv', 'mdlp', 'or_tools',
        'cart', 'kmeans', 'monotonic', 'genetic',
        'smooth', 'kernel_density', 'best_lift', 'target_bad_rate'
    ]

    def __init__(
        self,
        target: str = 'target',
        method: str = 'mdlp',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        monotonic: Union[bool, str] = False,
        missing_separate: bool = True,
        user_splits: Optional[Union[Dict[str, List], Callable]] = None,
        prebinning: Optional[Union[str, 'BaseBinning', Dict]] = None,
        prebinning_params: Optional[Dict] = None,
        special_codes: Optional[List] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        **kwargs
    ):
        if 'n_bins' in kwargs:
            raise ValueError("n_bins 参数已移除，请使用 max_n_bins")

        # 处理预分箱相关参数（从kwargs中提取）
        prebinning_params_from_kwargs = {}
        prebinning_keys = ['prebinning_method', 'prebinning_max_bins', 'prebinning_min_bins']
        for key in prebinning_keys:
            if key in kwargs:
                prebinning_params_from_kwargs[key.replace('prebinning_', '')] = kwargs.pop(key)
        
        super().__init__(
            target=target,
            missing_separate=missing_separate,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            monotonic=monotonic,
            special_codes=special_codes,
            cat_cutoff=cat_cutoff,
            random_state=random_state,
            verbose=verbose,
            decimal=decimal,
        )

        if method not in self.VALID_METHODS:
            raise ValueError(f"不支持的method: {method}，可选: {self.VALID_METHODS}")

        self.method = method
        self.user_splits = user_splits
        self.prebinning = prebinning
        # 保存原始 prebinning_params，不修改传入的参数（为了sklearn clone兼容）
        original_params = prebinning_params or {}
        # 合并从kwargs中提取的预分箱参数，创建新的字典而不是修改原字典
        merged_params = {**original_params, **prebinning_params_from_kwargs}
        self.prebinning_params = merged_params if merged_params else None
        
        # 清理kwargs，移除不应该传递给底层分箱器的参数
        self.kwargs = self._clean_kwargs(kwargs)
        self._binner = None
        self._prebinner = None
        self.monotonic_trend_ = {}
    
    def _clean_kwargs(self, kwargs: Dict) -> Dict:
        """清理kwargs，移除不应该传递给底层分箱器的参数.
        
        这些参数是OptimalBinning特有的，底层分箱器不需要。
        """
        # 需要过滤的参数列表
        invalid_keys = [
            'prebinning', 'prebinning_params', 'prebinning_method',
            'user_splits', 'method',
            'lift_refine', 'lift_focus_weight', 'sample_stability_weight',
            'lift_refine_max_bins', 'monotonic_bonus_weight'
        ]
        return {k: v for k, v in kwargs.items() if k not in invalid_keys}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'OptimalBinning':
        """拟合分箱.

        :param X: 训练数据
        :param y: 目标变量
        :param kwargs: 其他参数
        :return: 拟合后的分箱器
        """
        X, y = self._check_input(X, y)

        # 如果已经拟合过（例如通过import_rules），只计算统计信息
        if self._is_fitted:
            self._update_bin_stats(X, y)
            return self

        # 如果指定了 user_splits，优先使用
        if self.user_splits is not None:
            self._fit_with_user_splits(X, y)
        elif self.prebinning is not None:
            # 使用预分箱
            self._fit_with_prebinning(X, y)
        else:
            # 使用指定方法
            self._fit_with_method(X, y)

        # 统一后处理：围绕头尾Lift与样本稳定性微调切分点
        # 默认开启，可通过 lift_refine=False 关闭
        if self.kwargs.get('lift_refine', True) and self.method != 'uniform':
            self._refine_splits_for_lift_stability(X, y)

        # 统一收口约束：确保不同方法都遵守单调性/最小箱/最大箱限制
        self._apply_post_fit_constraints(X, y, enforce_monotonic=self.method != 'monotonic')

        self._is_fitted = True
        return self
    
    def _update_bin_stats(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """更新分箱统计信息（用于已导入规则的情况）.
        
        :param X: 特征数据
        :param y: 目标变量
        """
        for feature in self.splits_.keys():
            if feature not in X.columns:
                continue
            
            # 获取特征类型
            feature_type = self.feature_types_.get(feature, 'numerical')
            
            # 获取切分点
            splits = self.splits_[feature]
            
            # 对于类别型变量，优先使用_cat_bins_
            if feature_type == 'categorical' and feature in self._cat_bins_:
                bins = self._apply_bins(X[feature], self._cat_bins_[feature], feature_type, feature)
            else:
                bins = self._apply_bins(X[feature], splits, feature_type, feature)
            
            # 计算分箱统计
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

    def _fit_with_user_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """使用用户指定的切分点进行分箱."""
        for feature in X.columns:
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            # 获取切分点
            if callable(self.user_splits):
                splits = self.user_splits(X[feature])
            elif isinstance(self.user_splits, dict) and feature in self.user_splits:
                splits = self.user_splits[feature]
            else:
                # 如果没有指定该特征的切分点，使用默认方法
                splits = self._get_default_splits(X[feature], y, feature_type)

            if feature_type == 'numerical':
                # 数值型自定义切分点：允许用户传入 np.nan/None 表示缺失箱，
                # 实际切分点中自动忽略这些值（缺失值由 missing_separate 统一处理）
                numeric_splits = pd.to_numeric(pd.Series(list(splits)), errors='coerce')
                splits = numeric_splits[~numeric_splits.isna()].to_numpy(dtype=float)
                splits = np.unique(np.sort(splits))

                # 确保切分点在数据范围内
                x_non_missing = pd.to_numeric(X[feature], errors='coerce').dropna()
                if len(x_non_missing) > 0:
                    x_min, x_max = x_non_missing.min(), x_non_missing.max()
                    splits = splits[(splits > x_min) & (splits < x_max)]
                else:
                    splits = np.array([])

                self.splits_[feature] = self._round_splits(splits)
                self.n_bins_[feature] = len(self.splits_[feature]) + 1
            else:
                # 类别型特征
                splits = list(splits)
                
                # 检查是否为List[List]格式
                if len(splits) > 0 and isinstance(splits[0], list):
                    # List[List]格式，保存到_cat_bins_
                    self._cat_bins_[feature] = splits
                    # splits_保存为List[List]格式（用于export_rules）
                    self.splits_[feature] = splits
                    self.n_bins_[feature] = len(splits)
                else:
                    # 字符串格式（向后兼容）
                    self.splits_[feature] = splits
                    self.n_bins_[feature] = len(splits) + 1

            # 计算分箱统计
            bins = self._apply_bins(X[feature], self.splits_[feature], feature_type, feature)
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

    def _get_prebinning_params(self, override_dict: Optional[Dict] = None) -> Dict:
        """获取预分箱参数.
        
        :param override_dict: 覆盖默认参数的字典
        :return: 预分箱器参数字典
        """
        base_params = {
            'target': self.target,
            'max_n_bins': 20,  # 默认预分箱为20箱
            'min_n_bins': 2,
            'min_bin_size': self.min_bin_size,
            'max_bin_size': self.max_bin_size,
            'special_codes': self.special_codes,
            'cat_cutoff': self.cat_cutoff,
            'random_state': self.random_state,
            'verbose': False,  # 预分箱默认不输出详细信息
            'decimal': self.decimal,
        }
        
        # 应用 prebinning_params 中的参数
        if self.prebinning_params is not None:
            # 处理可能的键名映射（如 prebinning_method -> method）
            for key, value in self.prebinning_params.items():
                if key in ['method'] and key not in base_params:
                    continue  # method通过其他方式传递
                if key in base_params:
                    base_params[key] = value
        
        # 应用覆盖字典
        if override_dict:
            for key, value in override_dict.items():
                if key != 'method':  # method单独处理
                    base_params[key] = value
        
        return base_params
    
    def _fit_with_prebinning(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """使用预分箱进行分箱.
        
        先使用预分箱方法生成初始切分点，再使用主方法进行优化。
        参考optbinning的实现，如MDLP分箱前先用CART预分箱。
        
        支持的预分箱方法：所有 VALID_METHODS 中的方法都可以作为预分箱方法。
        """
        # 1. 创建预分箱器
        if isinstance(self.prebinning, BaseBinning):
            # 传入的是分箱器实例
            self._prebinner = self.prebinning
        elif isinstance(self.prebinning, str):
            # 传入的是方法名
            pre_params = self._get_prebinning_params()
            self._prebinner = OptimalBinning(method=self.prebinning, **pre_params)
        elif isinstance(self.prebinning, dict):
            # 传入的是配置字典
            pre_method = self.prebinning.get('method', 'cart')
            pre_params = self._get_prebinning_params(self.prebinning)
            self._prebinner = OptimalBinning(method=pre_method, **pre_params)
        else:
            raise ValueError(f"不支持的prebinning类型: {type(self.prebinning)}")
        
        # 2. 执行预分箱
        if self.verbose:
            print(f"执行预分箱: {self._prebinner.method}")
        self._prebinner.fit(X, y)
        
        # 3. 获取预分箱的切分点作为初始切分点
        pre_splits = {}
        for feature in X.columns:
            if feature in self._prebinner.splits_:
                pre_splits[feature] = self._prebinner.splits_[feature]
        
        # 4. 使用主方法，但限制在预分箱的切分点上
        # 对于支持初始切分点的方法，传入预分箱结果
        if self.method in ['best_ks', 'best_iv', 'chi', 'mdlp']:
            # 这些方法可以在预分箱基础上进一步优化
            self._fit_with_method_and_prebins(X, y, pre_splits)
        else:
            # 其他方法直接使用预分箱结果
            self.splits_ = self._prebinner.splits_
            self.n_bins_ = self._prebinner.n_bins_
            self.bin_tables_ = self._prebinner.bin_tables_
            self.feature_types_ = self._prebinner.feature_types_

    def _fit_with_method_and_prebins(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pre_splits: Dict[str, np.ndarray]
    ):
        """使用预分箱切分点进行优化分箱."""
        for feature in X.columns:
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type
            
            if feature_type == 'categorical':
                # 类别型特征直接使用预分箱结果
                if feature in self._prebinner.splits_:
                    self.splits_[feature] = self._prebinner.splits_[feature]
                    self.n_bins_[feature] = self._prebinner.n_bins_[feature]
                    self.bin_tables_[feature] = self._prebinner.bin_tables_[feature]
                continue
            
            # 数值型特征：在预分箱切分点基础上优化
            initial_splits = pre_splits.get(feature, np.array([]))

            # 兼容异常格式（如 tuple/list 嵌套），统一清洗为有序数值切分点
            if len(initial_splits) > 0:
                cleaned = []
                for v in list(initial_splits):
                    if isinstance(v, (list, tuple, np.ndarray)):
                        for t in v:
                            if pd.notna(t):
                                try:
                                    tv = float(t)
                                    if np.isfinite(tv):
                                        cleaned.append(tv)
                                except Exception:
                                    pass
                    else:
                        if pd.notna(v):
                            try:
                                vv = float(v)
                                if np.isfinite(vv):
                                    cleaned.append(vv)
                            except Exception:
                                pass
                initial_splits = np.unique(np.sort(np.array(cleaned, dtype=float))) if cleaned else np.array([])
            
            if len(initial_splits) == 0:
                # 没有预分箱切分点，使用默认方法
                self._fit_single_feature(X[[feature]], y, feature)
                continue
            
            # 使用预分箱切分点生成初始分箱
            x_clean = X[feature].dropna()
            y_clean = y[x_clean.index]
            
            # 基于预分箱切分点计算每个箱的统计信息
            bins = np.digitize(x_clean, initial_splits)
            
            # 根据主方法进行优化
            if self.method == 'best_iv':
                optimized_splits = self._optimize_iv_splits(
                    x_clean, y_clean, initial_splits
                )
            elif self.method == 'best_ks':
                optimized_splits = self._optimize_ks_splits(
                    x_clean, y_clean, initial_splits
                )
            elif self.method == 'chi':
                optimized_splits = self._optimize_chi_merge_splits(
                    x_clean, y_clean, initial_splits
                )
            elif self.method == 'mdlp':
                # MDLP: 如果预分箱超出限制，使用IV优化进行合并
                optimized_splits = self._optimize_mdlp_splits(
                    x_clean, y_clean, initial_splits
                )
            else:
                optimized_splits = initial_splits
            
            self.splits_[feature] = self._round_splits(optimized_splits)
            self.n_bins_[feature] = len(self.splits_[feature]) + 1
            
            # 计算最终分箱统计
            final_bins = self._apply_bins(X[feature], self.splits_[feature], feature_type)
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, final_bins
            )

    def _fit_single_feature(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str
    ):
        """对单个特征使用主方法分箱."""
        temp_binner = OptimalBinning(
            method=self.method,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            random_state=self.random_state,
            verbose=False,
            decimal=self.decimal,
        )
        temp_binner.fit(X, y)
        
        self.splits_[feature] = temp_binner.splits_[feature]
        self.n_bins_[feature] = temp_binner.n_bins_[feature]
        self.bin_tables_[feature] = temp_binner.bin_tables_[feature]

    def _optimize_iv_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """基于预分箱切分点优化IV."""
        if len(initial_splits) <= self.max_n_bins - 1:
            return initial_splits
        
        # 计算每个预分箱的IV贡献
        bins = np.digitize(x, initial_splits)
        bin_stats = []
        
        for bin_idx in range(len(initial_splits) + 1):
            mask = bins == bin_idx
            if mask.sum() == 0:
                continue
            
            y_bin = y[mask]
            bad_rate = y_bin.mean()
            bin_stats.append({
                'bin': bin_idx,
                'count': mask.sum(),
                'bad_rate': bad_rate,
                'split': initial_splits[bin_idx] if bin_idx < len(initial_splits) else None
            })
        
        # 基于IV贡献合并相邻箱，直到满足max_n_bins
        current_splits = list(initial_splits)
        
        while len(current_splits) >= self.max_n_bins:
            # 找到IV损失最小的合并方案
            min_iv_loss = float('inf')
            merge_idx = -1
            
            for i in range(len(current_splits)):
                temp_splits = current_splits[:i] + current_splits[i+1:]
                iv_loss = self._calculate_iv_loss(x, y, current_splits, temp_splits)
                
                if iv_loss < min_iv_loss:
                    min_iv_loss = iv_loss
                    merge_idx = i
            
            if merge_idx >= 0:
                current_splits.pop(merge_idx)
            else:
                break
        
        return np.array(current_splits)

    def _optimize_ks_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """基于预分箱切分点优化KS."""
        if len(initial_splits) <= self.max_n_bins - 1:
            return initial_splits
        
        # 类似IV优化，但基于KS统计量
        current_splits = list(initial_splits)
        
        while len(current_splits) >= self.max_n_bins:
            # 找到KS损失最小的合并方案
            min_ks_loss = float('inf')
            merge_idx = -1
            
            for i in range(len(current_splits)):
                temp_splits = current_splits[:i] + current_splits[i+1:]
                ks_loss = self._calculate_ks_loss(x, y, current_splits, temp_splits)
                
                if ks_loss < min_ks_loss:
                    min_ks_loss = ks_loss
                    merge_idx = i
            
            if merge_idx >= 0:
                current_splits.pop(merge_idx)
            else:
                break
        
        return np.array(current_splits)

    def _optimize_chi_merge_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """基于预分箱切分点进行卡方合并."""
        if len(initial_splits) <= self.max_n_bins - 1:
            return initial_splits
        
        current_splits = list(initial_splits)
        
        while len(current_splits) >= self.max_n_bins:
            min_chi2 = float('inf')
            merge_idx = -1
            
            for i in range(len(current_splits)):
                temp_splits = current_splits[:i] + current_splits[i+1:]
                chi2 = self._calculate_chi2_for_splits(x, y, temp_splits)
                
                if chi2 < min_chi2:
                    min_chi2 = chi2
                    merge_idx = i
            
            if merge_idx >= 0 and min_chi2 < 3.841:  # 卡方阈值
                current_splits.pop(merge_idx)
            else:
                break
        
        return np.array(current_splits)

    def _optimize_mdlp_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """基于预分箱切分点优化MDLP分箱.
        
        MDLP算法本身会根据信息增益自动确定分箱数。
        当使用预分箱时，如果预分箱的分箱数超过max_n_bins，
        使用坏样本率差异最小的策略合并相邻分箱。
        
        :param x: 特征数据
        :param y: 目标变量
        :param initial_splits: 预分箱切分点
        :return: 优化后的切分点
        """
        if len(initial_splits) <= self.max_n_bins - 1:
            return initial_splits
        
        # 使用坏样本率差异作为合并标准
        current_splits = list(initial_splits)
        
        while len(current_splits) >= self.max_n_bins:
            # 计算每个分箱的统计信息
            bins = np.digitize(x, current_splits)
            
            bin_stats = []
            for bin_idx in range(len(current_splits) + 1):
                mask = bins == bin_idx
                if mask.sum() > 0:
                    y_bin = y[mask]
                    bin_stats.append({
                        'bin': bin_idx,
                        'count': mask.sum(),
                        'bad_rate': y_bin.mean()
                    })
            
            # 找到坏样本率差异最小的相邻分箱
            min_diff = float('inf')
            merge_idx = -1
            
            for i in range(len(bin_stats) - 1):
                diff = abs(bin_stats[i]['bad_rate'] - bin_stats[i+1]['bad_rate'])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i
            
            # 合并选中的切分点
            if merge_idx >= 0:
                current_splits.pop(merge_idx)
            else:
                break
        
        return np.array(current_splits)

    def _calculate_iv_loss(
        self,
        x: pd.Series,
        y: pd.Series,
        splits_before: List,
        splits_after: List
    ) -> float:
        """计算合并前后的IV损失.

        IV损失 = 合并前IV - 合并后IV
        值越小表示合并带来的信息损失越小。

        使用 metrics.binning_metrics.compare_splits_iv 方法。
        """
        result = compare_splits_iv(
            x.values if isinstance(x, pd.Series) else x,
            y.values if isinstance(y, pd.Series) else y,
            np.array(splits_before),
            np.array(splits_after)
        )
        if isinstance(result, (tuple, list)):
            return float(result[0])
        return float(result)

    def _calculate_ks_loss(
        self,
        x: pd.Series,
        y: pd.Series,
        splits_before: List,
        splits_after: List
    ) -> float:
        """计算合并前后的KS损失.

        KS损失 = 合并前KS - 合并后KS
        值越小表示合并带来的区分度损失越小。

        使用 metrics.binning_metrics.compare_splits_ks 方法。
        """
        result = compare_splits_ks(
            x.values if isinstance(x, pd.Series) else x,
            y.values if isinstance(y, pd.Series) else y,
            np.array(splits_before),
            np.array(splits_after)
        )
        if isinstance(result, (tuple, list)):
            return float(result[0])
        return float(result)

    def _calculate_total_iv(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: List
    ) -> float:
        """计算给定切分点的总IV值.
        
        使用 metrics.binning_metrics.iv_for_splits 方法。
        """
        if len(splits) == 0:
            return 0.0
        return iv_for_splits(
            x.values if isinstance(x, pd.Series) else x,
            y.values if isinstance(y, pd.Series) else y,
            np.array(splits)
        )

    def _calculate_max_ks(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: List
    ) -> float:
        """计算给定切分点的最大KS值.
        
        使用 metrics.binning_metrics.ks_for_splits 方法。
        """
        if len(splits) == 0:
            return 0.0
        return ks_for_splits(
            x.values if isinstance(x, pd.Series) else x,
            y.values if isinstance(y, pd.Series) else y,
            np.array(splits)
        )

    def _calculate_chi2_for_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: List
    ) -> float:
        """计算给定切分点的卡方统计量."""
        if len(splits) == 0:
            return 0.0
        
        bins = np.digitize(x, splits)
        contingency = pd.crosstab(bins, y).values
        
        if contingency.shape[0] < 2:
            return 0.0
        
        row_sum = contingency.sum(axis=1)
        col_sum = contingency.sum(axis=0)
        total = contingency.sum()
        
        if total == 0:
            return 0.0
        
        expected = np.outer(row_sum, col_sum) / total
        expected = np.maximum(expected, 1e-10)
        chi2 = ((contingency - expected) ** 2 / expected).sum()
        
        return chi2


    def _evaluate_lift_stability_score(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        min_samples: int,
        focus_weight: float,
        sample_weight: float
    ) -> float:
        """评估切分点综合分数（越大越好）：头尾Lift + 稳定性 + 单调性倾向."""
        if len(splits) == 0:
            return -np.inf

        bins = np.digitize(x, splits)
        n_bins = len(splits) + 1
        total = len(y)
        total_bad = float(np.sum(y))

        if total == 0 or total_bad <= 0 or total_bad >= total:
            return -np.inf

        counts = np.bincount(bins, minlength=n_bins).astype(float)
        bad_counts = np.bincount(bins, weights=y, minlength=n_bins).astype(float)

        if np.any(counts < min_samples):
            return -np.inf

        bad_rates = bad_counts / np.maximum(counts, 1.0)
        overall_bad_rate = total_bad / total
        lifts = bad_rates / np.maximum(overall_bad_rate, 1e-10)

        # 1) 头尾Lift：优先看两端箱体（按数值顺序）
        edge_lift_left = float(lifts[0])
        edge_lift_right = float(lifts[-1])
        edge_strength = abs(edge_lift_right - edge_lift_left)
        edge_extreme = max(
            max(0.0, edge_lift_left - 1.0),
            max(0.0, 1.0 - edge_lift_left),
            max(0.0, edge_lift_right - 1.0),
            max(0.0, 1.0 - edge_lift_right),
        )

        # 2) 全局头尾（最大/最小Lift）
        max_lift = float(np.max(lifts))
        min_lift = float(np.min(lifts))
        global_tail_strength = max(0.0, max_lift - 1.0) + max(0.0, 1.0 - min_lift)

        tail_strength = 0.65 * edge_strength + 0.2 * edge_extreme + 0.15 * global_tail_strength

        # 3) 样本稳定性（极端箱不能太小）
        top_idx = int(np.argmax(lifts))
        bottom_idx = int(np.argmin(lifts))
        top_ratio = counts[top_idx] / total
        bottom_ratio = counts[bottom_idx] / total
        min_ratio = float(np.min(counts) / total)

        stability = (
            np.log1p(top_ratio * 100.0)
            + np.log1p(bottom_ratio * 100.0)
            + np.log1p(min_ratio * 100.0)
        )

        # 4) 单调性倾向：减少拐点，尤其在auto_asc_desc下
        diffs = np.diff(bad_rates)
        signs = np.sign(diffs)
        non_zero = signs[signs != 0]
        sign_changes = 0 if len(non_zero) <= 1 else int(np.sum(non_zero[1:] * non_zero[:-1] < 0))

        monotonic_bonus_weight = float(self.kwargs.get('monotonic_bonus_weight', 0.4))
        monotonic_bonus = -float(sign_changes)
        if self.monotonic in ['ascending', 'descending', 'auto_asc_desc']:
            monotonic_bonus *= 1.5

        iv_value = iv_for_splits(
            x.values if isinstance(x, pd.Series) else x,
            y.values if isinstance(y, pd.Series) else y,
            np.array(splits)
        )

        n_bins_score = np.log1p(max(0, len(splits)))

        return (
            focus_weight * tail_strength
            + sample_weight * stability
            + monotonic_bonus_weight * monotonic_bonus
            + 0.08 * iv_value
            + 0.25 * n_bins_score
        )

    def _refine_splits_for_lift_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """对已有分箱结果做局部搜索：先删点，再补点，兼顾头尾Lift与单调性."""
        methods_to_refine = {
            'uniform', 'quantile', 'tree', 'chi', 'best_ks', 'best_iv',
            'mdlp', 'cart', 'kmeans', 'genetic', 'smooth',
            'kernel_density', 'best_lift', 'target_bad_rate'
        }
        if self.method not in methods_to_refine:
            return

        min_samples_abs = self._get_min_samples(len(y))
        focus_weight = float(self.kwargs.get('lift_focus_weight', 3.0))
        sample_weight = float(self.kwargs.get('sample_stability_weight', 0.2))
        max_search_bins = int(self.kwargs.get('lift_refine_max_bins', self.max_n_bins))

        strict_mono = self.monotonic in ['ascending', 'descending', 'auto_asc_desc']

        def _is_ok_monotonic(xv: pd.Series, yv: pd.Series, sp: np.ndarray) -> bool:
            if not strict_mono:
                return True
            b = np.digitize(xv, sp)
            cnt = np.bincount(b, minlength=len(sp) + 1).astype(float)
            bad = np.bincount(b, weights=yv, minlength=len(sp) + 1).astype(float)
            br = bad / np.maximum(cnt, 1.0)
            d = np.diff(br)
            return bool(np.all(d >= -1e-10) or np.all(d <= 1e-10))

        for feature, splits in list(self.splits_.items()):
            if self.feature_types_.get(feature) != 'numerical':
                continue

            splits_arr = np.array(splits, dtype=float) if len(splits) > 0 else np.array([])
            if len(splits_arr) == 0:
                continue

            x = pd.to_numeric(X[feature], errors='coerce')
            valid_mask = x.notna()
            if self.special_codes:
                for code in self.special_codes:
                    valid_mask &= (x != code)

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            if len(x_valid) < max(min_samples_abs * self.min_n_bins, 50):
                continue

            current = np.unique(np.sort(splits_arr))
            best = current.copy()
            best_score = self._evaluate_lift_stability_score(
                x_valid, y_valid, best, min_samples_abs, focus_weight, sample_weight
            )

            # Step1: 删点搜索（提升鲁棒性）
            improved = True
            while improved and len(best) > 1 and len(best) >= self.min_n_bins:
                improved = False
                candidate_best = None
                candidate_score = best_score

                for i in range(len(best)):
                    cand = np.delete(best, i)
                    if len(cand) < max(1, self.min_n_bins - 1):
                        continue
                    if not _is_ok_monotonic(x_valid, y_valid, cand):
                        continue
                    score = self._evaluate_lift_stability_score(
                        x_valid, y_valid, cand, min_samples_abs, focus_weight, sample_weight
                    )
                    if score > candidate_score:
                        candidate_score = score
                        candidate_best = cand

                if candidate_best is not None:
                    best = candidate_best
                    best_score = candidate_score
                    improved = True

            # Step2: 补点搜索（避免过度合并，强化头尾区分）
            max_splits_allowed = max(1, max_search_bins - 1)
            pool = np.unique(np.quantile(x_valid, np.linspace(0.05, 0.95, 19)))
            pool = np.array([v for v in pool if np.isfinite(v)], dtype=float)

            improved = True
            while improved and len(best) < max_splits_allowed:
                improved = False
                candidate_best = None
                candidate_score = best_score

                for c in pool:
                    if np.any(np.isclose(best, c, atol=1e-10, rtol=0)):
                        continue
                    cand = np.unique(np.sort(np.append(best, c)))
                    if len(cand) > max_splits_allowed:
                        continue
                    if not _is_ok_monotonic(x_valid, y_valid, cand):
                        continue
                    score = self._evaluate_lift_stability_score(
                        x_valid, y_valid, cand, min_samples_abs, focus_weight, sample_weight
                    )
                    if score > candidate_score:
                        candidate_score = score
                        candidate_best = cand

                if candidate_best is not None:
                    best = candidate_best
                    best_score = candidate_score
                    improved = True

            if len(best) > 0 and not np.array_equal(best, current):
                self.splits_[feature] = self._round_splits(best)
                self.n_bins_[feature] = len(self.splits_[feature]) + 1
                bins = self._apply_bins(X[feature], self.splits_[feature], 'numerical', feature)
                self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)

    def _fit_with_method(

        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """使用指定方法进行分箱.
        
        注意：各个独立的分箱模块（如BestIVBinning、MDLPBinning等）
        保持简单，只执行一次分箱。预分箱功能是OptimalBinning特有的，
        通过prebinning参数在OptimalBinning层面实现。
        """
        # 基础参数（适用于大多数方法）- 过滤掉不相关的参数
        base_params = {
            'max_n_bins': self.max_n_bins,
            'min_n_bins': self.min_n_bins,
            'min_bin_size': self.min_bin_size,
            'max_bin_size': self.max_bin_size,
            'monotonic': self.monotonic,
            'special_codes': self.special_codes,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'decimal': self.decimal,
        }
        
        # 安全地更新参数，过滤无效参数
        for k, v in self.kwargs.items():
            if k not in ['prebinning', 'prebinning_params', 'prebinning_method', 'user_splits']:
                base_params[k] = v

        # 需要 target 参数的方法
        target_params = {
            'target': self.target,
            **base_params
        }

        # 需要 cat_cutoff 参数的方法
        full_params = {
            'target': self.target,
            'max_n_bins': self.max_n_bins,
            'min_n_bins': self.min_n_bins,
            'min_bin_size': self.min_bin_size,
            'max_bin_size': self.max_bin_size,
            'special_codes': self.special_codes,
            'cat_cutoff': self.cat_cutoff,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'decimal': self.decimal,
        }
        
        # 安全地更新参数
        for k, v in self.kwargs.items():
            if k not in ['prebinning', 'prebinning_params', 'prebinning_method', 'user_splits']:
                full_params[k] = v

        if self.method == 'uniform':
            self._binner = UniformBinning(**target_params)
        elif self.method == 'quantile':
            self._binner = QuantileBinning(**target_params)
        elif self.method == 'tree':
            self._binner = TreeBinning(**target_params)
        elif self.method == 'chi':
            self._binner = ChiMergeBinning(**target_params)
        elif self.method == 'best_ks':
            self._binner = BestKSBinning(**full_params)
        elif self.method == 'best_iv':
            self._binner = BestIVBinning(**full_params)
        elif self.method == 'mdlp':
            self._binner = MDLPBinning(**target_params)
        elif self.method == 'or_tools':
            if not ORTOOLS_AVAILABLE:
                raise ImportError(
                    "OR-Tools 未安装，无法使用 or_tools 方法。"
                    "请使用 pip install ortools 安装。"
                )
            or_params = full_params.copy()
            or_params['objective'] = self.kwargs.get('or_objective', 'iv')
            or_params['time_limit'] = self.kwargs.get('or_time_limit', 30)
            self._binner = ORBinning(**or_params)
        elif self.method == 'cart':
            self._binner = CartBinning(**full_params)
        elif self.method == 'kmeans':
            kmeans_params = {
                'max_n_bins': self.max_n_bins,
                'min_n_bins': self.min_n_bins,
                'min_bin_size': self.min_bin_size,
                'max_bin_size': self.max_bin_size,
                'special_codes': self.special_codes,
                'random_state': self.random_state,
                'verbose': self.verbose,
                'force_numerical': True,  # 强制作为数值型处理
            }
            kmeans_params.update(self.kwargs)
            self._binner = KMeansBinning(**kmeans_params)
        elif self.method == 'monotonic':
            mono_params = {
                'monotonic': self.monotonic if self.monotonic else 'auto',
                'max_n_bins': self.max_n_bins,
                'min_n_bins': self.min_n_bins,
                'min_bin_size': self.min_bin_size,
                'special_codes': self.special_codes,
                'random_state': self.random_state,
                'verbose': self.verbose,
            }
            mono_params.update(self.kwargs)
            self._binner = MonotonicBinning(**mono_params)
        elif self.method == 'genetic':
            self._binner = GeneticBinning(**base_params)
        elif self.method == 'smooth':
            self._binner = SmoothBinning(**base_params)
        elif self.method == 'kernel_density':
            kernel_params = {
                'target': self.target,
                'max_n_bins': self.max_n_bins,
                'min_n_bins': self.min_n_bins,
                'min_bin_size': self.min_bin_size,
                'monotonic': self.monotonic,
                'special_codes': self.special_codes,
                'missing_separate': self.missing_separate,
                'random_state': self.random_state,
                'verbose': self.verbose,
                'decimal': self.decimal,
            }
            for k, v in self.kwargs.items():
                if k not in ['prebinning', 'prebinning_params', 'prebinning_method', 'user_splits', 'max_bin_size', 'cat_cutoff']:
                    kernel_params[k] = v
            self._binner = KernelDensityBinning(**kernel_params)
        elif self.method == 'best_lift':
            self._binner = BestLiftBinning(**target_params)
        elif self.method == 'target_bad_rate':
            self._binner = TargetBadRateBinning(**base_params)

        self._binner.fit(X, y)

        # 复制属性
        self.splits_ = self._binner.splits_
        self.n_bins_ = self._binner.n_bins_
        self.bin_tables_ = self._binner.bin_tables_
        self.feature_types_ = self._binner.feature_types_
        if hasattr(self._binner, '_cat_bins_'):
            self._cat_bins_ = self._binner._cat_bins_

        if hasattr(self._binner, 'ks_stats_'):
            self.ks_stats_ = self._binner.ks_stats_
        if hasattr(self._binner, 'iv_stats_'):
            self.iv_stats_ = self._binner.iv_stats_
        if hasattr(self._binner, 'monotonic_trend_'):
            self.monotonic_trend_ = self._binner.monotonic_trend_

    def _get_default_splits(
        self,
        x: pd.Series,
        y: pd.Series,
        feature_type: str
    ):
        """获取默认切分点."""
        if feature_type == 'categorical':
            return x.dropna().unique().tolist()
        else:
            # 使用等频分箱
            x_clean = x.dropna()
            if len(x_clean) > 0:
                quantiles = np.linspace(0, 1, self.max_n_bins + 1)
                return np.percentile(x_clean, quantiles[1:-1] * 100)
            return np.array([])

    def _apply_bins(
        self,
        x: pd.Series,
        splits: Union[np.ndarray, List],
        feature_type: str,
        feature: Optional[str] = None
    ) -> np.ndarray:
        """应用分箱.
        
        对于类别型变量，支持两种格式：
        1. List[List]格式: [['A', 'B'], ['C'], [np.nan]]
        2. 字符串列表格式: ['A,B', 'C'] (向后兼容)
        """
        if isinstance(splits, list):
            bins = np.zeros(len(x), dtype=int)
            # 类别型比较时，将 Series 转为字符串以避免类型不匹配导致的静默失败
            x_str = x.astype(str).where(x.notna(), other=np.nan)
            
            # 检查是否为List[List]格式（类别型变量的新格式）
            if len(splits) > 0 and isinstance(splits[0], list):
                # List[List]格式: [['A', 'B'], ['C'], [np.nan]]
                for i, group in enumerate(splits):
                    if isinstance(group, list):
                        for value in group:
                            # 处理np.nan
                            if isinstance(value, float) and np.isnan(value):
                                bins[x.isna()] = i
                            else:
                                bins[x_str == str(value)] = i
                    else:
                        # 单个值（向后兼容）
                        bins[x_str == str(group)] = i
            else:
                # 字符串列表格式: ['A,B', 'C'] (向后兼容)
                for i, cat in enumerate(splits):
                    if ',' in str(cat):
                        cats = str(cat).split(',')
                        for c in cats:
                            bins[x_str == c.strip()] = i
                    else:
                        bins[x_str == str(cat)] = i
                bins[x.isna()] = -1
            
            if self.special_codes:
                for code in self.special_codes:
                    bins[(x == code) | (x_str == str(code))] = -2
            return bins
        else:
            bins = np.zeros(len(x), dtype=int)

            if self.missing_separate:
                bins[x.isna()] = -1

            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2

            mask = x.notna()
            if self.special_codes:
                for code in self.special_codes:
                    mask = mask & (x != code)

            if len(splits) > 0:
                bins[mask] = np.digitize(x[mask], splits)
            else:
                bins[mask] = 0

            return bins

    def _apply_monotonic_adjustment(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """基于当前 method 切分点执行单调性收口。"""
        super()._apply_monotonic_adjustment(X, y)


    def _is_peak_pattern(self, bad_rates: np.ndarray) -> bool:
        """检查是否为峰值模式（倒U型）.

        :param bad_rates: 坏样本率数组
        :return: 是否为峰值模式
        """
        if len(bad_rates) < 3:
            return False

        t = np.argmax(bad_rates)
        if t == 0 or t == len(bad_rates) - 1:
            return False

        # 检查前半部分递增，后半部分递减
        left_asc = np.all(bad_rates[1:t+1] - bad_rates[:t] >= -1e-10)
        right_desc = np.all(bad_rates[t+1:] - bad_rates[t:-1] <= 1e-10)

        return left_asc and right_desc

    def _is_valley_pattern(self, bad_rates: np.ndarray) -> bool:
        """检查是否为谷值模式（U型）.

        :param bad_rates: 坏样本率数组
        :return: 是否为谷值模式
        """
        if len(bad_rates) < 3:
            return False

        t = np.argmin(bad_rates)
        if t == 0 or t == len(bad_rates) - 1:
            return False

        # 检查前半部分递减，后半部分递增
        left_desc = np.all(bad_rates[1:t+1] - bad_rates[:t] <= 1e-10)
        right_asc = np.all(bad_rates[t+1:] - bad_rates[t:-1] >= -1e-10)

        return left_desc and right_asc

    def _get_bin_labels_dict(
        self,
        splits: Union[np.ndarray, List],
        feature_type: str,
        feature: Optional[str] = None
    ) -> Dict[int, str]:
        """生成分箱标签字典.
        
        对于类别型变量，支持List[List]格式。
        """
        labels = {}
        labels[-1] = 'missing'
        labels[-2] = 'special'
        
        # 对于类别型变量，优先使用_cat_bins_
        if feature_type == 'categorical' and feature and feature in self._cat_bins_:
            cat_bins = self._cat_bins_[feature]
            for i, group in enumerate(cat_bins):
                if isinstance(group, list):
                    # List[List]格式
                    # 将np.nan转换为字符串"nan"或保持为np.nan
                    group_str = [str(v) if not (isinstance(v, float) and np.isnan(v)) else 'nan' 
                                for v in group]
                    labels[i] = ','.join(group_str)
                else:
                    labels[i] = str(group)
        elif isinstance(splits, list) and len(splits) > 0:
            # 字符串列表格式（向后兼容）
            if isinstance(splits[0], list):
                # List[List]格式
                for i, group in enumerate(splits):
                    if isinstance(group, list):
                        group_str = [str(v) if not (isinstance(v, float) and np.isnan(v)) else 'nan' 
                                    for v in group]
                        labels[i] = ','.join(group_str)
                    else:
                        labels[i] = str(group)
            else:
                # 字符串列表格式
                for i, cat in enumerate(splits):
                    labels[i] = str(cat)
        elif isinstance(splits, np.ndarray) and len(splits) > 0:
            # 数值型切分点（左闭右开 [a, b) 格式）
            sp_l = [-np.inf] + splits.tolist() + [np.inf]
            for i in range(len(sp_l) - 1):
                labels[i] = f'[{sp_l[i]:.2f}, {sp_l[i+1]:.2f})'
        else:
            # 只有一个箱
            labels[0] = 'all'
        
        return labels

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.
        
        将原始特征值转换为分箱索引、分箱标签或WOE值。
        
        :param X: 待转换数据, DataFrame或数组格式
        :param metric: 转换类型, 可选值:
            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理
            - 'bins': 返回分箱标签字符串, 用于可视化或报告
            - 'woe': 返回WOE值, 用于逻辑回归建模
        :param kwargs: 其他参数
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
        >>> binner = OptimalBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        if self._binner is not None:
            result = self._binner.transform(X, metric=metric, **kwargs)
            if metric == 'woe' and isinstance(result, pd.DataFrame):
                result.attrs['hscredit_encoding'] = 'woe'
                result.attrs['hscredit_source'] = 'OptimalBinning'
            return result

        # 直接使用本类的分箱逻辑
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            splits = self.splits_[feature]
            feature_type = self.feature_types_[feature]
            
            # 对于类别型变量，优先使用_cat_bins_
            if feature_type == 'categorical' and feature in self._cat_bins_:
                bins = self._apply_bins(X[feature], self._cat_bins_[feature], feature_type, feature)
            else:
                bins = self._apply_bins(X[feature], splits, feature_type, feature)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels_dict = self._get_bin_labels_dict(splits, feature_type, feature)
                result[feature] = [labels_dict.get(b, f'bin_{b}') for b in bins]
            elif metric == 'woe':
                # 优先使用_woe_maps_（从export/load导入）
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"不支持的metric: {metric}")

        if metric == 'woe':
            result.attrs['hscredit_encoding'] = 'woe'
            result.attrs['hscredit_source'] = 'OptimalBinning'

        return result

    def get_stats(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """获取分箱统计信息."""
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合")

        if feature is not None:
            if feature not in self.bin_tables_:
                raise KeyError(f"特征 '{feature}' 未找到")

            stats = {
                'n_bins': self.n_bins_[feature],
                'bin_table': self.bin_tables_[feature],
            }

            if hasattr(self, 'ks_stats_') and feature in self.ks_stats_:
                stats['ks'] = self.ks_stats_[feature]
            if hasattr(self, 'iv_stats_') and feature in self.iv_stats_:
                stats['iv'] = self.iv_stats_[feature]
            if feature in self.monotonic_trend_:
                stats['monotonic_trend'] = self.monotonic_trend_[feature]

            return stats
        else:
            return {f: self.get_stats(f) for f in self.splits_.keys()}

    @staticmethod
    def auto_select_method(
        X: pd.DataFrame,
        y: pd.Series,
        feature: str,
        methods: Optional[List[str]] = None,
        criterion: str = 'iv'
    ) -> str:
        """自动选择最优分箱方法.
        
        :param X: 特征数据
        :param y: 目标变量
        :param feature: 特征名
        :param methods: 待评估的方法列表，默认为所有方法
        :param criterion: 选择标准，'iv' 或 'ks'
        :return: 最优方法名
        """
        if methods is None:
            methods = ['uniform', 'quantile', 'tree', 'chi',
                      'best_ks', 'best_iv', 'mdlp', 'cart', 'kmeans']

        best_method = methods[0]
        best_score = -1

        for method in methods:
            try:
                binner = OptimalBinning(method=method, verbose=False)
                binner.fit(X[[feature]], y)

                if criterion == 'iv' and hasattr(binner, 'iv_stats_'):
                    score = binner.iv_stats_.get(feature, 0)
                elif criterion == 'ks' and hasattr(binner, 'ks_stats_'):
                    score = binner.ks_stats_.get(feature, 0)
                else:
                    bin_table = binner.bin_tables_.get(feature)
                    if bin_table is not None:
                        # 使用中文列名
                        if '分档IV值' in bin_table.columns:
                            score = bin_table['分档IV值'].sum()
                        else:
                            score = 0
                    else:
                        score = 0

                if score > best_score:
                    best_score = score
                    best_method = method

            except Exception as e:
                warnings.warn(f"方法 {method} 在特征 {feature} 上失败: {e}")
                continue

        return best_method


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("OptimalBinning - 统一分箱接口测试")
    print("=" * 70)
    
    import numpy as np
    import pandas as pd
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n),
        'feature2': np.random.uniform(0, 100, n),
    })
    y = (X['feature1'] + X['feature2'] / 100 > 0.5).astype(int)
    
    # 测试各种方法
    methods_to_test = ['uniform', 'quantile', 'tree', 'chi',
                       'best_iv', 'cart', 'kmeans', 'mdlp']
    
    for method in methods_to_test:
        print(f"\n测试方法: {method}")
        print("-" * 50)
        try:
            binner = OptimalBinning(method=method, max_n_bins=5, verbose=False)
            binner.fit(X, y)
            
            table = binner.get_bin_table('feature1')
            print(f"  分箱数: {len(table)}")
            print(f"  前3行:\n{table[['bin', 'count', 'bad_rate']].head(3)}")
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
