"""单特征规则挖掘模块.

基于单个特征的阈值进行规则挖掘，支持hscredit所有分箱方法。

代码风格参考hscredit的binning模块，fit方法兼容scorecardpipeline风格。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency
import warnings

from .base import BaseRuleMiner, calculate_lift
from ...core.rules.rule import Rule
from ...core.binning import OptimalBinning, ChiMergeBinning, QuantileBinning


class SingleFeatureRuleMiner(BaseRuleMiner):
    """单特征规则挖掘器.
    
    对数据各特征的不同阈值进行效度分布分析，挖掘高区分度的单特征规则。
    
    代码风格参考hscredit的binning模块，fit方法兼容scorecardpipeline风格。
    支持hscredit中所有分箱方法。
    
    :param target: 目标变量列名，默认为'target'
    :param exclude_cols: 需要排除的列名列表
    :param method: 分箱方法，支持hscredit中所有分箱方法:
        - 'quantile': 等频分箱 (默认)
        - 'chi2': 卡方分箱 (ChiMergeBinning)
        - 'uniform': 等宽分箱
        - 'tree': 决策树分箱
        - 'cart': CART分箱
        - 'optimal_iv': 最优IV分箱
        - 'optimal_ks': 最优KS分箱
        - 'mdlp': MDLP分箱
        - 'kmeans': KMeans分箱
        - 'best_lift': Best Lift分箱
        - 'monotonic': 单调性约束分箱
        - 'genetic': 遗传算法分箱
        - 'smooth': 平滑分箱
        - 'kernel_density': 核密度分箱
        - 'target_bad_rate': 目标坏样本率分箱
        默认为'quantile'
    :param max_n_bins: 最大分箱数，默认20。同binning模块的max_n_bins
    :param min_n_bins: 最小分箱数，默认2。同binning模块的min_n_bins
    :param min_bin_size: 每箱最小样本数或占比，默认0.05
        - 如果 < 1, 表示占比 (如 0.05 表示 5%)
        - 如果 >= 1, 表示绝对数量
    :param max_bin_size: 每箱最大样本数或占比，默认None
    :param monotonic: 坏样本率单调性约束，默认False
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
        - 'peak': 允许单峰形态(先升后降)
        - 'valley': 允许单谷形态(先降后升)
    :param chi2_threshold: 卡方分箱合并阈值，默认3.841 (p=0.05, df=1)
    :param significance_level: 卡方显著性水平，默认0.05
    :param min_lift: 最小lift阈值，用于筛选规则，默认1.1
    :param min_samples: 最小样本数，默认10。同binning模块的约束
    :param special_codes: 特殊值列表，这些值会被单独处理，默认None
    :param cat_cutoff: 类别型变量处理阈值，默认None
    :param random_state: 随机种子，默认None
    :param verbose: 是否输出详细信息，默认False
    :param binning_kwargs: 分箱方法的其他参数，通过**kwargs传入
    
    示例:
        >>> # 使用等频分箱
        >>> miner = SingleFeatureRuleMiner(target='ISBAD', method='quantile', max_n_bins=20)
        >>> miner.fit(df)
        >>> rules = miner.get_top_rules(top_n=10, metric='lift')
        >>> 
        >>> # 使用卡方分箱
        >>> miner = SingleFeatureRuleMiner(target='ISBAD', method='chi2', max_n_bins=10, chi2_threshold=3.841)
        >>> miner.fit(df)
        >>> 
        >>> # 使用最优IV分箱
        >>> miner = SingleFeatureRuleMiner(target='ISBAD', method='optimal_iv', max_n_bins=5, monotonic=True)
        >>> miner.fit(df)
        >>> 
        >>> # 分析单个特征
        >>> feature_rules = miner.analyze_feature('age', max_n_bins=10)
    """
    
    # 支持的分箱方法映射
    METHOD_MAPPING = {
        'quantile': 'quantile',
        'chi2': 'chi',
        'chi': 'chi',
        'uniform': 'uniform',
        'tree': 'tree',
        'cart': 'cart',
        'best_iv': 'best_iv',
        'best_ks': 'best_ks',
        'mdlp': 'mdlp',
        'kmeans': 'kmeans',
        'best_lift': 'best_lift',
        'monotonic': 'monotonic',
        'genetic': 'genetic',
        'smooth': 'smooth',
        'kernel_density': 'kernel_density',
        'target_bad_rate': 'target_bad_rate',
        'or_tools': 'or_tools',
    }
    
    def __init__(
        self,
        target: str = 'target',
        exclude_cols: Optional[List[str]] = None,
        method: str = 'mdlp',
        max_n_bins: int = 10,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        monotonic: Union[bool, str] = False,
        chi2_threshold: float = 3.841,
        significance_level: float = 0.05,
        min_lift: float = 1.5,
        min_samples: int = 10,
        special_codes: Optional[List] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        **binning_kwargs
    ):
        super().__init__(target=target, exclude_cols=exclude_cols)
        
        if method not in self.METHOD_MAPPING:
            raise ValueError(f"不支持的method: {method}，可选: {list(self.METHOD_MAPPING.keys())}")
        
        self.method = method
        self.max_n_bins = max_n_bins
        self.min_n_bins = min_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.monotonic = monotonic
        self.chi2_threshold = chi2_threshold
        self.significance_level = significance_level
        self.min_lift = min_lift
        self.min_samples = min_samples
        self.special_codes = special_codes
        self.cat_cutoff = cat_cutoff
        self.random_state = random_state
        self.verbose = verbose
        self.binning_kwargs = binning_kwargs
        
        self.results_ = {}
        self.features_ = []
        self.numerical_features_ = []
        self.categorical_features_ = []
        self.overall_badrate_ = 0.0
        self._binning_instances_ = {}
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'SingleFeatureRuleMiner':
        """拟合挖掘器.
        
        :param X: 训练数据，DataFrame或numpy数组
        :param y: 目标变量（可选）
        :param kwargs: 额外参数，可覆盖初始化参数
        :return: self
        """
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        X, y = self._check_input_data(X, y)
        
        if y is None:
            raise ValueError("单特征规则挖掘需要目标变量y")
        
        self.X_ = X
        self.y_ = y
        
        # 分类特征
        self.numerical_features_ = self._get_numeric_features(X)
        self.categorical_features_ = self._get_categorical_features(X)
        self.features_ = self.numerical_features_ + self.categorical_features_
        
        # 计算整体坏账率
        self.overall_badrate_ = y.mean()
        
        # 分析所有特征
        self.results_ = {}
        self._binning_instances_ = {}
        
        for feature in self.features_:
            try:
                result, binner = self._analyze_feature(feature)
                self.results_[feature] = result
                if binner is not None:
                    self._binning_instances_[feature] = binner
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"分析特征 '{feature}' 时出错: {str(e)}")
        
        self._is_fitted = True
        return self
    
    def _analyze_feature(self, feature: str) -> Tuple[pd.DataFrame, Any]:
        """分析单个特征的不同阈值.
        
        :param feature: 特征名
        :return: (包含各阈值指标的DataFrame, 分箱器实例或None)
        """
        if feature in self.numerical_features_:
            return self._analyze_numerical_feature(feature)
        else:
            return self._analyze_categorical_feature(feature), None
    
    def _get_binning_instance(self) -> Any:
        """获取分箱器实例.
        
        :return: hscredit分箱器实例
        """
        internal_method = self.METHOD_MAPPING[self.method]
        
        # 构建分箱器参数
        binning_params = {
            'target': self.target,
            'max_n_bins': self.max_n_bins,
            'min_n_bins': self.min_n_bins,
            'min_bin_size': self.min_bin_size,
            'max_bin_size': self.max_bin_size,
            'monotonic': self.monotonic,
            'special_codes': self.special_codes,
            'cat_cutoff': self.cat_cutoff,
            'random_state': self.random_state,
            'verbose': self.verbose,
            **self.binning_kwargs
        }
        
        # 卡方分箱特殊参数
        if self.method == 'chi2':
            binning_params['min_chi2_threshold'] = self.chi2_threshold
            binning_params['significance_level'] = self.significance_level
        
        # 使用OptimalBinning作为统一接口
        if internal_method in OptimalBinning.VALID_METHODS:
            return OptimalBinning(method=internal_method, **binning_params)
        
        # 特殊处理quantile分箱
        if self.method == 'quantile':
            return QuantileBinning(
                target=self.target,
                n_bins=self.max_n_bins,
                random_state=self.random_state
            )
        
        raise ValueError(f"无法创建分箱器: {self.method}")
    
    def _analyze_numerical_feature(self, feature: str) -> Tuple[pd.DataFrame, Any]:
        """分析数值型特征.

        :param feature: 特征名
        :return: (各阈值指标DataFrame, 分箱器实例)
        """
        feature_values = self.X_[feature]
        valid_values = feature_values[feature_values.notna()]

        if len(valid_values) == 0:
            return pd.DataFrame([self._calculate_metrics(feature, None, 'isna')]), None

        binner = self._get_binning_instance()
        try:
            X_feature = pd.DataFrame({feature: valid_values})
            y_valid = self.y_.loc[valid_values.index]
            binner.fit(X_feature, y_valid)

            if hasattr(binner, 'splits_') and feature in binner.splits_:
                thresholds = binner.splits_[feature]
            elif hasattr(binner, 'bin_edges_'):
                thresholds = sorted(set(binner.bin_edges_))
            else:
                thresholds = self._get_quantile_thresholds(valid_values)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"分箱器拟合失败，回退到quantile分箱: {str(e)}")
            thresholds = self._get_quantile_thresholds(valid_values)
            binner = None

        if isinstance(thresholds, np.ndarray):
            thresholds = thresholds.tolist()
        if len(thresholds) > 2:
            thresholds = thresholds[1:-1]

        results = []
        for i, threshold in enumerate(thresholds):
            if i < len(thresholds) - 1:
                results.append(self._calculate_metrics(feature, threshold, '>='))
            if i > 0:
                results.append(self._calculate_metrics(feature, threshold, '<='))

        if feature_values.isna().any():
            results.append(self._calculate_metrics(feature, None, 'isna'))

        return pd.DataFrame(results), binner
    

    def _analyze_categorical_feature(self, feature: str) -> pd.DataFrame:
        """分析类别型特征.

        :param feature: 特征名
        :return: 各类别指标DataFrame
        """
        results = []

        feature_values = self.X_[feature].copy()
        if self.special_codes:
            for code in self.special_codes:
                feature_values = feature_values.replace(code, f'SPECIAL_{code}')

        value_counts = feature_values.value_counts(dropna=False)
        if self.cat_cutoff is not None:
            if self.cat_cutoff < 1:
                min_count = len(feature_values) * self.cat_cutoff
                valid_categories = value_counts[value_counts >= min_count].index
            else:
                valid_categories = value_counts.head(int(self.cat_cutoff)).index
        else:
            valid_categories = value_counts.index

        for category in valid_categories:
            if pd.isna(category):
                metric_row = self._calculate_metrics(feature, None, 'isna')
            else:
                metric_row = self._calculate_metrics(feature, category, '==')
            if metric_row['selected_samples'] < self.min_samples:
                continue
            results.append(metric_row)

        return pd.DataFrame(results)

    def _get_quantile_thresholds(
self, feature_values: pd.Series) -> List[float]:
        """使用等频分箱获取阈值.
        
        :param feature_values: 特征值
        :return: 阈值列表
        """
        n_unique = feature_values.nunique()
        n_bins = min(self.max_n_bins, n_unique)
        
        if n_bins < 2:
            return [feature_values.median()]
        
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        
        try:
            discretizer.fit(feature_values.values.reshape(-1, 1))
            thresholds = sorted(set(discretizer.bin_edges_[0]))
            return thresholds
        except Exception:
            return [feature_values.median()]
    
    def _calculate_metrics(
        self,
        feature: str,
        threshold: Any,
        operator: str,
    ) -> Dict:
        """计算单个条件的统计指标（支持缺失值规则）."""
        feature_values = self.X_[feature]
        target_values = self.y_

        if operator == '>=':
            mask = feature_values >= threshold
        elif operator == '<=':
            mask = feature_values <= threshold
        elif operator == '==':
            mask = feature_values == threshold
        elif operator == 'isna':
            mask = feature_values.isna()
        elif operator == 'notna':
            mask = feature_values.notna()
        else:
            raise ValueError(f'不支持的操作符: {operator}')

        total = len(target_values)
        selected_count = int(mask.sum())
        selected_bad = int(target_values[mask].sum())

        badrate = selected_bad / selected_count if selected_count > 0 else 0
        lift = calculate_lift(badrate, self.overall_badrate_)
        recall = selected_bad / target_values.sum() if target_values.sum() > 0 else 0
        precision = selected_bad / selected_count if selected_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        try:
            from ...core.metrics.classification import ks as ks_metric
            ks = ks_metric(target_values, mask.astype(int))
        except Exception:
            ks = np.nan

        return {
            'feature': feature,
            'threshold': threshold,
            'operator': operator,
            'total_samples': total,
            'selected_samples': selected_count,
            '命中样本占比': selected_count / total if total > 0 else 0,
            'total_bad': int(target_values.sum()),
            'selected_bad': selected_bad,
            'badrate': badrate,
            'total_badrate': self.overall_badrate_,
            'lift': lift,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'ks': ks
        }
    


    def get_top_rules(
        self,
        top_n: int = 10,
        metric: str = 'lift',
        feature: Optional[str] = None,
        min_lift: Optional[float] = None,
        min_samples: Optional[int] = None,
        ascending: bool = False,
        datasets: Optional[pd.DataFrame] = None,
        target: Optional[str] = None
    ) -> pd.DataFrame:
        """获取Top规则（使用Rule.expr与Rule.report命中结果，支持缺失值规则）."""
        self._check_fitted()

        min_lift = min_lift or self.min_lift
        min_samples = min_samples or self.min_samples

        if feature is not None:
            if feature not in self.results_:
                raise ValueError(f"特征 '{feature}' 未被分析")
            all_rules = self.results_[feature].copy()
        else:
            all_rules = pd.concat(self.results_.values(), ignore_index=True)

        if all_rules.empty:
            return pd.DataFrame()

        all_rules = all_rules[all_rules['lift'] >= min_lift]
        all_rules = all_rules[all_rules['selected_samples'] >= min_samples]
        if all_rules.empty:
            return pd.DataFrame()

        if metric in all_rules.columns:
            all_rules = all_rules.sort_values(by=metric, ascending=ascending)

        target_col = target or self.target
        if datasets is None:
            datasets = self.X_.copy()
            datasets[target_col] = self.y_.values

        rows = []
        for _, row in all_rules.head(top_n).iterrows():
            feature_name = row['feature']
            operator = row['operator']
            threshold = row['threshold']
            feature_expr = f"`{feature_name}`" if not str(feature_name).isidentifier() else str(feature_name)

            if operator == 'isna':
                expr = f"{feature_expr} != {feature_expr}"
            elif operator == 'notna':
                expr = f"{feature_expr} == {feature_expr}"
            else:
                expr = f"{feature_expr} {operator} {repr(threshold)}"

            rule = Rule(expr=expr, name=expr, description=expr, weight=1.0)
            report_df = rule.report(datasets=datasets, target=target_col)
            hit_rows = report_df[report_df['分箱'] == '命中'] if '分箱' in report_df.columns else pd.DataFrame()
            hit = hit_rows.iloc[0].to_dict() if not hit_rows.empty else {}

            rows.append({
                '规则表达式': rule.expr,
                '规则名称': rule.name,
                '命中样本数': hit.get('样本总数'),
                '命中样本占比': hit.get('样本占比'),
                '命中坏样本数': hit.get('坏样本数'),
                '命中坏样本率': hit.get('坏样本率'),
                '命中LIFT值': hit.get('LIFT值'),
                '坏账改善': hit.get('坏账改善'),
                '规则报告': report_df,
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            sort_map = {
                'lift': '命中LIFT值',
                'badrate': '命中坏样本率',
                'sample_ratio': '命中样本占比',
                'selected_samples': '命中样本数',
            }
            sort_col = sort_map.get(metric, metric)
            if sort_col in result_df.columns:
                result_df = result_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        return result_df


    def get_rules(
        self,
        min_lift: Optional[float] = None,
        min_samples: Optional[int] = None,
        target: Optional[str] = None,
        datasets: Optional[pd.DataFrame] = None,
        sort_by: str = 'lift',
        ascending: bool = False,
        top_n: int = 10000
    ) -> List[Rule]:
        """获取挖掘规则（只使用Rule.expr，并基于Rule.report命中结果）."""
        rules_df = self.get_top_rules(
            top_n=top_n,
            metric=sort_by,
            min_lift=min_lift,
            min_samples=min_samples,
            ascending=ascending,
            datasets=datasets,
            target=target,
        )

        if rules_df.empty:
            return []

        target_col = target or self.target
        if datasets is None:
            datasets = self.X_.copy()
            datasets[target_col] = self.y_.values

        rule_objects = []
        for _, row in rules_df.iterrows():
            expr = row['规则表达式']
            rule = Rule(expr=expr, name=row.get('规则名称', expr), description=expr, weight=1.0)
            report_df = rule.report(datasets=datasets, target=target_col)
            hit_rows = report_df[report_df['分箱'] == '命中'] if '分箱' in report_df.columns else pd.DataFrame()
            hit = hit_rows.iloc[0].to_dict() if not hit_rows.empty else {}
            metadata = {
                '规则表达式': rule.expr,
                '规则报告': report_df,
                '命中样本数': hit.get('样本总数'),
                '命中样本占比': hit.get('样本占比'),
                '命中坏样本数': hit.get('坏样本数'),
                '命中坏样本率': hit.get('坏样本率'),
                '命中LIFT值': hit.get('LIFT值'),
                '坏账改善': hit.get('坏账改善'),
            }
            rule.metadata_ = metadata
            rule.metric_score_ = metadata.get('命中LIFT值', 0)
            rule_objects.append(rule)

        return rule_objects

    def get_rule_objects(


        self,
        min_lift: Optional[float] = None,
        min_samples: Optional[int] = None,
        target: Optional[str] = None,
        datasets: Optional[pd.DataFrame] = None,
        sort_by: str = 'lift',
        ascending: bool = False,
        top_n: int = 10000
    ) -> List[Rule]:
        """获取Rule对象列表（与get_rules保持一致）."""
        return self.get_rules(
            min_lift=min_lift,
            min_samples=min_samples,
            target=target,
            datasets=datasets,
            sort_by=sort_by,
            ascending=ascending,
            top_n=top_n
        )

    def analyze_feature(

        self,
        feature: str,
        max_n_bins: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """分析单个特征.
        
        :param feature: 特征名
        :param max_n_bins: 分箱数量（None则使用初始化参数）
        :param kwargs: 其他覆盖参数
        :return: 各阈值指标DataFrame
        """
        self._check_fitted()
        
        if feature not in self.X_.columns:
            raise ValueError(f"特征 '{feature}' 不存在")
        
        # 临时修改参数
        orig_max_n_bins = self.max_n_bins
        if max_n_bins is not None:
            self.max_n_bins = max_n_bins
        
        # 应用其他覆盖参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, f'_{key}_backup', getattr(self, key))
                setattr(self, key, value)
        
        result, _ = self._analyze_feature(feature)
        
        # 恢复参数
        self.max_n_bins = orig_max_n_bins
        for key in kwargs.keys():
            backup_key = f'_{key}_backup'
            if hasattr(self, backup_key):
                setattr(self, key, getattr(self, backup_key))
                delattr(self, backup_key)
        
        return result
    
    def get_feature_summary(self) -> pd.DataFrame:
        """获取特征分析摘要.
        
        :return: 各特征的统计摘要
        """
        self._check_fitted()
        
        summaries = []
        for feature, df in self.results_.items():
            if df.empty:
                continue
            
            best_idx = df['lift'].idxmax()
            best = df.loc[best_idx]
            
            summaries.append({
                'feature': feature,
                'max_lift': df['lift'].max(),
                'max_ks': df['ks'].max() if 'ks' in df.columns else np.nan,
                'best_threshold': best['threshold'],
                'best_operator': best['operator'],
                'best_badrate': best['badrate'],
                'n_candidates': len(df)
            })
        
        return pd.DataFrame(summaries).sort_values('max_lift', ascending=False)
    
    def _check_fitted(self):
        """检查是否已拟合."""
        if not self._is_fitted:
            raise RuntimeError("请先调用fit()方法")
    
    def get_binning_table(self, feature: str) -> Optional[pd.DataFrame]:
        """获取特征的分箱统计表（如果使用了hscredit分箱方法）.
        
        :param feature: 特征名
        :return: 分箱统计表或None
        """
        self._check_fitted()
        
        if feature not in self._binning_instances_:
            return None
        
        binner = self._binning_instances_[feature]
        if hasattr(binner, 'get_bin_table'):
            return binner.get_bin_table(feature)
        elif hasattr(binner, 'bin_tables_') and feature in binner.bin_tables_:
            return binner.bin_tables_[feature]
        
        return None
    
    def plot_feature_analysis(
        self,
        feature: str,
        metric: str = 'lift',
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """绘制特征分析图.
        
        :param feature: 特征名
        :param metric: 可视化指标
        :param figsize: 图大小
        :param title: 自定义标题
        :param save_path: 保存路径
        :return: matplotlib.pyplot对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装matplotlib: pip install matplotlib")
        
        from ...viz.utils import DEFAULT_COLORS, setup_axis_style
        
        self._check_fitted()
        
        df = self.analyze_feature(feature)
        if df.empty:
            raise ValueError(f"特征 '{feature}' 无有效分析结果")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 分别绘制>=和<=两种情况
        colors = DEFAULT_COLORS[:2]
        for i, operator in enumerate(['>=', '<=']):
            subset = df[df['operator'] == operator]
            if not subset.empty:
                ax.plot(
                    subset['threshold'],
                    subset[metric],
                    label=f'{operator}',
                    marker='o',
                    markersize=5,
                    color=colors[i % len(colors)]
                )
        
        # 设置样式
        setup_axis_style(ax)
        
        plot_title = title or f'特征 {feature} - {metric.upper()} 分析'
        ax.set_title(plot_title)
        ax.set_xlabel(f'{feature} 阈值')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=240, bbox_inches='tight')
        
        return fig
