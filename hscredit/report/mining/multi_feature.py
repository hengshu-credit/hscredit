"""多特征交叉规则挖掘模块.

支持双特征交叉分析，生成交叉矩阵和交叉规则。
支持hscredit所有分箱方法。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
import warnings

from .base import BaseRuleMiner, calculate_lift
from ...core.rules.rule import Rule
from ...core.binning import OptimalBinning


class MultiFeatureRuleMiner(BaseRuleMiner):
    """多特征交叉规则挖掘器.
    
    生成双特征交叉分析结果，支持hscredit所有分箱方法。
    
    代码风格参考hscredit的binning模块，fit方法兼容scorecardpipeline风格。
    
    :param target: 目标变量列名，默认为'target'
    :param exclude_cols: 需要排除的列名列表
    :param method: 分箱方法，支持hscredit中所有分箱方法:
        - 'quantile': 等频分箱 (默认)
        - 'chi2': 卡方分箱
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
    :param max_n_bins: 最大分箱数，默认5。超过此值的数值型特征将分箱
    :param min_n_bins: 最小分箱数，默认2
    :param min_bin_size: 每箱最小样本数或占比，默认0.01
    :param max_bin_size: 每箱最大样本数或占比，默认None
    :param monotonic: 坏样本率单调性约束，默认False
    :param cat_cutoff: 类别型变量处理阈值，默认None
    :param min_samples: 最小样本数，默认10
    :param min_lift: 最小lift阈值，默认1.1
    :param special_codes: 特殊值列表，默认None
    :param random_state: 随机种子，默认None
    :param verbose: 是否输出详细信息，默认False
    :param binning_kwargs: 分箱方法的其他参数，通过**kwargs传入
    
    **参考样例**

    >>> miner = MultiFeatureRuleMiner(target='ISBAD', method='quantile', max_n_bins=5)  # 等频分箱：双特征交叉规则挖掘
    >>> miner.fit(df)
    >>> cross_matrix = miner.generate_cross_matrix('age', 'income')  # 生成年龄×收入的交叉分箱矩阵
    >>> rules = miner.get_cross_rules('age', 'income', top_n=10)  # 获取TOP10交叉规则，按LIFT排序
    >>> miner = MultiFeatureRuleMiner(target='ISBAD', method='chi2', max_n_bins=4)  # 卡方分箱：自动合并坏率相近的交叉箱
    >>> miner.fit(df)
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
        method: str = 'quantile',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        monotonic: Union[bool, str] = False,
        cat_cutoff: Optional[Union[float, int]] = None,
        min_samples: int = 10,
        min_lift: float = 1.1,
        special_codes: Optional[List] = None,
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
        self.cat_cutoff = cat_cutoff
        self.min_samples = min_samples
        self.min_lift = min_lift
        self.special_codes = special_codes
        self.random_state = random_state
        self.verbose = verbose
        self.binning_kwargs = binning_kwargs
        
        self.numerical_features_ = []
        self.categorical_features_ = []
        self.overall_badrate_ = 0.0
        self.cross_results_ = {}
        self._binner_instances_ = {}
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'MultiFeatureRuleMiner':
        """拟合挖掘器.
        
        :param X: 训练数据
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
            raise ValueError("多特征规则挖掘需要目标变量y")
        
        self.X_ = X
        self.y_ = y
        
        # 分类特征
        self.numerical_features_ = self._get_numeric_features(X)
        self.categorical_features_ = self._get_categorical_features(X)
        
        # 计算整体坏账率
        self.overall_badrate_ = y.mean()
        
        self._is_fitted = True
        return self
    
    def _get_binning_instance(self, **override_params) -> OptimalBinning:
        """获取分箱器实例.
        
        :param override_params: 覆盖参数
        :return: hscredit分箱器实例
        """
        internal_method = self.METHOD_MAPPING[self.method]
        
        # 构建分箱器参数
        binning_params = {
            'target': self.target,
            'method': internal_method,
            'max_n_bins': self.max_n_bins,
            'min_n_bins': self.min_n_bins,
            'min_bin_size': self.min_bin_size,
            'max_bin_size': self.max_bin_size,
            'monotonic': self.monotonic,
            'special_codes': self.special_codes,
            'cat_cutoff': self.cat_cutoff,
            'random_state': self.random_state,
            'verbose': self.verbose,
            **self.binning_kwargs,
            **override_params
        }
        
        return OptimalBinning(**binning_params)
    
    def _prepare_feature(
        self,
        feature: str,
        custom_splits: Optional[List[float]] = None
    ) -> pd.Series:
        """准备特征，对取值多的特征进行分箱.
        
        :param feature: 特征名
        :param custom_splits: 自定义分箱切分点
        :return: 处理后的特征值
        """
        unique_count = self.X_[feature].nunique(dropna=False)
        
        # 如果唯一值较少，直接返回
        if unique_count <= self.max_n_bins:
            return self.X_[feature].fillna('缺失')
        
        # 数值型特征分箱
        if feature in self.numerical_features_:
            if custom_splits is not None:
                return pd.cut(self.X_[feature], bins=custom_splits, right=True).astype(str).replace('nan', '缺失')
            else:
                return self._bin_feature(feature).fillna('缺失')
        
        # 类别型特征：根据cat_cutoff处理
        if self.cat_cutoff is not None:
            if self.cat_cutoff < 1:
                # 保留占比超过阈值的类别
                min_count = len(self.X_) * self.cat_cutoff
                top_values = self.X_[feature].value_counts()
                top_values = top_values[top_values >= min_count].index
            else:
                # 保留频率最高的N个类别
                top_values = self.X_[feature].value_counts().head(int(self.cat_cutoff)).index
        else:
            # 默认保留最常见的max_n_bins个值
            top_values = self.X_[feature].value_counts().head(self.max_n_bins).index
        
        return self.X_[feature].apply(lambda x: x if x in top_values else '其他').fillna('缺失')
    
    def _bin_feature(self, feature: str) -> pd.Series:
        """使用hscredit分箱方法对特征分箱.
        
        :param feature: 特征名
        :return: 分箱后的区间
        """
        # 创建分箱器
        binner = self._get_binning_instance()
        
        try:
            # 准备数据
            X_feature = self.X_[[feature]].copy()
            valid_mask = X_feature[feature].notna()
            
            if valid_mask.sum() < self.min_samples * 2:
                return self.X_[feature]
            
            # 拟合分箱器
            binner.fit(X_feature[valid_mask], self.y_[valid_mask])
            
            # 存储分箱器实例
            self._binner_instances_[feature] = binner
            
            # 转换数据
            X_binned = binner.transform(X_feature)
            
            # 获取分箱标签
            if hasattr(binner, 'get_bin_table'):
                bin_table = binner.get_bin_table(feature)
                if bin_table is not None and '分箱标签' in bin_table.columns:
                    # 创建映射
                    bin_labels = bin_table['分箱标签'].values
                    result = pd.Series(index=self.X_.index, dtype=object)
                    for i, label in enumerate(bin_labels):
                        result[X_binned[feature] == i] = label
                    return result
            
            # 回退：使用数值分箱索引
            return X_binned[feature].astype(str).replace('nan', '缺失')
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"分箱失败，使用原始值: {str(e)}")
            return self.X_[feature]
    
    def generate_cross_matrix(
        self,
        feature1: str,
        feature2: str,
        custom_splits1: Optional[List[float]] = None,
        custom_splits2: Optional[List[float]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """生成双特征交叉矩阵.
        
        :param feature1: 第一个特征
        :param feature2: 第二个特征
        :param custom_splits1: 特征1的自定义分箱切分点
        :param custom_splits2: 特征2的自定义分箱切分点
        :param kwargs: 其他参数
        :return: 交叉矩阵DataFrame
        """
        self._check_fitted()
        
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 准备特征
        f1_prepared = self._prepare_feature(feature1, custom_splits1)
        f2_prepared = self._prepare_feature(feature2, custom_splits2)
        
        # 创建交叉数据
        cross_data = pd.DataFrame({
            'feature1': f1_prepared,
            'feature2': f2_prepared,
            'target': self.y_
        })
        
        # 计算各类统计量
        count_matrix = pd.crosstab(
            cross_data['feature1'],
            cross_data['feature2'],
            rownames=[feature1],
            colnames=[feature2]
        )
        
        bad_count_matrix = pd.crosstab(
            cross_data['feature1'],
            cross_data['feature2'],
            values=cross_data['target'],
            aggfunc='sum'
        )
        
        # 坏账率矩阵
        badrate_matrix = bad_count_matrix / count_matrix
        badrate_matrix = badrate_matrix.fillna(0)
        
        # 样本占比矩阵
        total_samples = count_matrix.sum().sum()
        sample_ratio_matrix = count_matrix / total_samples if total_samples > 0 else count_matrix * 0
        
        # lift矩阵
        lift_matrix = badrate_matrix / self.overall_badrate_ if self.overall_badrate_ > 0 else badrate_matrix * 0
        lift_matrix = lift_matrix.fillna(0)
        
        # 计算KS值（如果可能）
        try:
            from ...metrics.classification import ks as ks_statistic
            ks_matrix = pd.DataFrame(index=count_matrix.index, columns=count_matrix.columns)
            for f1_val in count_matrix.index:
                for f2_val in count_matrix.columns:
                    mask = (cross_data['feature1'] == f1_val) & (cross_data['feature2'] == f2_val)
                    if mask.sum() > 0:
                        ks_matrix.loc[f1_val, f2_val] = ks_statistic(
                            cross_data['target'], mask.astype(int)
                        )
                    else:
                        ks_matrix.loc[f1_val, f2_val] = 0
        except Exception:
            ks_matrix = None
        
        # 构建MultiIndex DataFrame
        metrics = ['count', 'bad_count', 'badrate', 'sample_ratio', 'lift']
        if ks_matrix is not None:
            metrics.append('ks')
        
        cols = pd.MultiIndex.from_product(
            [count_matrix.columns, metrics],
            names=[feature2, 'metric']
        )
        
        result = pd.DataFrame(index=count_matrix.index, columns=cols)
        
        for f2_val in count_matrix.columns:
            result[(f2_val, 'count')] = count_matrix[f2_val]
            result[(f2_val, 'bad_count')] = bad_count_matrix[f2_val]
            result[(f2_val, 'badrate')] = badrate_matrix[f2_val]
            result[(f2_val, 'sample_ratio')] = sample_ratio_matrix[f2_val]
            result[(f2_val, 'lift')] = lift_matrix[f2_val]
            if ks_matrix is not None:
                result[(f2_val, 'ks')] = ks_matrix[f2_val]
        
        # 保存结果供后续使用
        key = (feature1, feature2)
        self.cross_results_[key] = {
            'matrix': result,
            'count': count_matrix,
            'badrate': badrate_matrix,
            'lift': lift_matrix,
            'sample_ratio': sample_ratio_matrix,
            'ks': ks_matrix
        }
        
        return result
    

    def get_cross_rules(
        self,
        feature1: str,
        feature2: str,
        top_n: int = 10,
        metric: str = 'lift',
        min_samples: Optional[int] = None,
        min_lift: Optional[float] = None,
        datasets: Optional[pd.DataFrame] = None,
        target: Optional[str] = None
    ) -> pd.DataFrame:
        """获取双特征交叉的top规则（使用rule_expr+Rule.report命中结果）."""
        min_samples = min_samples or self.min_samples
        min_lift = min_lift or self.min_lift

        # 生成交叉矩阵并转长表
        cross_matrix = self.generate_cross_matrix(feature1, feature2)
        long_df = self._matrix_to_long(cross_matrix, feature1, feature2)
        long_df = long_df[long_df['count'] >= min_samples]
        long_df = long_df[long_df['lift'] >= min_lift]

        if metric in long_df.columns:
            long_df = long_df.sort_values(by=metric, ascending=False)

        target_col = target or self.target
        if datasets is None:
            datasets = self.X_.copy()
            datasets[target_col] = self.y_.values

        rows = []
        for _, row in long_df.head(top_n).iterrows():
            v1, v2 = row['feature1_value'], row['feature2_value']
            f1_expr = f"`{feature1}`" if not str(feature1).isidentifier() else str(feature1)
            f2_expr = f"`{feature2}`" if not str(feature2).isidentifier() else str(feature2)
            if pd.isna(v1) or v1 == '缺失':
                c1 = f"({f1_expr} != {f1_expr})"
            else:
                c1 = f"({f1_expr} == {repr(v1)})"
            if pd.isna(v2) or v2 == '缺失':
                c2 = f"({f2_expr} != {f2_expr})"
            else:
                c2 = f"({f2_expr} == {repr(v2)})"
            expr = f"{c1} & {c2}"
            rule = Rule(expr=expr, name=expr, description=expr, weight=1.0)

            report_df = rule.report(datasets=datasets, target=target_col)
            hit_rows = report_df[report_df['分箱'] == '命中'] if '分箱' in report_df.columns else pd.DataFrame()
            hit = hit_rows.iloc[0].to_dict() if not hit_rows.empty else {}

            rows.append({
                '规则表达式': rule.expr,
                '规则名称': rule.name,
                'feature1_value': v1,
                'feature2_value': v2,
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
                'count': '命中样本数',
            }
            sort_col = sort_map.get(metric, metric)
            if sort_col in result_df.columns:
                result_df = result_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        return result_df

    def _matrix_to_long(

        self,
        matrix: pd.DataFrame,
        feature1: str,
        feature2: str
    ) -> pd.DataFrame:
        """将交叉矩阵转换为长格式.
        
        :param matrix: 交叉矩阵
        :param feature1: 特征1名称
        :param feature2: 特征2名称
        :return: 长格式DataFrame
        """
        rows = []
        
        metrics = ['count', 'bad_count', 'badrate', 'sample_ratio', 'lift']
        if (matrix.columns.get_level_values(0)[0], 'ks') in matrix.columns:
            metrics.append('ks')
        
        for f1_val in matrix.index:
            for f2_val in matrix.columns.get_level_values(0).unique():
                row = {'feature1_value': f1_val, 'feature2_value': f2_val}
                
                for metric in metrics:
                    try:
                        row[metric] = matrix.loc[f1_val, (f2_val, metric)]
                    except KeyError:
                        row[metric] = 0
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_all_cross_rules(
        self,
        top_n: int = 10,
        metric: str = 'lift',
        max_feature_pairs: int = 50,
        min_samples: Optional[int] = None,
        min_lift: Optional[float] = None
    ) -> pd.DataFrame:
        """获取所有特征对的交叉规则.
        
        :param top_n: 每对特征的返回规则数
        :param metric: 排序指标
        :param max_feature_pairs: 最大特征对数量
        :param min_samples: 最小样本数
        :param min_lift: 最小lift
        :return: 所有交叉规则DataFrame
        """
        self._check_fitted()
        
        all_features = self.numerical_features_ + self.categorical_features_
        
        # 选择信息价值最高的特征对
        if len(all_features) > 10:
            # 简单启发式：选择方差最大的特征
            feature_variance = self.X_[all_features].var().sort_values(ascending=False)
            selected_features = feature_variance.head(10).index.tolist()
        else:
            selected_features = all_features
        
        feature_pairs = list(combinations(selected_features, 2))[:max_feature_pairs]
        
        all_rules = []
        
        for f1, f2 in feature_pairs:
            try:
                rules = self.get_cross_rules(
                    f1, f2,
                    top_n=top_n,
                    metric=metric,
                    min_samples=min_samples,
                    min_lift=min_lift
                )
                if not rules.empty:
                    rules['特征组合'] = f"{f1} × {f2}"
                    all_rules.append(rules)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"生成{f1}和{f2}的交叉规则时出错: {str(e)}")
        
        if not all_rules:
            return pd.DataFrame()
        
        combined = pd.concat(all_rules, ignore_index=True)
        sort_map = {
            'lift': '命中LIFT值',
            'badrate': '命中坏样本率',
            'sample_ratio': '命中样本占比',
            'count': '命中样本数',
        }
        sort_col = sort_map.get(metric, metric)
        if sort_col in combined.columns:
            combined = combined.sort_values(by=sort_col, ascending=False)
        return combined



    def get_rules(
        self,
        top_n: int = 10,
        metric: str = 'lift',
        target: Optional[str] = None,
        datasets: Optional[pd.DataFrame] = None,
        min_samples: Optional[int] = None,
        min_lift: Optional[float] = None
    ) -> List[Rule]:
        """获取挖掘的交叉规则（只使用Rule.expr，并基于Rule.report命中结果）."""
        rules_df = self.get_all_cross_rules(
            top_n=top_n,
            metric=metric,
            min_samples=min_samples,
            min_lift=min_lift
        )

        if rules_df.empty:
            return []

        target_col = target or self.target
        if datasets is None:
            datasets = self.X_.copy()
            datasets[target_col] = self.y_.values

        rule_objects = []
        for _, row in rules_df.iterrows():
            expr = row['规则表达式'] if '规则表达式' in row else None
            if not expr:
                feature_pair = row.get('特征组合', '')
                if ' × ' not in feature_pair:
                    continue
                f1, f2 = feature_pair.split(' × ')
                v1, v2 = row['feature1_value'], row['feature2_value']
                f1_expr = f"`{f1}`" if not str(f1).isidentifier() else str(f1)
                f2_expr = f"`{f2}`" if not str(f2).isidentifier() else str(f2)

                if pd.isna(v1) or v1 == '缺失':
                    c1 = f"({f1_expr} != {f1_expr})"
                else:
                    c1 = f"({f1_expr} == {repr(v1)})"

                if pd.isna(v2) or v2 == '缺失':
                    c2 = f"({f2_expr} != {f2_expr})"
                else:
                    c2 = f"({f2_expr} == {repr(v2)})"

                expr = f"{c1} & {c2}"

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
        top_n: int = 10,
        metric: str = 'lift',
        target: Optional[str] = None,
        datasets: Optional[pd.DataFrame] = None,
        min_samples: Optional[int] = None,
        min_lift: Optional[float] = None
    ) -> List[Rule]:
        """获取Rule对象列表（与get_rules保持一致）."""
        return self.get_rules(
            top_n=top_n,
            metric=metric,
            target=target,
            datasets=datasets,
            min_samples=min_samples,
            min_lift=min_lift
        )

    def get_binning_table(self, feature: str) -> Optional[pd.DataFrame]:
        """获取特征的分箱统计表（如果使用了hscredit分箱方法）.
        
        :param feature: 特征名
        :return: 分箱统计表或None
        """
        self._check_fitted()
        
        if feature not in self._binner_instances_:
            return None
        
        binner = self._binner_instances_[feature]
        if hasattr(binner, 'get_bin_table'):
            return binner.get_bin_table(feature)
        elif hasattr(binner, 'bin_tables_') and feature in binner.bin_tables_:
            return binner.bin_tables_[feature]
        
        return None
    
    def plot_cross_heatmap(
        self,
        feature1: str,
        feature2: str,
        metric: str = 'lift',
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        cmap: Optional[str] = None
    ):
        """绘制交叉热力图.
        
        :param feature1: 特征1
        :param feature2: 特征2
        :param metric: 指标
        :param figsize: 图大小
        :param title: 自定义标题
        :param save_path: 保存路径
        :param cmap: 颜色映射，默认使用RdYlGn
        :return: matplotlib Figure对象
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("需要安装matplotlib和seaborn")
        
        from ...core.viz.utils import DEFAULT_COLORS, setup_axis_style
        
        cross_matrix = self.generate_cross_matrix(feature1, feature2)
        
        # 提取指标矩阵
        metric_matrix = cross_matrix.xs(metric, level='metric', axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 默认颜色映射
        if cmap is None:
            cmap = 'RdYlGn' if metric in ['lift', 'badrate'] else 'Blues'
        
        # 中心值（lift以1为中心）
        center = 1.0 if metric == 'lift' else None
        
        sns.heatmap(
            metric_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=center,
            ax=ax,
            linewidths=0.5
        )
        
        # 设置样式
        setup_axis_style(ax)
        
        plot_title = title or f'{feature1} x {feature2} - {metric.upper()} 热力图'
        ax.set_title(plot_title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=240, bbox_inches='tight')
        
        return fig

    def plot_2d_binning(
        self,
        feature1: str,
        feature2: str,
        metric: str = 'lift',
        figsize: Tuple[int, int] = (14, 10),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        cmap: Optional[str] = None
    ):
        """绘制二维分箱图（主图+上下文分箱轴）.

        - 中心主图：二维网格指标热力图
        - 顶部：feature2 分箱轴（与主图共享x轴）
        - 右侧：feature1 分箱轴（与主图共享y轴）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib import gridspec
        except ImportError:
            raise ImportError("需要安装matplotlib和seaborn")

        from ...viz.utils import setup_axis_style

        cross_matrix = self.generate_cross_matrix(feature1, feature2)
        metric_matrix = cross_matrix.xs(metric, level='metric', axis=1)

        # 数值化，避免object类型导致热力图显示异常
        metric_matrix = metric_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

        if cmap is None:
            cmap = 'RdYlGn' if metric in ['lift', 'badrate'] else 'Blues'

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            height_ratios=[0.9, 8.0],
            width_ratios=[8.0, 1.4],
            hspace=0.02,
            wspace=0.02
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        center = 1.0 if metric == 'lift' else None
        hm = sns.heatmap(
            metric_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=center,
            ax=ax_main,
            linewidths=0.5,
            cbar=True,
            cbar_kws={'shrink': 0.9, 'pad': 0.02}
        )

        # 明确标注主图数值含义，避免看不出来是lift
        cbar = hm.collections[0].colorbar
        cbar.set_label(f'{metric.upper()} value', rotation=90)
        ax_main.text(
            0.005,
            1.02,
            f'Cell number = {metric.upper()}',
            transform=ax_main.transAxes,
            fontsize=10,
            va='bottom',
            ha='left'
        )

        # 顶部分箱轴：复用主图x轴（按列格对齐）
        n_cols = metric_matrix.shape[1]
        for i, label in enumerate(metric_matrix.columns):
            ax_top.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor='#E8EEF7', edgecolor='white', linewidth=1.0))
            ax_top.text(i + 0.5, 0.5, str(label), ha='center', va='center', fontsize=8, rotation=0)
        ax_top.set_xlim(0, n_cols)
        ax_top.set_ylim(0, 1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_ylabel(f'{feature2}\n分箱', fontsize=9)

        # 右侧分箱轴：复用主图y轴（按行格对齐）
        n_rows = metric_matrix.shape[0]
        for i, label in enumerate(metric_matrix.index):
            ax_right.add_patch(plt.Rectangle((0, i), 1, 1, facecolor='#E8EEF7', edgecolor='white', linewidth=1.0))
            ax_right.text(0.5, i + 0.5, str(label), ha='center', va='center', fontsize=8, rotation=0)
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(0, n_rows)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ax_right.set_xlabel(f'{feature1} 分箱', fontsize=9)

        # 统一样式
        setup_axis_style(ax_main)
        for a in [ax_top, ax_right]:
            for spine in a.spines.values():
                spine.set_visible(False)

        plot_title = title or f'{feature1} x {feature2} - {metric.upper()} 二维分箱图'
        ax_main.set_title(plot_title)
        ax_main.set_xlabel(feature2)
        ax_main.set_ylabel(feature1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=240, bbox_inches='tight')

        return fig
    
    def _check_fitted(self):
        """检查是否已拟合."""
        if not self._is_fitted:
            raise RuntimeError("请先调用fit()方法")
