"""Swap规则置换分析模块.

提供金融信贷业务中规则置换时的风险指标分析功能，
支持基于评分的风险预估和swap四象限分析。

核心概念:
- in-in: 原策略通过，新策略也通过（保留样本）
- in-out: 原策略通过，新策略拒绝（置出样本）
- out-in: 原策略拒绝，新策略通过（置入样本）- 核心关注
- out-out: 原策略拒绝，新策略也拒绝（仍拒绝样本）

主要功能:
- 基于参考数据集计算评分区间逾期率
- 对无标签swap数据进行风险预估
- out-in样本支持风险上浮因子
- 完整的swap四象限风险分析报告
- 通过率变化分析
- 风险拒绝率分析
- 多标签支持
- 订单口径和金额口径分离

Author: hscredit
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class SwapType(Enum):
    """Swap四象限类型."""
    IN_IN = "in-in"       # 原策略通过，新策略通过
    IN_OUT = "in-out"     # 原策略通过，新策略拒绝
    OUT_IN = "out-in"     # 原策略拒绝，新策略通过
    OUT_OUT = "out-out"   # 原策略拒绝，新策略拒绝


@dataclass
class SwapRiskConfig:
    """Swap风险分析配置.
    
    :param score_col: 评分字段名，用于风险预估
    :param swap_type_col: swap类型字段名
    :param amount_col: 金额字段名（可选）
    :param out_in_uplift: out-in样本风险上浮因子，默认2.0
    :param bin_method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param custom_bins: 自定义分箱边界（可选）
    :param original_pass_rate: 原策略通过率（可选，用于无out-out数据场景）
    :param targets: 目标变量列表（可选）
    :param target_aliases: 目标变量别名（可选），用于报告展示
    """
    score_col: str = "score"
    swap_type_col: str = "swap_type"
    amount_col: Optional[str] = None
    out_in_uplift: float = 2.0
    bin_method: str = "quantile"
    max_n_bins: int = 10
    custom_bins: Optional[List[float]] = None
    original_pass_rate: Optional[float] = None
    targets: Optional[List[str]] = None
    target_aliases: Optional[Dict[str, str]] = None


@dataclass
class PassRateAnalysis:
    """通过率分析结果.
    
    :param original_pass_rate: 原策略通过率
    :param new_pass_rate: 新策略通过率
    :param absolute_change: 通过率绝对变化（百分点）
    :param relative_change_pct: 通过率相对变化（百分比）
    """
    original_pass_rate: float
    new_pass_rate: float
    absolute_change: float
    relative_change_pct: float
    
    def to_dict(self) -> Dict:
        """转换为字典."""
        return {
            '原策略通过率': f"{self.original_pass_rate:.2%}",
            '新策略通过率': f"{self.new_pass_rate:.2%}",
            '通过率变化(绝对)': f"{self.absolute_change:.2%}",
            '通过率变化(相对)': f"{self.relative_change_pct:.2f}%",
        }


@dataclass
class RiskMetrics:
    """风险指标.
    
    :param bad_rate: 坏账率
    :param lift: LIFT值
    :param bad_improvement: 风险改善
    :param risk_reject_ratio: 风险拒绝比
    """
    bad_rate: float
    lift: float
    bad_improvement: float
    risk_reject_ratio: float


@dataclass
class RiskRejectionMetrics:
    """风险拒绝率指标.
    
    :param rejection_rate: 拒绝率
    :param risk_rejection_rate: 风险拒绝率
    :param pass_rate_drop_1pct_improvement: 通过率下降1%的坏账改善
    :param bad_rate_before: 原策略坏账率
    :param bad_rate_after: 新策略坏账率
    :param bad_rate_improvement: 坏账率改善（绝对）
    :param bad_rate_relative_improvement: 坏账率改善（相对，百分比）
    """
    rejection_rate: float
    risk_rejection_rate: float
    pass_rate_drop_1pct_improvement: float
    bad_rate_before: float
    bad_rate_after: float
    bad_rate_improvement: float
    bad_rate_relative_improvement: float
    
    def to_dict(self) -> Dict:
        """转换为字典."""
        return {
            '拒绝率': f"{self.rejection_rate:.2%}",
            '风险拒绝率': f"{self.risk_rejection_rate:.4f}",
            '通过率下降1%的坏账改善': f"{self.pass_rate_drop_1pct_improvement:.4f}",
            '原坏账率': f"{self.bad_rate_before:.4f}",
            '新坏账率': f"{self.bad_rate_after:.4f}",
            '坏账率改善(绝对)': f"{self.bad_rate_improvement:.4f}",
            '坏账率改善(相对)': f"{self.bad_rate_relative_improvement:.2f}%",
        }


class ReferenceDataProvider:
    """参考数据提供者.
    
    从有标签的参考数据计算评分区间逾期率，用于swap分析中的风险预估。
    
    :param score_col: 评分字段名
    :param target_cols: 目标变量字段名或列表
    :param amount_col: 金额字段名（可选）
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param custom_bins: 自定义分箱边界（可选）
    """
    
    def __init__(
        self,
        score_col: str = "score",
        target_cols: Union[str, List[str]] = "target",
        amount_col: Optional[str] = None,
        method: str = "quantile",
        max_n_bins: int = 10,
        custom_bins: Optional[List[float]] = None
    ):
        self.score_col = score_col
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.amount_col = amount_col
        self.method = method
        self.max_n_bins = max_n_bins
        self.custom_bins = custom_bins
        self.bin_stats: Dict[str, List[Dict]] = {t: [] for t in self.target_cols}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "ReferenceDataProvider":
        """从参考数据计算评分区间逾期率.
        
        :param df: 参考数据集，包含score_col和target_cols
        :return: self
        """
        df = df.copy()
        
        if self.custom_bins is not None:
            bins = self.custom_bins
        else:
            bins = self._generate_bins(df[self.score_col])
        
        self.bins = bins
        df['bin'] = pd.cut(df[self.score_col], bins=bins, include_lowest=True)
        
        for target_col in self.target_cols:
            stats = []
            for interval in df['bin'].cat.categories:
                mask = df['bin'] == interval
                subset = df[mask]
                
                if len(subset) == 0:
                    continue
                
                sample_count = len(subset)
                bad_count = subset[target_col].sum()
                bad_rate = bad_count / sample_count if sample_count > 0 else 0.0
                
                amount_sum = None
                if self.amount_col:
                    amount_sum = subset[self.amount_col].sum()
                
                stats.append({
                    'bin_label': f"[{interval.left:.2f}, {interval.right:.2f})",
                    'score_min': interval.left,
                    'score_max': interval.right,
                    'sample_count': sample_count,
                    'bad_count': bad_count,
                    'bad_rate': bad_rate,
                    'amount_sum': amount_sum
                })
            
            self.bin_stats[target_col] = stats
        
        self._is_fitted = True
        return self
    
    def _generate_bins(self, scores: pd.Series) -> List[float]:
        """生成分箱边界.
        
        :param scores: 评分序列
        :return: 分箱边界列表
        """
        if self.method == "quantile":
            quantiles = np.linspace(0, 1, self.max_n_bins + 1)
            bins = scores.quantile(quantiles).values
        elif self.method == "uniform":
            bins = np.linspace(scores.min(), scores.max(), self.max_n_bins + 1)
        elif self.method == "custom" and self.custom_bins:
            bins = self.custom_bins
        else:
            quantiles = np.linspace(0, 1, self.max_n_bins + 1)
            bins = scores.quantile(quantiles).values
        
        bins = np.unique(bins)
        return bins.tolist()
    
    def predict_bad_rate(self, scores: Union[float, pd.Series], target_col: str = None) -> Union[float, pd.Series]:
        """根据评分预估坏样本率.
        
        :param scores: 单个评分或评分序列
        :param target_col: 目标变量名（可选，默认使用第一个）
        :return: 预估坏样本率
        """
        if not self._is_fitted:
            raise ValueError("需要先调用fit方法")
        
        target = target_col or self.target_cols[0]
        stats = self.bin_stats[target]
        
        is_scalar = np.isscalar(scores)
        if is_scalar:
            scores = pd.Series([scores])
        else:
            scores = pd.Series(scores)
        
        bad_rates = []
        for score in scores:
            found = False
            for stat in stats:
                if stat['score_min'] <= score < stat['score_max']:
                    bad_rates.append(stat['bad_rate'])
                    found = True
                    break
            if not found:
                if score >= stats[-1]['score_max']:
                    bad_rates.append(stats[-1]['bad_rate'])
                elif score < stats[0]['score_min']:
                    bad_rates.append(stats[0]['bad_rate'])
                else:
                    bad_rates.append(0.0)
        
        if is_scalar:
            return bad_rates[0]
        return pd.Series(bad_rates)


class SwapAnalyzer:
    """Swap规则置换分析器.
    
    对swap数据进行风险预估和指标计算。
    
    :param config: Swap风险分析配置
    :param ref_provider: 参考数据提供者（可选）
    """
    
    def __init__(
        self,
        config: Optional[SwapRiskConfig] = None,
        ref_provider: Optional[ReferenceDataProvider] = None
    ):
        self.config = config or SwapRiskConfig()
        self.ref_provider = ref_provider
        self.result: Optional[SwapAnalysisResult] = None
    
    def analyze(
        self,
        swap_df: pd.DataFrame,
        ref_provider: Optional[ReferenceDataProvider] = None
    ) -> "SwapAnalysisResult":
        """执行swap分析.
        
        :param swap_df: swap数据集
        :param ref_provider: 参考数据提供者（可选）
        :return: SwapAnalysisResult分析结果
        """
        if ref_provider is not None:
            self.ref_provider = ref_provider
        
        if self.ref_provider is None:
            raise ValueError("需要提供ReferenceDataProvider")
        
        if not self.ref_provider._is_fitted:
            raise ValueError("ReferenceDataProvider需要先调用fit方法")
        
        df = swap_df.copy()
        cfg = self.config
        targets = cfg.targets or self.ref_provider.target_cols
        
        # 为每个样本、每个标签预估坏样本率
        for target in targets:
            df[f'predicted_bad_rate_{target}'] = self.ref_provider.predict_bad_rate(df[cfg.score_col], target)
            df[f'adjusted_bad_rate_{target}'] = df[f'predicted_bad_rate_{target}']
            out_in_mask = df[cfg.swap_type_col] == SwapType.OUT_IN.value
            df.loc[out_in_mask, f'adjusted_bad_rate_{target}'] *= cfg.out_in_uplift
            df[f'adjusted_bad_rate_{target}'] = df[f'adjusted_bad_rate_{target}'].clip(upper=1.0)
        
        self.result = self._calculate_stats(df, targets)
        return self.result
    
    def _calculate_stats(self, df: pd.DataFrame, targets: List[str]) -> "SwapAnalysisResult":
        """计算各swap类型的统计信息.
        
        :param df: 数据集
        :param targets: 目标变量列表
        :return: SwapAnalysisResult
        """
        cfg = self.config
        total_samples = len(df)
        
        # 计算订单口径统计
        count_stats = self._calculate_stats_by_metric(df, targets, metric='count')
        
        # 计算金额口径统计（如果有金额字段）
        amount_stats = None
        if cfg.amount_col:
            amount_stats = self._calculate_stats_by_metric(df, targets, metric='amount')
        
        # 计算组合统计
        count_combined = self._calculate_combined_stats(count_stats, total_samples, metric='count')
        amount_combined = None
        if amount_stats:
            total_amount = df[cfg.amount_col].sum()
            amount_combined = self._calculate_combined_stats(amount_stats, total_samples, metric='amount', total_amount=total_amount)
        
        # 计算通过率分析
        pass_rate_analysis = self._calculate_pass_rate_analysis(count_stats, total_samples)
        
        # 计算风险拒绝率指标（每个标签）
        risk_rejection_metrics_dict = {}
        for target in targets:
            risk_rejection_metrics_dict[target] = self._calculate_risk_rejection_metrics(count_stats, count_combined, target)
        
        # 使用第一个标签作为默认风险指标
        risk_rejection_metrics = risk_rejection_metrics_dict[targets[0]]
        
        return SwapAnalysisResult(
            count_stats=count_stats,
            amount_stats=amount_stats,
            count_combined=count_combined,
            amount_combined=amount_combined,
            config=cfg,
            targets=targets,
            pass_rate_analysis=pass_rate_analysis,
            risk_rejection_metrics=risk_rejection_metrics,
            risk_rejection_metrics_dict=risk_rejection_metrics_dict,
            total_samples=total_samples
        )
    
    def _calculate_stats_by_metric(self, df: pd.DataFrame, targets: List[str], metric: str = 'count') -> Dict[SwapType, Dict]:
        """按口径计算各swap类型的统计.
        
        :param df: 数据集
        :param targets: 目标变量列表
        :param metric: 统计口径，'count'或'amount'
        :return: 各swap类型统计字典
        """
        cfg = self.config
        stats = {}
        
        for swap_type in SwapType:
            mask = df[cfg.swap_type_col] == swap_type.value
            subset = df[mask]
            
            if len(subset) == 0:
                stats[swap_type] = self._empty_stats(swap_type, targets)
                continue
            
            stat = {'swap_type': swap_type.value}
            
            if metric == 'count':
                stat['total_count'] = len(subset)
                stat['sample_ratio'] = len(subset) / len(df)
            else:
                total_amount = df[cfg.amount_col].sum()
                subset_amount = subset[cfg.amount_col].sum()
                stat['total_count'] = subset_amount
                stat['sample_ratio'] = subset_amount / total_amount if total_amount > 0 else 0
            
            # 为每个标签计算指标
            for target in targets:
                if metric == 'count':
                    predicted_bad = subset[f'predicted_bad_rate_{target}'].sum()
                    adjusted_bad = subset[f'adjusted_bad_rate_{target}'].sum()
                    count = len(subset)
                else:
                    predicted_bad = (subset[f'predicted_bad_rate_{target}'] * subset[cfg.amount_col]).sum()
                    adjusted_bad = (subset[f'adjusted_bad_rate_{target}'] * subset[cfg.amount_col]).sum()
                    count = subset[cfg.amount_col].sum()
                
                stat[f'predicted_bad_count_{target}'] = predicted_bad
                stat[f'adjusted_bad_count_{target}'] = adjusted_bad
                stat[f'predicted_bad_rate_{target}'] = predicted_bad / count if count > 0 else 0
                stat[f'adjusted_bad_rate_{target}'] = adjusted_bad / count if count > 0 else 0
            
            stats[swap_type] = stat
        
        return stats
    
    def _empty_stats(self, swap_type: SwapType, targets: List[str]) -> Dict:
        """生成空的统计信息.
        
        :param swap_type: swap类型
        :param targets: 目标变量列表
        :return: 空统计字典
        """
        stat = {
            'swap_type': swap_type.value,
            'total_count': 0,
            'sample_ratio': 0.0,
        }
        for target in targets:
            stat[f'predicted_bad_count_{target}'] = 0.0
            stat[f'adjusted_bad_count_{target}'] = 0.0
            stat[f'predicted_bad_rate_{target}'] = 0.0
            stat[f'adjusted_bad_rate_{target}'] = 0.0
        return stat
    
    def _calculate_combined_stats(
        self,
        stats: Dict[SwapType, Dict],
        total_samples: int,
        metric: str = 'count',
        total_amount: Optional[float] = None
    ) -> Dict[str, Dict]:
        """计算组合统计信息.
        
        :param stats: 各swap类型统计
        :param total_samples: 总样本数
        :param metric: 统计口径
        :param total_amount: 总金额（可选）
        :return: 组合统计字典
        """
        combined = {}
        
        # 原策略通过 = in-in + in-out
        combined['original_pass'] = self._merge_stats(
            [stats[SwapType.IN_IN], stats[SwapType.IN_OUT]], "原策略通过"
        )
        
        # 新策略置出 = in-out
        combined['swap_out'] = stats[SwapType.IN_OUT].copy()
        combined['swap_out']['name'] = "新策略置出(in-out)"
        
        # 原策略保留 = in-in
        combined['original_keep'] = stats[SwapType.IN_IN].copy()
        combined['original_keep']['name'] = "原策略保留(in-in)"
        
        # 新策略置入 = out-in
        combined['swap_in'] = stats[SwapType.OUT_IN].copy()
        combined['swap_in']['name'] = "新策略置入(out-in)"
        
        # 新策略通过 = in-in + out-in
        combined['new_pass'] = self._merge_stats(
            [stats[SwapType.IN_IN], stats[SwapType.OUT_IN]], "新策略通过"
        )
        
        # 如果有out-out数据
        if stats[SwapType.OUT_OUT]['total_count'] > 0:
            combined['original_reject'] = self._merge_stats(
                [stats[SwapType.OUT_IN], stats[SwapType.OUT_OUT]], "原策略拒绝"
            )
            combined['new_reject'] = self._merge_stats(
                [stats[SwapType.IN_OUT], stats[SwapType.OUT_OUT]], "现策略拒绝"
            )
            combined['all'] = self._merge_stats(list(stats.values()), "全部样本")
        
        return combined
    
    def _merge_stats(self, stats_list: List[Dict], name: str) -> Dict:
        """合并多个统计信息.
        
        :param stats_list: 统计信息列表
        :param name: 合并后名称
        :return: 合并后统计字典
        """
        result = {'name': name}
        
        total_count = sum(s['total_count'] for s in stats_list)
        result['total_count'] = total_count
        result['sample_ratio'] = sum(s['sample_ratio'] for s in stats_list)
        
        # 合并各标签的指标
        target_cols = [k for k in stats_list[0].keys() if k.startswith('predicted_bad_count_')]
        targets = [k.replace('predicted_bad_count_', '') for k in target_cols]
        
        for target in targets:
            predicted_bad = sum(s.get(f'predicted_bad_count_{target}', 0) for s in stats_list)
            adjusted_bad = sum(s.get(f'adjusted_bad_count_{target}', 0) for s in stats_list)
            
            result[f'predicted_bad_count_{target}'] = predicted_bad
            result[f'adjusted_bad_count_{target}'] = adjusted_bad
            result[f'predicted_bad_rate_{target}'] = predicted_bad / total_count if total_count > 0 else 0
            result[f'adjusted_bad_rate_{target}'] = adjusted_bad / total_count if total_count > 0 else 0
        
        return result
    
    def _calculate_pass_rate_analysis(
        self,
        stats: Dict[SwapType, Dict],
        total_samples: int
    ) -> PassRateAnalysis:
        """计算通过率分析.
        
        :param stats: 各swap类型统计
        :param total_samples: 总样本数
        :return: PassRateAnalysis
        """
        cfg = self.config
        
        in_in_count = stats[SwapType.IN_IN]['total_count']
        in_out_count = stats[SwapType.IN_OUT]['total_count']
        out_out_count = stats[SwapType.OUT_OUT]['total_count']
        
        # 计算原策略通过率
        if out_out_count > 0:
            original_total = sum(s['total_count'] for s in stats.values())
            original_pass_rate = (in_in_count + in_out_count) / original_total if original_total > 0 else 0
        elif cfg.original_pass_rate is not None:
            original_pass_rate = cfg.original_pass_rate
        else:
            original_pass_rate = 1.0
        
        # 计算新策略通过率
        new_pass_count = in_in_count + stats[SwapType.OUT_IN]['total_count']
        new_pass_rate = new_pass_count / total_samples if total_samples > 0 else 0
        
        # 计算变化
        absolute_change = new_pass_rate - original_pass_rate
        # 相对变化 = (新-旧)/|旧| * 100%
        if original_pass_rate != 0:
            relative_change_pct = (absolute_change / abs(original_pass_rate)) * 100
        else:
            relative_change_pct = 0.0
        
        return PassRateAnalysis(
            original_pass_rate=original_pass_rate,
            new_pass_rate=new_pass_rate,
            absolute_change=absolute_change,
            relative_change_pct=relative_change_pct
        )
    
    def _calculate_risk_rejection_metrics(
        self,
        stats: Dict[SwapType, Dict],
        combined_stats: Dict[str, Dict],
        target: str
    ) -> RiskRejectionMetrics:
        """计算风险拒绝率指标.
        
        :param stats: 各swap类型统计
        :param combined_stats: 组合统计
        :param target: 目标变量名
        :return: RiskRejectionMetrics
        """
        original_pass = combined_stats.get('original_pass', {})
        new_pass = combined_stats.get('new_pass', {})
        
        bad_rate_before = original_pass.get(f'adjusted_bad_rate_{target}', 0)
        bad_rate_after = new_pass.get(f'adjusted_bad_rate_{target}', 0)
        bad_rate_improvement = bad_rate_before - bad_rate_after
        
        # 计算相对改善（百分比）
        if bad_rate_before > 0:
            bad_rate_relative_improvement = (bad_rate_improvement / bad_rate_before) * 100
        else:
            bad_rate_relative_improvement = 0.0
        
        in_out_ratio = stats[SwapType.IN_OUT]['sample_ratio']
        out_out_ratio = stats[SwapType.OUT_OUT]['sample_ratio']
        rejection_rate = in_out_ratio + out_out_ratio
        
        if rejection_rate > 0:
            risk_rejection_rate = bad_rate_improvement / rejection_rate
        else:
            risk_rejection_rate = 0.0
        
        pass_rate_analysis = self._calculate_pass_rate_analysis(stats, sum(s['total_count'] for s in stats.values()))
        pass_rate_drop = -pass_rate_analysis.absolute_change
        
        if pass_rate_drop > 0:
            pass_rate_drop_1pct_improvement = bad_rate_improvement / (pass_rate_drop * 100)
        else:
            pass_rate_drop_1pct_improvement = 0.0
        
        return RiskRejectionMetrics(
            rejection_rate=rejection_rate,
            risk_rejection_rate=risk_rejection_rate,
            pass_rate_drop_1pct_improvement=pass_rate_drop_1pct_improvement,
            bad_rate_before=bad_rate_before,
            bad_rate_after=bad_rate_after,
            bad_rate_improvement=bad_rate_improvement,
            bad_rate_relative_improvement=bad_rate_relative_improvement
        )


class SwapAnalysisResult:
    """Swap分析结果.
    
    :param count_stats: 订单口径统计
    :param amount_stats: 金额口径统计（可选）
    :param count_combined: 订单口径组合统计
    :param amount_combined: 金额口径组合统计（可选）
    :param config: Swap风险分析配置
    :param targets: 目标变量列表
    :param pass_rate_analysis: 通过率分析结果
    :param risk_rejection_metrics: 风险拒绝率指标
    :param risk_rejection_metrics_dict: 各标签的风险拒绝率指标
    :param total_samples: 总样本数
    """
    
    def __init__(
        self,
        count_stats: Dict[SwapType, Dict],
        amount_stats: Optional[Dict[SwapType, Dict]],
        count_combined: Dict[str, Dict],
        amount_combined: Optional[Dict[str, Dict]],
        config: SwapRiskConfig,
        targets: List[str],
        pass_rate_analysis: PassRateAnalysis,
        risk_rejection_metrics: RiskRejectionMetrics,
        risk_rejection_metrics_dict: Dict[str, RiskRejectionMetrics],
        total_samples: int
    ):
        self.count_stats = count_stats
        self.amount_stats = amount_stats
        self.count_combined = count_combined
        self.amount_combined = amount_combined
        self.config = config
        self.targets = targets
        self.pass_rate_analysis = pass_rate_analysis
        self.risk_rejection_metrics = risk_rejection_metrics
        self.risk_rejection_metrics_dict = risk_rejection_metrics_dict
        self.total_samples = total_samples
    
    def get_summary_report(self, metric: str = 'count', target: str = None) -> pd.DataFrame:
        """生成汇总报告.
        
        :param metric: 'count'订单口径或'amount'金额口径
        :param target: 指定标签，默认使用第一个
        :return: 汇总报告DataFrame
        """
        target = target or self.targets[0]
        combined = self.count_combined if metric == 'count' else self.amount_combined
        
        if combined is None:
            raise ValueError(f"未提供{metric}口径的数据")
        
        # 获取目标别名
        target_alias = self.config.target_aliases.get(target, target) if self.config.target_aliases else target
        
        rows = []
        key_order = ['original_pass', 'swap_out', 'original_keep', 'swap_in', 'new_pass']
        
        overall_bad_rate = combined.get('all', {}).get(f'adjusted_bad_rate_{target}', 0) or combined['new_pass'].get(f'adjusted_bad_rate_{target}', 0)
        
        for key in key_order:
            if key in combined:
                row = self._format_summary_row(combined[key], target, overall_bad_rate)
                
                # 添加通过率变化信息到关键行
                if key == 'original_pass':
                    row['原通过率'] = f"{self.pass_rate_analysis.original_pass_rate:.2%}"
                    row['通过率变化(绝对)'] = "-"
                    row['通过率变化(相对)'] = "-"
                elif key == 'swap_out':
                    row['原通过率'] = "-"
                    row['通过率变化(绝对)'] = "-"
                    row['通过率变化(相对)'] = "-"
                elif key == 'original_keep':
                    row['原通过率'] = "-"
                    row['通过率变化(绝对)'] = "-"
                    row['通过率变化(相对)'] = "-"
                elif key == 'swap_in':
                    row['原通过率'] = "-"
                    row['通过率变化(绝对)'] = "-"
                    row['通过率变化(相对)'] = "-"
                elif key == 'new_pass':
                    row['新通过率'] = f"{self.pass_rate_analysis.new_pass_rate:.2%}"
                    row['通过率变化(绝对)'] = f"{self.pass_rate_analysis.absolute_change:.2%}"
                    row['通过率变化(相对)'] = f"{self.pass_rate_analysis.relative_change_pct:.2f}%"
                
                # 添加风险指标到每行
                if overall_bad_rate > 0:
                    bad_rate = combined[key].get(f'adjusted_bad_rate_{target}', 0)
                    row['坏账率变化(绝对)'] = f"{bad_rate - overall_bad_rate:.4f}"
                    if overall_bad_rate > 0:
                        row['坏账率变化(相对)'] = f"{((bad_rate - overall_bad_rate) / overall_bad_rate) * 100:.2f}%"
                    else:
                        row['坏账率变化(相对)'] = "-"
                else:
                    row['坏账率变化(绝对)'] = "-"
                    row['坏账率变化(相对)'] = "-"
                
                rows.append(row)
        
        # 添加扩展流程
        if 'original_reject' in combined:
            rows.append({})
            extended_order = ['original_reject', 'swap_in', 'original_keep', 'swap_out', 'new_reject', 'new_pass', 'all']
            for key in extended_order:
                if key in combined:
                    row = self._format_summary_row(combined[key], target, overall_bad_rate)
                    
                    # 添加通过率变化
                    if key == 'original_reject':
                        row['原通过率'] = f"{self.pass_rate_analysis.original_pass_rate:.2%}"
                        row['通过率变化(绝对)'] = "-"
                        row['通过率变化(相对)'] = "-"
                    elif key == 'new_pass':
                        row['新通过率'] = f"{self.pass_rate_analysis.new_pass_rate:.2%}"
                        row['通过率变化(绝对)'] = f"{self.pass_rate_analysis.absolute_change:.2%}"
                        row['通过率变化(相对)'] = f"{self.pass_rate_analysis.relative_change_pct:.2f}%"
                    else:
                        row['原通过率'] = "-"
                        row['通过率变化(绝对)'] = "-"
                        row['通过率变化(相对)'] = "-"
                    
                    # 添加风险指标
                    if overall_bad_rate > 0:
                        bad_rate = combined[key].get(f'adjusted_bad_rate_{target}', 0)
                        row['坏账率变化(绝对)'] = f"{bad_rate - overall_bad_rate:.4f}"
                        if overall_bad_rate > 0:
                            row['坏账率变化(相对)'] = f"{((bad_rate - overall_bad_rate) / overall_bad_rate) * 100:.2f}%"
                        else:
                            row['坏账率变化(相对)'] = "-"
                    else:
                        row['坏账率变化(绝对)'] = "-"
                        row['坏账率变化(相对)'] = "-"
                    
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 统一列名
        columns = ['阶段', '样本数', '样本占比', '坏样本数', '坏样本率', 
                   'LIFT', '风险改善', '风险拒绝比',
                   '原通过率', '新通过率', '通过率变化(绝对)', '通过率变化(相对)',
                   '坏账率变化(绝对)', '坏账率变化(相对)']
        
        # 确保所有列存在
        for col in columns:
            if col not in df.columns:
                df[col] = None
        
        return df[columns]
    
    def _format_summary_row(self, stat: Dict, target: str, overall_bad_rate: float) -> Dict:
        """格式化汇总行.
        
        :param stat: 统计数据
        :param target: 目标变量名
        :param overall_bad_rate: 整体坏账率
        :return: 格式化后的行字典
        """
        if not stat:
            return {}
        
        bad_rate = stat.get(f'adjusted_bad_rate_{target}', 0)
        sample_ratio = stat.get('sample_ratio', 0)
        bad_count = stat.get(f'adjusted_bad_count_{target}', 0)
        total_count = stat.get('total_count', 0)
        
        # 计算LIFT
        lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 1.0
        
        # 计算风险改善
        bad_improvement = (overall_bad_rate - bad_rate) / overall_bad_rate if overall_bad_rate > 0 else 0
        
        # 计算风险拒绝比
        risk_reject_ratio = bad_improvement / sample_ratio if sample_ratio > 0 else 0
        
        return {
            '阶段': stat.get('name', ''),
            '样本数': total_count,
            '样本占比': f"{sample_ratio:.2%}",
            '坏样本数': f"{bad_count:.1f}",
            '坏样本率': f"{bad_rate:.4f}",
            'LIFT': f"{lift:.4f}",
            '风险改善': f"{bad_improvement:.4f}",
            '风险拒绝比': f"{risk_reject_ratio:.4f}",
        }
    
    @property
    def summary_report_count(self) -> pd.DataFrame:
        """订单口径汇总报告."""
        return self.get_summary_report('count')
    
    @property
    def summary_report_amount(self) -> Optional[pd.DataFrame]:
        """金额口径汇总报告."""
        if self.amount_combined:
            return self.get_summary_report('amount')
        return None
    
    def get_detail_report(self, metric: str = 'count', target: str = None) -> pd.DataFrame:
        """生成详细报告（四象限）.
        
        :param metric: 'count'订单口径或'amount'金额口径
        :param target: 指定标签，默认使用第一个
        :return: 详细报告DataFrame
        """
        target = target or self.targets[0]
        stats = self.count_stats if metric == 'count' else self.amount_stats
        
        # 获取目标别名
        target_alias = self.config.target_aliases.get(target, target) if self.config.target_aliases else target
        
        rows = []
        for swap_type in SwapType:
            stat = stats[swap_type]
            row = {
                'Swap类型': swap_type.value,
                '说明': self._get_swap_description(swap_type),
                '样本数': stat['total_count'],
                '样本占比': f"{stat['sample_ratio']:.2%}",
                f'坏样本率({target_alias})': f"{stat.get(f'adjusted_bad_rate_{target}', 0):.4f}",
                '风险上浮': f"{self.config.out_in_uplift:.1f}x" if swap_type == SwapType.OUT_IN else "-",
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _get_swap_description(self, swap_type: SwapType) -> str:
        """获取swap类型的说明.
        
        :param swap_type: swap类型
        :return: 说明字符串
        """
        descriptions = {
            SwapType.IN_IN: "原策略通过，新策略通过",
            SwapType.IN_OUT: "原策略通过，新策略拒绝",
            SwapType.OUT_IN: "原策略拒绝，新策略通过（置入）",
            SwapType.OUT_OUT: "原策略拒绝，新策略拒绝",
        }
        return descriptions.get(swap_type, "")
    
    @property
    def pass_rate_report(self) -> pd.DataFrame:
        """生成通过率分析报告."""
        return pd.DataFrame([self.pass_rate_analysis.to_dict()])
    
    @property
    def risk_rejection_report(self) -> pd.DataFrame:
        """生成风险拒绝率报告."""
        return pd.DataFrame([self.risk_rejection_metrics.to_dict()])
    
    def get_risk_rejection_report_by_target(self, target: str = None) -> pd.DataFrame:
        """获取指定标签的风险拒绝率报告.
        
        :param target: 目标变量名，默认使用第一个
        :return: 风险拒绝率报告DataFrame
        """
        target = target or self.targets[0]
        if target in self.risk_rejection_metrics_dict:
            return pd.DataFrame([self.risk_rejection_metrics_dict[target].to_dict()])
        return pd.DataFrame()


def create_swap_dataset(
    df: pd.DataFrame,
    original_rule_col: str,
    new_rule_col: str,
    score_col: str,
    swap_type_col: str = "swap_type",
    amount_col: Optional[str] = None,
    rule_type: str = "reject"
) -> pd.DataFrame:
    """创建swap数据集.
    
    :param df: 输入数据
    :param original_rule_col: 原策略规则字段
    :param new_rule_col: 新策略规则字段
    :param score_col: 评分字段
    :param swap_type_col: swap类型输出字段名
    :param amount_col: 金额字段（可选）
    :param rule_type: 规则类型，"reject"表示拒绝规则（1=拒绝,0=通过），
                      "pass"表示通过规则（1=通过,0=拒绝）
    :return: 包含swap_type的数据集
    """
    df = df.copy()
    
    if rule_type == "reject":
        # 拒绝规则：1=拒绝，0=通过
        # 通过 = 非拒绝
        original_pass = (df[original_rule_col] == 0)
        new_pass = (df[new_rule_col] == 0)
    else:
        # 通过规则：1=通过，0=拒绝
        original_pass = (df[original_rule_col] == 1)
        new_pass = (df[new_rule_col] == 1)
    
    conditions = [
        original_pass & new_pass,           # in-in: 原通过，新通过
        original_pass & (~new_pass),        # in-out: 原通过，新拒绝
        (~original_pass) & new_pass,        # out-in: 原拒绝，新通过
        (~original_pass) & (~new_pass),     # out-out: 原拒绝，新拒绝
    ]
    choices = [SwapType.IN_IN.value, SwapType.IN_OUT.value, 
               SwapType.OUT_IN.value, SwapType.OUT_OUT.value]
    
    df[swap_type_col] = np.select(conditions, choices, default="unknown")
    
    cols = [score_col, swap_type_col]
    if amount_col:
        cols.append(amount_col)
    
    return df[cols]


def create_swap_dataset_from_rules(
    df: pd.DataFrame,
    original_rule,
    new_rule,
    score_col: str,
    swap_type_col: str = "swap_type",
    amount_col: Optional[str] = None,
    rule_type: str = "reject",
    original_rule_name: str = "original_reject",
    new_rule_name: str = "new_reject"
) -> pd.DataFrame:
    """从Rule对象创建swap数据集.
    
    :param df: 输入数据
    :param original_rule: 原策略Rule对象
    :param new_rule: 新策略Rule对象
    :param score_col: 评分字段
    :param swap_type_col: swap类型输出字段名
    :param amount_col: 金额字段（可选）
    :param rule_type: 规则类型，"reject"表示拒绝规则，"pass"表示通过规则
    :param original_rule_name: 原策略规则结果临时字段名
    :param new_rule_name: 新策略规则结果临时字段名
    :return: 包含swap_type的数据集
    """
    df = df.copy()
    
    # 应用规则
    df[original_rule_name] = original_rule.predict(df)
    df[new_rule_name] = new_rule.predict(df)
    
    return create_swap_dataset(
        df, 
        original_rule_name, 
        new_rule_name, 
        score_col, 
        swap_type_col, 
        amount_col,
        rule_type
    )


def swap_analysis(
    swap_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    score_col: str = "score",
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    swap_type_col: str = "swap_type",
    amount_col: Optional[str] = None,
    out_in_uplift: float = 2.0,
    original_pass_rate: Optional[float] = None,
    target_aliases: Optional[Dict[str, str]] = None,
    **kwargs
) -> SwapAnalysisResult:
    """统一的Swap分析入口函数.
    
    传入数据集和参数配置，直接得到完整的swap分析结果。
    
    :param swap_df: swap数据集（含swap_type和score）
    :param reference_df: 参考数据集（含score和target/overdue）
    :param score_col: 评分字段名，用于风险预估
    :param target: 目标变量字段名（可选），与overdue+dpds二选一
    :param overdue: 逾期天数字段名或列表（可选），如'MOB1'或['MOB1','MOB3']
    :param dpds: 逾期定义天数或列表（可选），如15或[15,30]
        - 逾期天数>dpds为坏样本(1)，其他为好样本(0)
    :param swap_type_col: swap类型字段名
    :param amount_col: 金额字段名（可选），用于金额口径分析
    :param out_in_uplift: out-in风险上浮因子，默认2.0
    :param original_pass_rate: 原策略通过率（可选），用于无out-out数据场景
    :param target_aliases: 目标变量别名（可选），如{'target_dpd15': 'DPD15+'}
    :param kwargs: 其他配置参数，如bin_method, max_n_bins, custom_bins等
    :return: SwapAnalysisResult分析结果对象
    
    **示例**
    
    >>> # 单标签分析
    >>> result = swap_analysis(
    ...     swap_df, reference_df,
    ...     score_col='score',
    ...     target='target_dpd15'
    ... )
    >>> 
    >>> # 多标签分析（使用overdue+dpds）
    >>> result = swap_analysis(
    ...     swap_df, reference_df,
    ...     score_col='score',
    ...     overdue='MOB1',
    ...     dpds=[15, 30],
    ...     target_aliases={'MOB1_15+': 'DPD15+', 'MOB1_30+': 'DPD30+'}
    ... )
    >>> 
    >>> # 获取结果
    >>> result.summary_report_count  # 订单口径汇总报告
    >>> result.summary_report_amount  # 金额口径汇总报告（如有）
    >>> result.pass_rate_report  # 通过率分析报告
    >>> result.risk_rejection_report  # 风险拒绝率报告
    """
    # 构建目标变量列表
    target_cols = []
    if overdue is not None and dpds is not None:
        # 逾期分析模式
        if isinstance(overdue, str):
            overdue = [overdue]
        if isinstance(dpds, int):
            dpds = [dpds]
        
        for mob_col in overdue:
            for d in dpds:
                target_name = f"{mob_col}_{d}+"
                reference_df[target_name] = (reference_df[mob_col] > d).astype(int)
                target_cols.append(target_name)
    elif target is not None:
        # 普通目标模式
        target_cols = [target]
    else:
        raise ValueError("必须传入target或overdue+dpds参数")
    
    # 构建配置
    config = SwapRiskConfig(
        score_col=score_col,
        swap_type_col=swap_type_col,
        amount_col=amount_col,
        out_in_uplift=out_in_uplift,
        original_pass_rate=original_pass_rate,
        targets=target_cols,
        target_aliases=target_aliases or {},
        **{k: v for k, v in kwargs.items() if k in ['bin_method', 'max_n_bins', 'custom_bins']}
    )
    
    # 创建参考数据提供者
    ref_provider = ReferenceDataProvider(
        score_col=score_col,
        target_cols=target_cols,
        amount_col=amount_col,
        **{k: v for k, v in kwargs.items() if k in ['method', 'max_n_bins', 'custom_bins']}
    )
    ref_provider.fit(reference_df)
    
    # 执行分析
    analyzer = SwapAnalyzer(config, ref_provider)
    return analyzer.analyze(swap_df)
