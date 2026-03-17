"""规则置换风险预估模块.

提供金融信贷业务中规则置换时的风险指标分析功能，
用于预估置换后放款的风险指标。

主要功能:
- 基于历史数据的分箱规则，预估新样本的风险表现
- 支持单一目标变量和多逾期标签分析
- 支持按金额加权计算风险指标
- 输出包含LIFT、坏账改善、风险拒绝比等指标的风险预估表

参考: scorecardpipeline.sawpin_badrate_prediction_by_score
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from functools import reduce

from ..core.binning import OptimalBinning
from ..core.binning.base import BaseBinning
from ..core.rules import Rule


def _calculate_group_risk(group: pd.DataFrame, amount: Optional[str] = None) -> pd.Series:
    """计算分组风险指标.
    
    :param group: 分组数据
    :param amount: 金额字段名称，如果提供则按金额加权计算
    :return: 风险指标Series
    """
    if amount is None:
        return pd.Series({
            '样本总数': len(group),
            '坏样本数': group['BAD_RATE'].sum(),
            '坏样本率': group['BAD_RATE'].mean(),
        })
    else:
        total_amount = group[amount].sum()
        if total_amount == 0:
            bad_rate = 0.0
        else:
            bad_rate = (group['BAD_RATE'] * group[amount]).sum() / total_amount
        return pd.Series({
            '样本总数': total_amount,
            '坏样本数': (group['BAD_RATE'] * group[amount]).sum(),
            '坏样本率': bad_rate,
        })


def swapin_risk_prediction(
    base: pd.DataFrame,
    test: pd.DataFrame,
    swapin_rules: Union[Rule, List[Rule]],
    feature: str,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    rules: Optional[Union[List, Dict[str, List]]] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    amount: Optional[str] = None,
    desc: str = '',
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """规则置换风险预估 - 基于历史数据预估新样本的风险指标.
    
    该函数用于金融信贷业务中规则置换场景，分析置入规则的风险表现，
    预估置换后放款的风险指标（坏样本率、LIFT、坏账改善等）。
    
    :param base: pd.DataFrame, 有表现的历史数据集，用于建立分箱规则
    :param test: pd.DataFrame, 测试数据集（包含置入样本），需要预估风险
    :param swapin_rules: Rule或List[Rule], 置入规则列表
    :param feature: str, 用于分箱的特征名称（如信用分）
    :param target: str, 目标变量名称（与overdue+dpds二选一）
    :param overdue: str或List[str], 逾期天数字段名称，如'MOB1'或['MOB1', 'MOB3']
    :param dpds: int或List[int], 逾期定义天数，如7或[0, 7, 30]
        - 逾期天数 > dpd 为坏样本(1)，其他为好样本(0)
    :param rules: list或dict, 自定义分箱规则
        - list: [300, 500, 700] 表示切分点
        - dict: {'score': [300, 500, 700]} 多特征时使用
    :param method: str, 分箱方法，可选：
        - 基础方法: 'uniform'(等宽), 'quantile'(等频), 'tree'(决策树), 'chi_merge'(卡方)
        - 优化方法: 'optimal_ks'(最优KS), 'optimal_iv'(最优IV), 'mdlp'(信息论)
        默认: 'quantile'
    :param max_n_bins: int, 最大分箱数，默认10
    :param amount: str, 金额字段名称，如果提供则按金额加权计算风险指标
    :param desc: str, 特征描述
    :param kwargs: 其他分箱器参数，如monotonic='ascending'等
    
    :return: 
        - pd.DataFrame: 单一目标时的风险预估表
        - Tuple[pd.DataFrame, pd.DataFrame]: (数量维度表, 金额维度表) 当amount参数提供时
        
    **输出指标说明**
    
    - 样本总数: 该分组的总样本数（或总金额）
    - 坏样本数: 该分组的坏样本数（或加权坏样本数）
    - 坏样本率: 该分组的坏样本率
    - 样本占比: 该分组占总样本的比例
    - LIFT值: 该分组坏样本率 / 整体坏样本率，>1表示风险高于平均
    - 坏账改善: (整体坏样本率 - 该分组坏样本率) / 整体坏样本率
    - 风险拒绝比: 坏账改善 / 样本占比，衡量拒绝效率
    
    **使用示例**
    
    >>> from hscredit import Rule
    >>> from hscredit.report import swapin_risk_prediction
    >>> 
    >>> # 定义置入规则
    >>> swapin_rules = [
    >>>     Rule("(当前履约机构数 < 22) & (自营资质分 >= 0) & (自营资质分 < 555)")
    >>> ]
    >>> 
    >>> # 单一target分析
    >>> risk_table = swapin_risk_prediction(
    >>>     base=train_data,
    >>>     test=new_data,
    >>>     swapin_rules=swapin_rules,
    >>>     feature='自营资质分',
    >>>     target='target',
    >>>     amount='放款金额'
    >>> )
    >>> print(risk_table)
    >>> 
    >>> # 多逾期标签分析
    >>> risk_table = swapin_risk_prediction(
    >>>     base=train_data,
    >>>     test=new_data,
    >>>     swapin_rules=swapin_rules,
    >>>     feature='自营资质分',
    >>>     overdue=['MOB1', 'MOB3'],
    >>>     dpds=[0, 7, 30],
    >>>     amount='放款金额'
    >>> )
    >>> print(risk_table)
    """
    test = test.copy()
    
    # 统一处理置入规则
    if isinstance(swapin_rules, list):
        if len(swapin_rules) == 0:
            raise ValueError("swapin_rules不能为空列表")
        rule_swapin = reduce(lambda r1, r2: r1 | r2, swapin_rules)
    else:
        rule_swapin = swapin_rules
    
    # 预测置入标记
    test['SWAPIN'] = rule_swapin.predict(test).astype(int)
    
    # 检查参数
    if target is None and (overdue is None or dpds is None):
        raise ValueError("必须传入target或(overdue+dpds)参数")
    
    # 如果没有指定overdue和dpd，使用单一target进行分析
    if overdue is None or dpds is None:
        return _analyze_single_target(
            base=base,
            test=test,
            feature=feature,
            target=target,
            rule_expr=rule_swapin.expr,
            rules=rules,
            method=method,
            max_n_bins=max_n_bins,
            amount=amount,
            desc=desc,
            **kwargs
        )
    else:
        # 处理多个逾期标签的情况
        return _analyze_multi_overdue(
            base=base,
            test=test,
            feature=feature,
            overdue=overdue,
            dpds=dpds,
            rule_expr=rule_swapin.expr,
            rules=rules,
            method=method,
            max_n_bins=max_n_bins,
            amount=amount,
            desc=desc,
            **kwargs
        )


def _analyze_single_target(
    base: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    target: str,
    rule_expr: str,
    rules: Optional[Union[List, Dict]] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    amount: Optional[str] = None,
    desc: str = '',
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """单一目标变量的风险分析."""
    
    # 在base数据上建立分箱
    if rules is not None:
        # 使用自定义规则
        if isinstance(rules, dict) and feature in rules:
            custom_splits = np.array(rules[feature])
        elif isinstance(rules, list):
            custom_splits = np.array(rules)
        else:
            custom_splits = np.array([])
        
        # 创建分箱器并手动设置状态
        binner = OptimalBinning(method=method, max_n_bins=max_n_bins, **kwargs)
        binner.splits_ = {feature: custom_splits}
        binner.feature_types_ = {feature: 'numerical'}
        binner.n_bins_ = {feature: len(custom_splits) + 1}
        binner._is_fitted = True
        
        # 生成bin_table用于后续计算
        bins = np.digitize(base[feature].values, custom_splits, right=True)
        from ..core.metrics.binning_metrics import compute_bin_stats
        temp_stats = compute_bin_stats(bins, base[target].values)
        binner.bin_tables_ = {feature: temp_stats}
    else:
        # 使用分箱方法拟合
        binner = OptimalBinning(method=method, max_n_bins=max_n_bins, **kwargs)
        binner.fit(base[[feature]], base[target])
    
    # 对test数据进行分箱转换（使用索引而非标签）
    test['BINS'] = binner.transform(test[[feature]], metric='indices')[feature].values
    
    # 获取各分箱的坏样本率（从base数据）
    bin_table = binner.bin_tables_[feature]
    # 使用'分箱'列作为键建立映射（分箱列包含数值索引）
    bad_rate_map = dict(zip(bin_table['分箱'], bin_table['坏样本率']))
    test['BAD_RATE'] = test['BINS'].map(bad_rate_map).fillna(0)
    
    # 按SWAPIN分组计算风险指标
    result_by_swapin = test.groupby('SWAPIN').apply(
        lambda x: _calculate_group_risk(x, amount)
    ).sort_index(ascending=False)
    
    result_by_swapin.index = [rule_expr, '剩余样本']
    result_by_swapin.index.name = '规则详情'
    
    # 添加合计行
    total_row = _calculate_group_risk(test, amount)
    result_by_swapin.loc['合计', :] = total_row
    
    # 计算衍生指标
    overall_bad_rate = result_by_swapin.loc['合计', '坏样本率']
    total_samples = result_by_swapin.loc['合计', '样本总数']
    
    result_by_swapin = result_by_swapin.assign(
        样本占比=lambda x: x['样本总数'] / total_samples,
        LIFT值=lambda x: x['坏样本率'] / overall_bad_rate if overall_bad_rate > 0 else 1.0,
        坏账改善=lambda x: (overall_bad_rate - x['坏样本率']) / overall_bad_rate if overall_bad_rate > 0 else 0.0,
        风险拒绝比=lambda x: x['坏账改善'] / x['样本占比'] if x['样本占比'].iloc[0] > 0 else 0.0,
    )
    
    # 如果提供了金额字段，同时返回金额维度的分析
    if amount is not None:
        # 计算数量维度的结果
        result_count = test.groupby('SWAPIN').apply(
            lambda x: _calculate_group_risk(x, amount=None)
        ).sort_index(ascending=False)
        result_count.index = [rule_expr, '剩余样本']
        result_count.index.name = '规则详情'
        
        total_row_count = _calculate_group_risk(test, amount=None)
        result_count.loc['合计', :] = total_row_count
        
        overall_bad_rate_count = result_count.loc['合计', '坏样本率']
        total_samples_count = result_count.loc['合计', '样本总数']
        
        result_count = result_count.assign(
            样本占比=lambda x: x['样本总数'] / total_samples_count,
            LIFT值=lambda x: x['坏样本率'] / overall_bad_rate_count if overall_bad_rate_count > 0 else 1.0,
            坏账改善=lambda x: (overall_bad_rate_count - x['坏样本率']) / overall_bad_rate_count if overall_bad_rate_count > 0 else 0.0,
            风险拒绝比=lambda x: x['坏账改善'] / x['样本占比'] if x['样本占比'].iloc[0] > 0 else 0.0,
        )
        
        return result_count, result_by_swapin
    
    return result_by_swapin


def _analyze_multi_overdue(
    base: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    overdue: Union[str, List[str]],
    dpds: Union[int, List[int]],
    rule_expr: str,
    rules: Optional[Union[List, Dict]] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    amount: Optional[str] = None,
    desc: str = '',
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """多逾期标签的风险分析."""
    
    # 统一处理为列表
    if isinstance(overdue, str):
        overdue = [overdue]
    if isinstance(dpds, int):
        dpds = [dpds]
    
    merge_columns = ['样本总数', '样本占比']
    result_final, result_amount_final = None, None
    
    amount_feature = [amount] if amount is not None else []
    
    # 遍历所有逾期标签组合
    for i, col in enumerate(overdue):
        for j, d in enumerate(dpds):
            target_name = f"{col}_{d}+"
            
            # 在base数据上创建新的目标变量
            _base = base[[feature] + amount_feature + [col]].copy()
            _base[target_name] = (_base[col] > d).astype(int)
            
            # 递归调用单一目标分析
            result = _analyze_single_target(
                base=_base,
                test=test,
                feature=feature,
                target=target_name,
                rule_expr=rule_expr,
                rules=rules,
                method=method,
                max_n_bins=max_n_bins,
                amount=amount,
                desc=desc,
                **kwargs
            )
            
            # 处理返回结果
            if amount is not None:
                result_count, result_amount = result
            else:
                result_count = result
                result_amount = None
            
            # 重命名列名为多级索引
            result_count.columns = pd.MultiIndex.from_tuples(
                [("规则详情", c) if c in merge_columns else (target_name, c) 
                 for c in result_count.columns]
            )
            
            if result_amount is not None:
                result_amount.columns = pd.MultiIndex.from_tuples(
                    [("规则详情", c) if c in merge_columns else (target_name, c) 
                     for c in result_amount.columns]
                )
            
            # 合并结果
            if result_final is None:
                result_final = result_count
                result_amount_final = result_amount
            else:
                # 合并数量维度
                merge_on = [("规则详情", c) for c in merge_columns]
                result_final = result_final.merge(
                    result_count.drop(columns=[("规则详情", c) for c in merge_columns]),
                    left_index=True, right_index=True
                )
                
                if result_amount is not None and result_amount_final is not None:
                    result_amount_final = result_amount_final.merge(
                        result_amount.drop(columns=[("规则详情", c) for c in merge_columns]),
                        left_index=True, right_index=True
                    )
    
    # 重新排列列顺序
    def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        rule_columns = [col for col in df.columns if col[0] == "规则详情"]
        other_columns = [col for col in df.columns if col[0] != "规则详情"]
        return df[rule_columns + other_columns]
    
    if result_final is not None:
        result_final = reorder_columns(result_final)
        if result_amount_final is not None:
            result_amount_final = reorder_columns(result_amount_final)
    
    if amount is not None:
        return result_final, result_amount_final
    
    return result_final


class SwapinRiskAnalyzer:
    """规则置换风险分析器.
    
    提供批量规则置换风险分析功能，支持多规则、多特征、多目标的综合分析。
    
    **使用示例**
    
    >>> from hscredit import Rule
    >>> from hscredit.report import SwapinRiskAnalyzer
    >>> 
    >>> # 初始化分析器
    >>> analyzer = SwapinRiskAnalyzer(
    ...     method='optimal_iv',
    ...     max_n_bins=5
    ... )
    >>> 
    >>> # 定义多个置入规则
    >>> rules = [
    ...     Rule("score < 500"),
    ...     Rule("(age < 25) & (income < 3000)")
    ... ]
    >>> 
    >>> # 批量分析
    >>> results = analyzer.analyze(
    ...     base=train_data,
    ...     test=new_data,
    ...     swapin_rules=rules,
    ...     feature='score',
    ...     target='target'
    ... )
    """
    
    def __init__(
        self,
        method: str = 'quantile',
        max_n_bins: int = 10,
        **kwargs
    ):
        """初始化风险分析器.
        
        :param method: 分箱方法
        :param max_n_bins: 最大分箱数
        :param kwargs: 其他分箱器参数
        """
        self.method = method
        self.max_n_bins = max_n_bins
        self.kwargs = kwargs
    
    def analyze(
        self,
        base: pd.DataFrame,
        test: pd.DataFrame,
        swapin_rules: Union[Rule, List[Rule]],
        feature: str,
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        amount: Optional[str] = None,
        desc: str = '',
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """执行规则置换风险分析.
        
        :param base: 历史数据集
        :param test: 测试数据集
        :param swapin_rules: 置入规则
        :param feature: 特征名称
        :param target: 目标变量
        :param overdue: 逾期字段
        :param dpds: 逾期天数阈值
        :param amount: 金额字段
        :param desc: 特征描述
        :param kwargs: 其他参数
        :return: 风险分析结果
        """
        # 合并参数
        analysis_kwargs = {
            'method': kwargs.get('method', self.method),
            'max_n_bins': kwargs.get('max_n_bins', self.max_n_bins),
            **self.kwargs,
            **kwargs
        }
        
        return swapin_risk_prediction(
            base=base,
            test=test,
            swapin_rules=swapin_rules,
            feature=feature,
            target=target,
            overdue=overdue,
            dpds=dpds,
            amount=amount,
            desc=desc,
            **analysis_kwargs
        )
    
    def compare_features(
        self,
        base: pd.DataFrame,
        test: pd.DataFrame,
        swapin_rules: Union[Rule, List[Rule]],
        features: List[str],
        target: str,
        amount: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """对比不同特征的风险分析结果.
        
        :param base: 历史数据集
        :param test: 测试数据集
        :param swapin_rules: 置入规则
        :param features: 特征名称列表
        :param target: 目标变量
        :param amount: 金额字段
        :param kwargs: 其他参数
        :return: 各特征的分析结果字典
        """
        results = {}
        for feat in features:
            result = self.analyze(
                base=base,
                test=test,
                swapin_rules=swapin_rules,
                feature=feat,
                target=target,
                amount=amount,
                desc=feat,
                **kwargs
            )
            results[feat] = result
        return results


def swapin_summary_report(
    base: pd.DataFrame,
    test: pd.DataFrame,
    swapin_rules: List[Rule],
    feature: str,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    amount: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """规则置换汇总报告 - 对比多个规则的置换效果.
    
    对比分析多条置入规则的风险表现，输出汇总对比表。
    
    :param base: 历史数据集
    :param test: 测试数据集
    :param swapin_rules: 多条置入规则列表
    :param feature: 特征名称
    :param target: 目标变量
    :param overdue: 逾期字段
    :param dpds: 逾期天数阈值
    :param amount: 金额字段
    :param kwargs: 其他参数
    :return: 汇总对比表
    """
    results = []
    
    for rule in swapin_rules:
        result = swapin_risk_prediction(
            base=base,
            test=test,
            swapin_rules=rule,
            feature=feature,
            target=target,
            overdue=overdue,
            dpds=dpds,
            amount=amount,
            **kwargs
        )
        
        # 提取置入规则的指标
        if isinstance(result, tuple):
            result = result[0]  # 使用数量维度
        
        swapin_row = result.iloc[0].copy()
        swapin_row['规则表达式'] = rule.expr
        results.append(swapin_row)
    
    return pd.DataFrame(results)
