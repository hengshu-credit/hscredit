"""规则评估指标模块.

提供丰富的规则评估指标，所有指标计算统一收口到hscredit.core.metrics。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# 从统一metrics模块导入指标
from ....core.metrics import (
    KS, AUC, Gini,
    PSI, PSI_table,
    IV, IV_table,
    lift as metrics_lift,
    lift_table as metrics_lift_table,
    rule_lift,
    badrate
)


class RuleMetrics:
    """规则评估指标计算器.
    
    提供全面的规则评估指标，支持训练集和测试集的对比分析。
    所有指标计算统一收口到hscredit.core.metrics。
    
    示例:
        >>> from hscredit.core.rules.mining import RuleMetrics
        >>> metrics = RuleMetrics()
        >>> 
        >>> # 评估单个规则
        >>> result = metrics.evaluate_rule(rule, X_train, y_train, X_test, y_test)
        >>> 
        >>> # 批量评估
        >>> results = metrics.evaluate_rules(rules, X_train, y_train, X_test, y_test)
    """
    
    def __init__(self, target_positive: int = 1):
        """
        :param target_positive: 正类标签，默认1
        """
        self.target_positive = target_positive
    
    def evaluate_rule(
        self,
        rule,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        amount_train: Optional[pd.Series] = None,
        amount_test: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """评估单个规则.
        
        :param rule: 规则对象（Rule或MinedRule）
        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :param X_test: 测试集特征（可选）
        :param y_test: 测试集标签（可选）
        :param amount_train: 训练集金额（可选）
        :param amount_test: 测试集金额（可选）
        :return: 评估指标字典
        """
        result = {}
        
        # 训练集评估
        train_metrics = self._calculate_metrics(
            rule, X_train, y_train, amount_train
        )
        for k, v in train_metrics.items():
            result[f'train_{k}'] = v
        
        # 测试集评估
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(
                rule, X_test, y_test, amount_test
            )
            for k, v in test_metrics.items():
                result[f'test_{k}'] = v
            
            # 稳定性指标 - 使用统一的PSI计算
            result['badrate_diff'] = (
                train_metrics.get('lift', 0) - test_metrics.get('lift', 0)
            )
        
        return result
    
    def _calculate_metrics(
        self,
        rule,
        X: pd.DataFrame,
        y: pd.Series,
        amount: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """计算基础指标.
        
        使用hscredit.core.metrics.rule_lift统一计算Lift指标。
        
        :param rule: 规则对象
        :param X: 特征数据
        :param y: 标签
        :param amount: 金额
        :return: 指标字典
        """
        # 应用规则
        if hasattr(rule, 'evaluate'):
            mask = rule.evaluate(X)
        elif hasattr(rule, 'predict'):
            mask = rule.predict(X)
        else:
            raise ValueError("规则对象必须有evaluate或predict方法")
        
        if isinstance(mask, pd.Series):
            mask = mask.values
        
        # 使用统一的rule_lift计算指标
        lift_result = rule_lift(y, mask, amount)
        
        # 分类指标
        y_pred = np.zeros_like(y)
        y_pred[mask] = 1
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # 压降分析
        total_bad = y.sum()
        total = len(y)
        hit_bad = lift_result['bad_count']
        hit_count = lift_result['hit_count']
        overall_badrate = total_bad / total if total > 0 else 0
        
        badrate_after = (total_bad - hit_bad) / (total - hit_count) if (total - hit_count) > 0 else 0
        badrate_reduction = (overall_badrate - badrate_after) / overall_badrate if overall_badrate > 0 else 0
        
        metrics = {
            'hit_count': lift_result['hit_count'],
            'hit_rate': lift_result['hit_rate'],
            'bad_count': lift_result['bad_count'],
            'good_count': lift_result['good_count'],
            'badrate': lift_result['badrate'],
            'lift': lift_result['lift'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn),
            'badrate_after_interception': badrate_after,
            'badrate_reduction': badrate_reduction
        }
        
        # 金额口径指标
        if 'amount_lift' in lift_result:
            metrics.update({
                'total_amount': lift_result.get('total_amount', 0),
                'selected_amount': lift_result.get('hit_amount', 0),
                'amount_hit_rate': lift_result.get('hit_amount', 0) / lift_result.get('total_amount', 1) if lift_result.get('total_amount', 0) > 0 else 0,
                'loss_rate': lift_result.get('bad_amount', 0) / lift_result.get('hit_amount', 1) if lift_result.get('hit_amount', 0) > 0 else 0,
                'loss_lift': lift_result.get('amount_lift', 0)
            })
        
        return metrics
    
    def evaluate_rules(
        self,
        rules: List,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        **kwargs
    ) -> pd.DataFrame:
        """批量评估规则.
        
        :param rules: 规则列表
        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :param X_test: 测试集特征
        :param y_test: 测试集标签
        :return: 评估结果DataFrame
        """
        results = []
        
        for i, rule in enumerate(rules):
            try:
                metrics = self.evaluate_rule(
                    rule, X_train, y_train, X_test, y_test
                )
                metrics['rule_id'] = i
                metrics['rule'] = str(rule)
                results.append(metrics)
            except Exception as e:
                print(f"评估规则 {i} 时出错: {str(e)}")
        
        return pd.DataFrame(results)
    
    def calculate_ks(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算KS统计量.
        
        使用统一的KS计算。
        
        :param y_true: 真实标签
        :param y_score: 预测分数
        :return: KS值
        """
        return KS(y_true, y_score)
    
    def calculate_iv(
        self,
        feature: pd.Series,
        target: pd.Series,
        n_bins: int = 10
    ) -> float:
        """计算IV值（信息价值）.
        
        使用统一的IV计算。
        
        :param feature: 特征值
        :param target: 目标变量
        :param n_bins: 分箱数
        :return: IV值
        """
        return IV(target, feature, bins=n_bins)
    
    def calculate_gini(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算Gini系数.
        
        使用统一的Gini计算。
        
        :param y_true: 真实标签
        :param y_score: 预测分数
        :return: Gini系数
        """
        return Gini(y_true, y_score)


def calculate_rule_metrics(
    rule,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """便捷函数：计算规则评估指标.
    
    :param rule: 规则对象
    :param X: 训练集特征
    :param y: 训练集标签
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    :return: 评估指标字典
    """
    metrics = RuleMetrics()
    return metrics.evaluate_rule(rule, X, y, X_test, y_test)


def calculate_lift_chart(
    y_true: pd.Series,
    y_score: pd.Series,
    n_buckets: int = 10
) -> pd.DataFrame:
    """计算Lift图表数据.
    
    使用统一的lift_table计算。
    
    :param y_true: 真实标签
    :param y_score: 预测分数
    :param n_buckets: 分桶数
    :return: Lift数据DataFrame
    """
    return metrics_lift_table(y_true, y_score, n_bins=n_buckets)
