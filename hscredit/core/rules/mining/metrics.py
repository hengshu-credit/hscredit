"""规则评估指标模块.

提供丰富的规则评估指标，包括基础指标、高级指标和稳定性指标。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


class RuleMetrics:
    """规则评估指标计算器.
    
    提供全面的规则评估指标，支持训练集和测试集的对比分析。
    
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
            
            # 稳定性指标
            result['psi'] = self._calculate_psi(
                train_metrics.get('badrate', 0),
                test_metrics.get('badrate', 0)
            )
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
        
        # 基础统计
        total = len(y)
        hit_count = mask.sum()
        hit_bad = y[mask].sum() if hit_count > 0 else 0
        hit_good = hit_count - hit_bad
        
        total_bad = y.sum()
        total_good = total - total_bad
        
        # 基础指标
        hit_rate = hit_count / total if total > 0 else 0
        badrate = hit_bad / hit_count if hit_count > 0 else 0
        overall_badrate = total_bad / total if total > 0 else 0
        lift = badrate / overall_badrate if overall_badrate > 0 else 0
        
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
        badrate_after = (total_bad - hit_bad) / (total - hit_count) if (total - hit_count) > 0 else 0
        badrate_reduction = (overall_badrate - badrate_after) / overall_badrate if overall_badrate > 0 else 0
        
        metrics = {
            'hit_count': int(hit_count),
            'hit_rate': hit_rate,
            'bad_count': int(hit_bad),
            'good_count': int(hit_good),
            'badrate': badrate,
            'lift': lift,
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
        if amount is not None:
            amount_metrics = self._calculate_amount_metrics(
                mask, y, amount
            )
            metrics.update(amount_metrics)
        
        return metrics
    
    def _calculate_amount_metrics(
        self,
        mask: np.ndarray,
        y: pd.Series,
        amount: pd.Series
    ) -> Dict[str, float]:
        """计算金额口径指标.
        
        :param mask: 规则命中掩码
        :param y: 标签
        :param amount: 金额
        :return: 金额指标字典
        """
        total_amount = amount.sum()
        selected_amount = amount[mask].sum()
        
        total_bad_amount = amount[y == 1].sum()
        selected_bad_amount = amount[mask & (y == 1)].sum()
        
        loss_rate = selected_bad_amount / selected_amount if selected_amount > 0 else 0
        overall_loss_rate = total_bad_amount / total_amount if total_amount > 0 else 0
        loss_lift = loss_rate / overall_loss_rate if overall_loss_rate > 0 else 0
        
        return {
            'total_amount': total_amount,
            'selected_amount': selected_amount,
            'amount_hit_rate': selected_amount / total_amount if total_amount > 0 else 0,
            'loss_rate': loss_rate,
            'loss_lift': loss_lift
        }
    
    def _calculate_psi(self, expected: float, actual: float) -> float:
        """计算PSI（群体稳定性指数）.
        
        :param expected: 预期值
        :param actual: 实际值
        :return: PSI值
        """
        if expected <= 0 or actual <= 0:
            return 0.0
        
        return (actual - expected) * np.log(actual / expected)
    
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
        
        :param y_true: 真实标签
        :param y_score: 预测分数
        :return: KS值
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return np.max(tpr - fpr)
    
    def calculate_iv(
        self,
        feature: pd.Series,
        target: pd.Series,
        n_bins: int = 10
    ) -> float:
        """计算IV值（信息价值）.
        
        :param feature: 特征值
        :param target: 目标变量
        :param n_bins: 分箱数
        :return: IV值
        """
        from sklearn.preprocessing import KBinsDiscretizer
        
        # 分箱
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        
        try:
            binned = discretizer.fit_transform(
                feature.values.reshape(-1, 1)
            ).flatten()
        except Exception:
            return 0.0
        
        # 计算IV
        iv = 0.0
        total_good = (target == 0).sum()
        total_bad = (target == 1).sum()
        
        for bin_idx in range(n_bins):
            mask = binned == bin_idx
            if mask.sum() == 0:
                continue
            
            bin_good = (target[mask] == 0).sum()
            bin_bad = (target[mask] == 1).sum()
            
            good_dist = bin_good / total_good if total_good > 0 else 0
            bad_dist = bin_bad / total_bad if total_bad > 0 else 0
            
            if good_dist > 0 and bad_dist > 0:
                iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)
        
        return iv
    
    def calculate_gini(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算Gini系数.
        
        :param y_true: 真实标签
        :param y_score: 预测分数
        :return: Gini系数
        """
        auc = roc_auc_score(y_true, y_score)
        return 2 * auc - 1


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
    
    :param y_true: 真实标签
    :param y_score: 预测分数
    :param n_buckets: 分桶数
    :return: Lift数据DataFrame
    """
    # 按分数排序
    df = pd.DataFrame({
        'score': y_score,
        'target': y_true
    })
    df = df.sort_values('score', ascending=False)
    
    # 分桶
    df['bucket'] = pd.qcut(df.index, n_buckets, labels=False, duplicates='drop')
    
    # 计算每桶指标
    results = []
    total_bad_rate = y_true.mean()
    
    for bucket in range(n_buckets):
        bucket_data = df[df['bucket'] == bucket]
        if len(bucket_data) == 0:
            continue
        
        bad_rate = bucket_data['target'].mean()
        lift = bad_rate / total_bad_rate if total_bad_rate > 0 else 0
        
        results.append({
            'bucket': bucket + 1,
            'sample_count': len(bucket_data),
            'bad_rate': bad_rate,
            'lift': lift,
            'cumulative_samples': df[df['bucket'] <= bucket].shape[0],
            'cumulative_bad_rate': df[df['bucket'] <= bucket]['target'].mean()
        })
    
    return pd.DataFrame(results)
