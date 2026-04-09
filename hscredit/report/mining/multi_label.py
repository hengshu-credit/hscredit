"""多标签联合规则挖掘.

支持同时针对多个标签（如 MOB3@30 和 MOB6@30）挖掘规则，
并分析规则在不同标签下的有效性差异。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.base import BaseEstimator

from .base import BaseRuleMiner
from .single_feature import SingleFeatureRuleMiner
from .metrics import calculate_rule_metrics


class MultiLabelRuleMiner(BaseRuleMiner):
    """多标签规则挖掘器.

    支持同时针对多个标签挖掘规则，并分析规则在不同标签下的有效性差异。

    典型应用场景：
    - 长短期标签都有效的强规则（稳定拒绝规则）
    - 仅短期标签有效（可能是偶发风险，谨慎使用）
    - 仅长期标签有效（长期风险，可做预警规则）
    - 两标签均无效（噪声规则，丢弃）

    :param labels: 标签列名列表，如 ['mob3_30', 'mob6_30']
    :param label_names: 标签中文名列表，如 ['短期标签(MOB3@30)', '长期标签(MOB6@30)']
    :param min_support: 最小支持度（规则覆盖率），默认 0.02
    :param min_lift: 最小 LIFT 值，默认 1.5
    :param max_rules: 每个特征最大规则数，默认 10
    :param n_bins: 数值特征分箱数，默认 10
    :param exclude_cols: 需要排除的列名列表

    示例::

        >>> miner = MultiLabelRuleMiner(
        ...     labels=['mob3_30', 'mob6_30'],
        ...     label_names=['短期标签(MOB3@30)', '长期标签(MOB6@30)'],
        ...     min_support=0.02,
        ...     min_lift=1.5,
        ... )
        >>> miner.fit(df, features=['age', 'income', 'credit_score'])
        >>> rules = miner.get_rules(effectiveness='both')
        >>> report = miner.get_effectiveness_matrix()
    """

    def __init__(
        self,
        labels: List[str] = None,
        label_names: Optional[List[str]] = None,
        min_support: float = 0.02,
        min_lift: float = 1.5,
        max_rules: int = 10,
        n_bins: int = 10,
        exclude_cols: Optional[List[str]] = None,
    ):
        super().__init__(target=labels[0] if labels else 'target', exclude_cols=exclude_cols)
        self.labels = labels or []
        self.label_names = label_names or labels or []
        self.min_support = min_support
        self.min_lift = min_lift
        self.max_rules = max_rules
        self.n_bins = n_bins
        self._rules: List[Dict[str, Any]] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        features: Optional[List[str]] = None,
    ) -> 'MultiLabelRuleMiner':
        """拟合多标签规则挖掘器.

        :param X: 输入数据 DataFrame，须包含 labels 指定的标签列
        :param y: 忽略
        :param features: 需要挖掘的特征列表，为 None 时自动选择数值特征
        :return: self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X 必须为 DataFrame，且须包含标签列")

        df = X.copy()

        # 验证标签列存在
        missing_labels = [lb for lb in self.labels if lb not in df.columns]
        if missing_labels:
            raise ValueError(f"标签列缺失: {missing_labels}")

        # 确定特征列
        if features is None:
            exclude = set(self.labels) | set(self.exclude_cols)
            features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        self._rules = []

        # 对每个标签独立运行单特征规则挖掘
        label_rules = {}  # {label: {rule_expr: metrics_dict}}
        all_rule_exprs = set()

        for label in self.labels:
            miner = SingleFeatureRuleMiner(
                target=label,
                n_bins=self.n_bins,
                exclude_cols=[lb for lb in self.labels if lb != label] + self.exclude_cols,
            )
            miner.fit(df[features + [label]])
            rules = miner.get_rules() if hasattr(miner, 'get_rules') else miner.get_top_rules(top_n=self.max_rules * len(features))

            label_rules[label] = {}
            for rule in rules:
                expr = rule.get('规则', rule.get('rule', rule.get('expression', '')))
                if expr:
                    label_rules[label][expr] = rule
                    all_rule_exprs.add(expr)

        # 合并规则，为每条规则计算各标签的指标
        for expr in all_rule_exprs:
            try:
                mask = df.eval(expr)
            except Exception:
                continue

            n_match = mask.sum()
            support = n_match / len(df)
            if support < self.min_support:
                continue

            rule_info = {
                '规则': expr,
                '覆盖样本数': int(n_match),
                '覆盖率': round(support * 100, 2),
            }

            effective_labels = []
            for i, label in enumerate(self.labels):
                lname = self.label_names[i] if i < len(self.label_names) else label
                overall_rate = df[label].mean()
                rule_rate = df.loc[mask, label].mean() if n_match > 0 else 0
                lift = rule_rate / overall_rate if overall_rate > 0 else 0

                rule_info[f'{lname}_坏率'] = round(rule_rate * 100, 2)
                rule_info[f'{lname}_LIFT'] = round(lift, 4)
                rule_info[f'{lname}_有效'] = lift >= self.min_lift

                if lift >= self.min_lift:
                    effective_labels.append(lname)

            # 判断规则分类
            if len(effective_labels) == len(self.labels):
                rule_info['规则类型'] = '强规则（全标签有效）'
                rule_info['建议'] = '稳定拒绝规则'
            elif len(effective_labels) > 0:
                rule_info['规则类型'] = f'部分有效（{",".join(effective_labels)}）'
                rule_info['建议'] = '谨慎使用/预警规则'
            else:
                rule_info['规则类型'] = '无效规则'
                rule_info['建议'] = '放弃'

            self._rules.append(rule_info)

        # 按第一个标签的 LIFT 降序排序
        first_lift_col = f'{self.label_names[0]}_LIFT' if self.label_names else f'{self.labels[0]}_LIFT'
        self._rules.sort(key=lambda r: r.get(first_lift_col, 0), reverse=True)

        self._is_fitted = True
        return self

    def get_rules(
        self,
        effectiveness: str = 'any',
        min_lift_per_label: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取筛选后的规则表.

        :param effectiveness: 筛选条件
            - 'both': 所有标签均有效
            - 'any': 任一标签有效（默认）
            - 'short_only': 仅第一个标签有效
            - 'long_only': 仅最后一个标签有效
            - 'all': 不做筛选
        :param min_lift_per_label: 覆盖最小 LIFT 阈值，为 None 时使用 self.min_lift
        :param top_n: 返回前 N 条规则
        :return: 规则 DataFrame
        """
        if not self._is_fitted:
            raise ValueError("请先调用 fit()")

        min_lift = min_lift_per_label or self.min_lift
        rules = self._rules.copy()

        if effectiveness == 'both':
            rules = [r for r in rules if r.get('规则类型', '').startswith('强规则')]
        elif effectiveness == 'any':
            rules = [r for r in rules if r.get('规则类型', '') != '无效规则']
        elif effectiveness == 'short_only':
            first_eff_col = f'{self.label_names[0]}_有效' if self.label_names else f'{self.labels[0]}_有效'
            rules = [r for r in rules if r.get(first_eff_col, False)]
        elif effectiveness == 'long_only':
            last_eff_col = f'{self.label_names[-1]}_有效' if self.label_names else f'{self.labels[-1]}_有效'
            rules = [r for r in rules if r.get(last_eff_col, False)]

        if top_n:
            rules = rules[:top_n]

        return pd.DataFrame(rules)

    def get_effectiveness_matrix(self) -> pd.DataFrame:
        """规则有效性矩阵：行=规则，列=各标签，格=LIFT值.

        :return: 有效性矩阵 DataFrame
        """
        if not self._is_fitted:
            raise ValueError("请先调用 fit()")

        rows = []
        for rule in self._rules:
            row = {'规则': rule['规则']}
            for i, label in enumerate(self.labels):
                lname = self.label_names[i] if i < len(self.label_names) else label
                row[f'{lname}_LIFT'] = rule.get(f'{lname}_LIFT', 0)
            row['规则类型'] = rule.get('规则类型', '')
            rows.append(row)

        return pd.DataFrame(rows)

    def get_report(self) -> pd.DataFrame:
        """获取完整规则分析报告.

        :return: 含规则分类和业务解读的 DataFrame
        """
        return self.get_rules(effectiveness='all')
